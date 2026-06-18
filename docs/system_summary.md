# LoL T2 Prediction Model — Complete System Summary

*For humans and AI agents. Last updated: 2026-06-18.*

---

## Purpose

This system predicts win probabilities for **League of Legends Tier 2** (academy/secondary league) professional matches. The long-term goal is to detect +EV betting opportunities on Polymarket and alert via Discord, but no T2 match markets currently exist on the platform.

**Performance:** V1 ELO model achieves 63.3% accuracy (Brier 0.2256). V2 LogReg with form features reaches 63.9% accuracy (Brier 0.2222). Random baseline is 50%/0.2500.

---

## Architecture Overview

The system has three layers that run as a pipeline:

```
DATA INGESTION          STRENGTH COMPUTATION         PREDICTION
─────────────          ────────────────────         ──────────
OE Scraper ──┐                                     
             ├──→ SQLite ──→ SoloQ Ratings ─┐
TTP Scraper ─┤              Pro ELO Engine ──┼──→ Blend ──→ Win Probability
             │                               │
Roster Scraper ┘         Rolling Features ───┘         V2 Model + Calibration
Riot Scraper ──┘
```

All components are Python 3.9 scripts that read/write a single **SQLite database** (`db/lol_model.db`). There is no web server, no API layer, no microservices — just scripts run in sequence.

---

## Layer 1: Data Ingestion (Scrapers)

Four scrapers populate the database. They should be run periodically in this order:

### 1.1 Oracle's Elixir Match Scraper — `scrapers/oe_scraper.py`

**What:** Downloads annual CSV files of pro match data from Google Drive (Oracle's Elixir dataset), filters to T2 leagues only, and inserts match results into the `matches` table.

**How it works:**
- Checks the Datalisk API for CSV update timestamps
- Google Drive file IDs are cached in `data/raw/oracleselixir/gdrive_ids.json`; if missing, Playwright discovers them by browsing the GDrive folder
- Historical-year CSVs are cached permanently; current-year CSV refreshes if >7 days old
- Each OE CSV has one row per player per game (10 rows per game, 12 rows counting team summaries). The scraper collapses the two team-summary rows into a single match record (blue_team, red_team, winner)
- Uses INSERT OR IGNORE on gameid to avoid duplicates

**Writes:** `matches` (gameid, date, league, patch, playoffs, blue_team, red_team, winner, gamelength)
**External APIs:** Google Drive (public, no auth), Datalisk API (public)
**Frequency:** Weekly

**T2 leagues tracked (18):** NACL, LCKC, EM, LEC, NLC, LFL, ESLOL, LVP SL, TCL, LCO, LLA, LTA N, LTA S, CBLOL Academy, LRN, LRS, PCS, VCS, LJL

### 1.2 TrackingThePros Scraper — `scrapers/ttp_scraper.py`

**What:** Fetches solo queue rank data for ~2,300 pro players from TTP's DataTables API and stores daily snapshots.

**How it works:**
- Paginates TTP's `/d/list_players` endpoint (200 players per page, ~12 pages)
- For each player, parses rank strings like "Diamond I 72LP" or "Challenger 3,744LP" into a numeric `soloq_rating`
- Rating formula:
  - Below Master: `tier_base + division_offset + LP` (e.g., Diamond I 72LP = 2400 + 300 + 72 = 2772)
  - Master+: `2800 + 400 * log(1 + LP/400)` (log-compressed to prevent Challenger outliers from dominating)
- Upserts into `players` (identity) and `accounts` (daily snapshot with rating)

**Writes:** `players` (player_name, role, team, region), `accounts` (soloq_rating, rank_tier, lp, snapshot_date)
**External APIs:** TrackingThePros DataTables API (public, no auth)
**Frequency:** Daily

**Critical implementation detail:** The rank parser regex uses explicit division alternatives `(IV|III|II|I)` in that order. An earlier version that used a simpler pattern silently broke Challenger LP parsing. Do not simplify.

### 1.3 Roster Scraper — `scrapers/roster_scraper.py`

**What:** Fetches current T2 team rosters from Leaguepedia's Cargo API, fuzzy-matches player names against TTP data, and stores roster snapshots.

**How it works:**
- Queries Leaguepedia for all 2026 Secondary-level tournaments, picks the most recent non-playoff tournament per league
- Fetches roster data via batched Cargo queries
- Fuzzy-matches each player name against the `players` table (from TTP) to link identities
  - Length-aware thresholds: ≤4 chars = exact match, 5-6 chars = 90% similarity, ≥7 chars = 85%
  - Unmatched players saved to `data/processed/unmatched_players.json`
- Deletes old roster snapshot for each team, inserts fresh snapshot

**Writes:** `rosters` (team, player_name, role, snapshot_date, tournament)
**Reads:** `players` (for fuzzy matching)
**External APIs:** Leaguepedia Cargo API (aggressive rate limiting — 2s delay between calls, exponential backoff on failures)
**Frequency:** Match days

**Gotchas:**
- Leaguepedia uses `TournamentLevel = 'Secondary'` (NOT `Tier = '2'`)
- League names differ from OE abbreviations (e.g., "North American Challengers League" not "NACL")
- 6 league mappings are still missing: LCO, ESLOL, LEC, LTA N, LTA S, CBLOL Academy

### 1.4 Riot API Scraper — `scrapers/riot_scraper.py`

**What:** Backfills solo queue ratings for roster players who couldn't be matched to TTP. Uses Leaguepedia's SoloqueueIds field to resolve Riot IDs, then queries the Riot API for current rank.

**How it works:**
- Finds roster players with no match in `players` table
- Queries Leaguepedia for their SoloqueueIds (markup like "NA: AoJune#3354")
- Resolves each Riot ID to a PUUID via account-v1 API
- Fetches RANKED_SOLO_5x5 tier/LP via league-v4 API
- Implements sliding-window rate limiting (20 req/1s AND 100 req/120s per routing value)

**Writes:** `players`, `accounts` (new rows for previously unmatched players)
**External APIs:** Riot API (requires `RIOT_API_KEY` in `.env`), Leaguepedia Cargo API
**Frequency:** After roster scraper

---

## Layer 2: Strength Computation (Model Core)

Three modules transform raw data into team strength ratings.

### 2.1 SoloQ Rating Module — `model/soloq_rating.py`

**What:** Computes per-player numeric ratings from their rank data and aggregates to team level.

**Key functions:**
- `compute_all_ratings()` — Batch recomputes `accounts.soloq_rating` from rank_tier + LP (same formula as TTP scraper)
- `get_player_rating(player_name)` — Returns the best soloq_rating across all of a player's accounts
- `get_team_player_ratings(team_name)` — Looks up the current roster, returns `{role: best_rating}` for each role

**Reads:** `rosters`, `players`, `accounts`
**Writes:** `accounts.soloq_rating` (recompute only)

### 2.2 Pro ELO Engine — `model/pro_elo.py`

**What:** Computes team ELO ratings from match history, optionally seeded with soloq baselines. This is the core rating engine.

**How it works — two phases:**

**Phase 1: SoloQ Baseline (for teams with roster data)**
- For each team, fetches roster → per-role soloq ratings → weighted average
  - Role weights: Top 0.20, Jungle 0.22, Mid 0.23, Bot 0.20, Support 0.15
  - Requires ≥3 roles with data to produce a baseline
- Z-score normalizes all team soloq averages onto a 1500-centered ELO scale
- Only 29 of 440 teams have enough soloq data for a baseline

**Phase 2: ELO from Match History**
- Processes all matches chronologically (sorted by date)
- Teams with a soloq baseline start at that ELO; others start at 1500
- For each match: `expected = 1 / (1 + 10^((elo_b - elo_a) / 400))`, then `elo_new = elo_old + K * (actual - expected)` where K=32
- Writes final ELO + games_played to `teams` table

**Reads:** `rosters`, `players`, `accounts`, `matches`
**Writes:** `teams` (team_name, pro_elo, games_played, league, region)

**Key parameters:** K=32 (ELO volatility), DEFAULT_ELO=1500, MIN_ROLES_FOR_BASELINE=3

### 2.3 Dynamic Blending — `model/blend.py`

**What:** Blends pro ELO and soloq baseline using an alpha that increases with match count. Does NOT write to DB — computed on-the-fly at prediction time.

**Formula:**
```
alpha = games_played / (games_played + blend_k)
blended = alpha * pro_elo + (1 - alpha) * soloq_elo
```

- 0 games → alpha=0 → 100% soloq (brand new team)
- blend_k games → alpha=0.5 → 50/50
- 3×blend_k games → alpha=0.75 → mostly pro ELO

**Default blend_k=10.** The backtester found 5 optimal (CLAUDE.md says 5), but code defaults to 10.

**Reads:** `teams` (pro_elo, games_played), soloq baselines from pro_elo.py (cached)
**Writes:** Nothing

---

## Layer 3: Prediction

### 3.1 Win Probability — `model/predict.py`

**What:** Given two team names, outputs P(team_a wins) using blended ratings.

**Formula:** `P(A wins) = 1 / (1 + 10^(-(rating_a - rating_b) / scale))` where scale=400

**CLI usage:**
```bash
python model/predict.py "Solary" "Karmine Corp"   # single matchup
python model/predict.py --list                     # all teams ranked by blended rating
```

**Reads:** `teams` via blend.py

### 3.2 Rolling Features — `model/features.py`

**What:** Builds a feature dataset from OE match CSVs for the V2 model. Extracts rolling team statistics using a 15-game sliding window.

**Features per team (12):**
- `win_rate` (all games in window), `win_rate_last5` (most recent 5)
- `streak` (current consecutive W/L run, positive = wins)
- `avg_kills`, `avg_deaths`, `kda` (kills+assists / deaths)
- `avg_gamelength`
- `avg_gd10`, `avg_gd15` (gold differential at 10/15 minutes)
- `fb_rate`, `ft_rate` (first blood / first tower conversion %)

**Output:** A DataFrame with 29 columns per match:
- 12 blue_* features + 12 red_* features + 5 differential features (wr_diff, kda_diff, gd15_diff, gd10_diff, streak_diff) + target `result` (1=blue wins)

**Critical:** Features are extracted BEFORE each game is played (no lookahead). Walk-forward only.

**Reads:** OE CSVs from `data/raw/oracleselixir/` (not from DB)
**Writes:** Nothing (returns DataFrame)

### 3.3 V2 Model — `model/v2_model.py`

**What:** Trains a gradient-boosted classifier (HistGradientBoostingClassifier) on the feature dataset with walk-forward evaluation.

**Training:**
- Walk-forward with expanding window: warmup on first 15% of data (or min 200 rows), then predict forward, retrain every 500 matches
- Produces both V2 predictions and V1 ELO baseline predictions for comparison
- Model spec: max_iter=200, max_depth=4, learning_rate=0.05, min_samples_leaf=20

**Note:** Despite CLAUDE.md saying V2 uses logistic regression, the code uses `HistGradientBoostingClassifier`. The CLAUDE.md key decisions log says "GBM overfit; LogReg stays disciplined" — the code may have been updated or this is the experimental GBM version.

**Persistence:** Saves trained model to `model/v2_model.pkl`
**Reads:** Feature DataFrame from features.py, backtest module (imports `ELOTracker`, `load_matches`, `WARMUP_MONTHS` from `backtest.backtest` — which does not exist yet)

### 3.4 Platt Calibration — `model/calibration.py`

**What:** Corrects overconfident predictions. The V1 model says 87% but teams only win ~80% of the time at that confidence level. Platt scaling fits a logistic regression on the model's own predictions to shrink extremes.

**Formula:** `P_calibrated = 1 / (1 + exp(A * logit(P_raw) + B))`

**Persistence:** Parameters saved to `model/calibration_params.json` (contains `a`, `b`, `fitted` flag)

---

## Database Schema

SQLite at `db/lol_model.db`, initialized by `db/init_db.py` which runs `db/schema.sql`.

| Table | Purpose | Rows | Populated By | Key Columns |
|-------|---------|------|-------------|-------------|
| `players` | Pro player identities | 2,291 | ttp_scraper, riot_scraper | player_name (UNIQUE), role, team, region |
| `accounts` | Daily soloq snapshots | 2,300+ | ttp_scraper, riot_scraper, soloq_rating | player_id (FK), rank_tier, lp, soloq_rating, snapshot_date |
| `teams` | Team strength ratings | 440 | pro_elo.py | team_name (UNIQUE), pro_elo, games_played, league |
| `matches` | Historical T2 match results | 10,372 | oe_scraper | gameid (UNIQUE), date, league, blue_team, red_team, winner |
| `rosters` | Current team rosters | 598 | roster_scraper | team, player_name, role, snapshot_date |

**Indexes:** date and league on matches, player_id and snapshot_date on accounts, team+snapshot_date on rosters.

**No paper_trades or paper_portfolio tables exist yet** — those are listed in CLAUDE.md but the polymarket/ directory hasn't been created.

---

## What Exists vs. What Doesn't

### Exists and works:
- All 4 scrapers (OE, TTP, roster, Riot)
- Full model pipeline (soloq_rating → pro_elo → blend → predict)
- Feature engineering (features.py)
- V2 model training (v2_model.py) — though it imports from a missing `backtest` module
- Platt calibration (calibration.py)
- Database schema and initialization

### Listed in CLAUDE.md as "Done" but directories/files don't exist:
- `backtest/backtest.py` — Walk-forward backtester & grid search optimizer
- `polymarket/scanner.py` — Polymarket market scanner
- `polymarket/edge.py` — Edge calculator
- `polymarket/paper_trader.py` — Paper trading engine
- `polymarket/bot.py` — Discord bot with slash commands

These may have existed in a previous branch or been planned but not merged to main.

---

## Execution Pipeline

Run in this order for a full refresh:

```bash
# 1. Ingest data
python scrapers/oe_scraper.py          # fetch match CSVs (weekly)
python scrapers/ttp_scraper.py         # fetch soloq snapshots (daily)
python scrapers/roster_scraper.py      # fetch rosters (match days)
python scrapers/riot_scraper.py        # backfill unmatched players (after rosters)

# 2. Compute ratings
python model/soloq_rating.py           # recompute player ratings
python model/pro_elo.py                # rebuild team ELOs from all matches

# 3. Predict
python model/predict.py --list         # ranked leaderboard
python model/predict.py "Team A" "Team B"  # single matchup
```

---

## Key Formulas Reference

**SoloQ rating (below Master):**
`rating = tier_base + division_offset + LP`
(Iron IV = 0, Bronze IV = 400, Silver IV = 800, Gold IV = 1200, Platinum IV = 1600, Emerald IV = 2000, Diamond IV = 2400)

**SoloQ rating (Master+):**
`rating = 2800 + 400 * log(1 + LP / 400)`

**Team soloq baseline:**
`weighted_avg = Σ(role_weight × player_rating)` then z-score normalize to mean=1500

**ELO expected score:**
`E = 1 / (1 + 10^((elo_b - elo_a) / 400))`

**ELO update:**
`elo_new = elo_old + K × (actual - expected)` where K=32

**Blend:**
`alpha = games / (games + blend_k)`, `blended = alpha × pro_elo + (1-alpha) × soloq_elo`

**Win probability:**
`P(A) = 1 / (1 + 10^(-(rating_a - rating_b) / 400))`

---

## Known Issues

1. **No Polymarket T2 match markets exist.** The scanner concept is valid but there's nothing to scan.
2. **Only 29/440 teams have soloq baselines** — sparse roster-to-TTP matching means most teams start at default 1500 ELO.
3. **6 league mappings missing** in roster scraper (LCO, ESLOL, LEC, LTA N, LTA S, CBLOL Academy).
4. **Model overconfident above 85%** — Platt scaling helps but doesn't fully fix it.
5. **Cross-region ELO is imprecise** — T2 teams from different regions rarely/never play each other.
6. **v2_model.py imports from backtest.backtest which doesn't exist** — V2 training pipeline is broken without it.
7. **CLAUDE.md overstates completion** — backtest/, polymarket/, and Discord bot are listed as "Done" but aren't on the main branch.

---

## Configuration & Environment

- **Python 3.9** (use `Optional[X]` not `X | None`)
- **Logging:** loguru everywhere
- **Dependencies:** see `requirements.txt`
- **Secrets:** `.env` file (not tracked) with `DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID`, `RIOT_API_KEY`
- **Database:** SQLite at `db/lol_model.db` (not tracked in git)
- **Raw data:** `data/raw/` (not tracked in git)
