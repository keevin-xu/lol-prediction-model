# LoL T2 Prediction Model — CLAUDE.md
*Two-person project. Read this before touching anything.*

---

## What This Project Does

Predicts win probabilities for League of Legends Tier 2 professional matches, then scans Polymarket for +EV betting opportunities and alerts via Discord.

**Current model performance (backtested on 8,000+ matches):**
- V1 ELO: 63.3% accuracy, 0.2256 Brier score
- V2 LogReg (ELO + recent form): 63.9% accuracy, 0.2222 Brier score

---

## Project Status

| Component | Status | Notes |
|---|---|---|
| OE scraper | ✅ Done | 10,372 T2 matches (2024–2026) in SQLite |
| TTP scraper | ✅ Done | 2,291 players with soloq ratings |
| Roster scraper | ✅ Done | 598 roster entries across 97 teams |
| V1 ELO model | ✅ Done | Regional offsets + recency decay |
| V2 LogReg model | ✅ Done | ELO + form features, +0.9% accuracy over V1 |
| Backtester | ✅ Done | Walk-forward eval + grid search optimizer |
| Platt calibration | ✅ Done | Fixes overconfident predictions at extremes |
| Polymarket scanner | ✅ Done | Scans every 5 min, 0 T2 match markets exist currently |
| Discord bot | ✅ Done | `/predict`, `/scan`, `/portfolio`, `/trades`, `/leaderboard`, `/status`, `/settle` |
| Paper trading | ✅ Done | Auto-bets on +EV signals, tracks P&L |

**Known limitation:** Polymarket has had exactly 1 LoL market ever (2020 Worlds winner). T2 head-to-head match markets don't exist on the platform yet. The bot will catch them if they appear.

---

## Directory Structure

```
lol-prediction-model/
├── CLAUDE.md                        ← you are here
├── data/
│   ├── raw/
│   │   ├── trackingthepros/         ← daily player soloq snapshots (JSON)
│   │   ├── oracleselixir/           ← annual match CSVs + gdrive_ids.json
│   │   └── rosters/                 ← daily roster snapshots (JSON)
│   └── processed/
│       └── unmatched_players.json   ← roster players not matched to TTP
├── scrapers/
│   ├── oe_scraper.py                ← Oracle's Elixir match data (Google Drive)
│   ├── ttp_scraper.py               ← TrackingThePros soloq data (DataTables API)
│   └── roster_scraper.py            ← Leaguepedia Cargo API rosters
├── model/
│   ├── soloq_rating.py              ← rank/LP → numeric rating + team aggregation
│   ├── pro_elo.py                   ← ELO engine with regional offsets + decay
│   ├── blend.py                     ← dynamic alpha blend of soloq + pro ELO
│   ├── predict.py                   ← win probability output + CLI
│   ├── features.py                  ← rolling team features from OE match data
│   ├── v2_model.py                  ← gradient boosting / logistic regression v2
│   ├── calibration.py               ← Platt scaling for probability calibration
│   ├── calibration_params.json      ← fitted calibration parameters
│   └── v2_model.pkl                 ← serialized v2 model
├── backtest/
│   └── backtest.py                  ← walk-forward backtester + grid search
├── polymarket/
│   ├── scanner.py                   ← find active LoL T2 markets on Polymarket
│   ├── edge.py                      ← compare model prob vs market prob
│   ├── paper_trader.py              ← paper trading position tracking + settlement
│   └── bot.py                       ← Discord bot (scan loop + slash commands)
├── db/
│   ├── schema.sql                   ← SQLite schema (7 tables)
│   ├── init_db.py                   ← creates/resets database
│   └── lol_model.db                 ← SQLite database (not in git)
├── docs/
│   ├── model_system_deep_dive.md    ← full math, data flow, worked examples
│   ├── context_handoff.md           ← partner handoff notes
│   └── scraper_roadmap.md           ← original scraper planning doc
├── requirements.txt
└── .env.example                     ← env vars (NEVER put secrets here)
```

---

## Data Sources

### 1. Oracle's Elixir — Pro Match Results
**Source:** Google Drive folder (NOT the old S3 bucket — that's dead)
**Scraper:** `scrapers/oe_scraper.py`
**Data:** 10,372 T2 matches across 18 leagues, 440 teams, 2024–2026

The datalisk.io API checks for updates; Google Drive file IDs are cached in `gdrive_ids.json`.

**T2 leagues (OE abbreviations):**
```
NACL, LCKC, EM, LEC, NLC, LFL, ESLOL, LVP SL, TCL, LCO,
LLA, LTA N, LTA S, CBLOL Academy, LRN, LRS, PCS, VCS, LJL
```

### 2. TrackingThePros — Solo Queue Ratings
**Source:** TTP DataTables API (`/d/list_players`, paginated at 200/page)
**Scraper:** `scrapers/ttp_scraper.py`
**Data:** 2,291 players, 1,161 with ranked soloq data

**Critical:** The rank parser regex in `parse_rank()` uses explicit division alternatives `(IV|III|II|I)` in that order. Do not simplify — an earlier version silently broke Challenger LP parsing.

### 3. Leaguepedia — Current Rosters
**Source:** Cargo API at `lol.fandom.com/api.php`
**Scraper:** `scrapers/roster_scraper.py`
**Data:** 598 roster entries, 97 teams

**Leaguepedia gotchas:**
- Aggressive rate limiting — scraper has 30s×N exponential backoff
- Field is `TournamentLevel = 'Secondary'` (NOT `Tier = '2'`)
- League names differ from OE (e.g., `"North American Challengers League"` not `"NACL"`)
- Uses POST requests with `User-Agent` header including contact email

---

## Model Architecture

### V1: ELO + SoloQ Blend (production)

```
accounts table → soloq_rating.py → player ratings
                                          ↓
rosters table ──→ pro_elo.py ──→ team ELO (with regional offsets + decay)
                       ↑                  ↓
matches table ─────────┘           blend.py → predict.py → P(win)
```

**Optimized parameters (from grid search over 100+ combos):**
- `K = 64` — ELO volatility (high for T2 instability)
- `blend_k = 5` — trust pro results over soloq faster
- `scale = 400` — ELO scale factor
- `half_life = 270 days` — ELO decays toward 1500 after inactivity

**Regional offsets (soloq-derived):**
```
KR: +126  |  EU: +120  |  CN: +89  |  JP: +38  |  NA: +36
VN: +23   |  BR: -16   |  TR: -23  |  PCS: -78 |  OCE: -260
```

### V2: Logistic Regression (experimental, +0.9% accuracy)

Adds rolling team features on top of ELO:
- Recent win rate (last 5/15 games)
- Average gold diff at 10/15 min
- KDA ratio
- Win/loss streak

Trained via walk-forward with expanding window. Model saved at `model/v2_model.pkl`.

---

## Discord Bot

**File:** `polymarket/bot.py`

**Requires in `.env`:**
```
DISCORD_BOT_TOKEN=your_token
DISCORD_CHANNEL_ID=your_channel_id
```

**Commands:**
| Command | What it does |
|---|---|
| `/predict <team_a> <team_b>` | Model prediction for a matchup |
| `/scan` | Force immediate Polymarket scan |
| `/portfolio` | Paper trading bankroll, P&L, win rate |
| `/trades` | Open positions + recent settled bets |
| `/settle` | Force check for resolved markets |
| `/leaderboard` | Top 20 teams by blended rating |
| `/status` | Bot uptime, scan count, bankroll |

**Background loops:**
- Scan loop: every 5 minutes, scans Polymarket for LoL T2 markets
- Settlement loop: every 1 hour, checks if open paper bets resolved

**Paper trading:** Starting bankroll $1,000. Auto-places bets on +EV signals (3%+ edge, Kelly-sized, 15% max position). Tracks P&L in `paper_trades` and `paper_portfolio` SQLite tables.

---

## Database Schema (SQLite)

| Table | Rows | Populated by |
|---|---|---|
| `players` | 2,291 | ttp_scraper.py |
| `accounts` | 2,300 | ttp_scraper.py (soloq_rating computed on insert) |
| `matches` | 10,372 | oe_scraper.py |
| `rosters` | 598 | roster_scraper.py |
| `teams` | 440 | pro_elo.py (ELO + games_played) |
| `paper_trades` | 0+ | paper_trader.py (paper bets) |
| `paper_portfolio` | 0+ | paper_trader.py (daily snapshots) |

---

## Dev Setup

```bash
git clone https://github.com/keevin-xu/lol-prediction-model.git
cd lol-prediction-model
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium

cp .env.example .env
# Fill in DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID

python db/init_db.py
```

---

## Running Everything

```bash
# Scrapers (run periodically)
python scrapers/oe_scraper.py         # match data (weekly)
python scrapers/ttp_scraper.py        # soloq snapshots (daily)
python scrapers/roster_scraper.py     # rosters (match days)

# Model pipeline (after scraper updates)
python model/soloq_rating.py          # recompute player ratings
python model/pro_elo.py               # rebuild team ELOs

# Predictions
python model/predict.py "Solary" "Karmine Corp"
python model/predict.py --list        # all teams ranked

# Backtesting
python backtest/backtest.py           # default params
python backtest/backtest.py --optimize  # grid search

# Discord bot (runs continuously)
python polymarket/bot.py
```

---

## Coding Conventions

- **Python 3.9** (system Python on macOS — use `Optional[X]` not `X | None`)
- `loguru` for all logging
- All scrapers save raw output to `data/raw/` before processing
- Retry with exponential backoff on all HTTP calls
- Time-based train/val splits only — never random
- Don't commit `data/`, `db/lol_model.db`, `.env`, or `*.pkl` files
- **NEVER put secrets in `.env.example`** — that file is tracked by git

---

## Key Decisions Log

| Decision | Choice | Reason |
|---|---|---|
| SoloQ source | TrackingThePros | Best T2 coverage, DataTables API |
| Match data | Oracle's Elixir (Google Drive) | S3 bucket is dead; GDrive is current |
| Roster source | Leaguepedia Cargo API | Best coverage but aggressive rate limits |
| DB | SQLite | Simple, sufficient for single-user |
| Browser automation | Playwright | Used for GDrive ID discovery + TTP fallback |
| ELO K-factor | 64 | Backtested optimal for T2 volatility |
| Blend denominator | 5 | Trust pro results quickly |
| Half-life decay | 270 days | Handles roster changes + inactive teams |
| V2 model | Logistic regression | GBM overfit; LogReg stays disciplined |
| Market scanner | Polymarket Gamma API | Public, no auth needed for reads |
| Discord notifications | discord.py bot | Full slash commands + background loops |

---

## Backtest Results (Latest)

```
V1 ELO:     63.3% accuracy  |  0.2256 Brier  |  0.6430 Log Loss
V2 LogReg:  63.9% accuracy  |  0.2222 Brier  |  0.6351 Log Loss
Random:     50.0% accuracy  |  0.2500 Brier  |  0.6931 Log Loss
```

**Calibration (V1, 80%+ range still overconfident):**
- 70-75% predicted → 71% actual ✓
- 80-85% predicted → 80% actual ✓
- 85-90% predicted → 80% actual ✗ (overconfident)
- 90%+ predicted → 80% actual ✗ (overconfident)

---

## What's NOT Working / Known Issues

1. **No Polymarket T2 match markets exist** — the scanner runs but finds nothing. May need to target traditional esports bookmakers instead.
2. **6 league mappings missing** from roster scraper: LCO, ESLOL, LEC, LTA N, LTA S, CBLOL Academy — Leaguepedia names unconfirmed.
3. **Only 29 of 440 teams have soloq baselines** — roster coverage is sparse.
4. **Cross-region ELO is imprecise** — soloq offsets help but teams from different leagues never play each other (except EM).
5. **Model overconfident above 85%** — Platt scaling helps slightly but doesn't fully fix it.
6. **Bot token was leaked** — the old token was committed to `.env.example` and GitHub revoked it. Always reset and use the new token in `.env` only.
