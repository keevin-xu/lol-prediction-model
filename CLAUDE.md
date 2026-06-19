# LoL T2 Prediction Model — CLAUDE.md
*Two-person project. Read this before touching anything.*

---

## What This Project Does

Predicts win probabilities for League of Legends Tier 2 professional matches and bets against Polymarket opening lines where the model has edge.

**The strategy:** Polymarket opening lines on T2 LoL are only 61% accurate. Our ELO model is 67% accurate. By betting at market open on same-region matches where the model disagrees by >10%, we capture edge that decays as sharp money arrives.

**Backtested P&L (177 trades, realistic liquidity model):**
```
Hit rate:     70.6% (CI: 63.8% – 77.4%)
ROI per bet:  +37.5% (CI: +23.4% – +51.4%)
P&L:          $1,000 → $3,288 (+229%)   |   $5,000 → $15,110 (+202%)
Max drawdown: 12.9%
Mean CLV:     +0.150 (line drifts toward our position 72% of the time)
```
Liquidity model: volume-dependent costs (3-8%), opening fillable capped at 1-3% of total volume.

**Adversarial validation:** Survives out-of-sample holdout (73.3% hit, +39.2% ROI), cost stress (+5%), threshold perturbation, and league-removal tests. CLV coherent (83.7% win rate when line confirms us). No time decay — Q2 2026 stronger than Q1.

---

## Project Status

| Component | Status | Notes |
|---|---|---|
| OE scraper | ✅ Done | 10,372 T2 matches (2024–2026) in SQLite |
| TTP scraper | ✅ Done | 2,291 players with soloq ratings |
| Roster scraper | ✅ Done | 598 roster entries across 97 teams |
| Riot API scraper | ✅ Done | Backfills soloq for unmatched roster players |
| V1 ELO model | ✅ Done | Regional offsets + decay + calibration |
| Platt calibration | ✅ Done | Wired into all predictions |
| Blue side offset | ✅ Done | +22.3 ELO for blue side (53.2% WR) |
| Cross-region detector | ✅ Done | Flags + adjusts international matchups |
| Bo3/Bo5/Bo7 conversion | ✅ Done | Negative binomial series probability |
| Polymarket scanner | ✅ Done | tag_slug discovery, finds 43+ LoL events |
| Price tracker | ✅ Done | 5-min snapshots, resolution tracking |
| Live engine | ✅ Done | Opening-line detector, signal, gates, paper exec |
| Discord bot | ✅ Done | Decision cards, health dashboard, alerts |
| Paper trading | ✅ Running | Waiting for same-region T2 markets |
| P&L backtester | ✅ Done | Realistic costs, Kelly sizing, trade log CSV |
| V3 features (107 cols) | ❌ Abandoned | 0% accuracy gain over V1 — archived |
| Draft features | ❌ Abandoned | 0% accuracy gain — archived |
| Live in-game model | ❌ No-go | Mid-game liquidity worse than open |

---

## The Shipped Strategy

**Rule (frozen, validated):**
- Same-region T2 LoL only
- Model disagrees with Polymarket opening price by >10%
- Bet at market open, before sharp money arrives
- Suppress: cross-region, edge 5-10% (costs eat it), roster changes post-creation

**Sizing:**
- Quarter-Kelly (6.25% max) on lower CI bound of edge
- Per-market cap: 2% of bankroll
- Depth-gated: capped at estimated fillable size
- Typical bet: $20-75 at current bankroll

**Deployment:**
- Paper-trading now (LIVE_TRADING=False in live_engine.py)
- Promote to live after 30+ paper bets show positive CLV (CI clears zero)

---

## Key Findings (June 18, 2026)

1. **Polymarket closing prices are 96% accurate** — but this is in-game contamination. Markets stay open during matches.
2. **Opening prices are only 61% accurate** — soft lines, thin liquidity, no sharp money yet.
3. **The model beats the opening line** — 67% vs 61% accuracy, Brier 0.2011 vs 0.2178.
4. **V3 model (107 features) gave 0% improvement** — dragon control, baron rates, vision, draft quality are all redundant with ELO. The accuracy ceiling from pre-match public data is ~64%.
5. **The edge is in WHEN you bet, not how good the model is.** A 64% model beats a 61% opening line.
6. **Live in-game betting is not viable** — mid-game T2 liquidity ($352) is worse than at open ($979).

---

## Directory Structure

```
lol-prediction-model/
├── CLAUDE.md                         ← you are here
├── model/
│   ├── predict.py                    ← prediction pipeline (ELO + calibration + adjustments)
│   ├── pro_elo.py                    ← ELO engine with regional offsets + decay
│   ├── blend.py                      ← dynamic alpha blend of soloq + pro ELO
│   ├── soloq_rating.py               ← rank/LP → numeric rating + team aggregation
│   ├── calibration.py                ← Platt scaling for probability calibration
│   └── calibration_params.json       ← fitted calibration parameters
├── polymarket/
│   ├── bot.py                        ← Discord bot (scan loop + slash commands)
│   ├── live_engine.py                ← opening-line detector + paper execution + CLV
│   ├── scanner.py                    ← Polymarket market discovery (tag_slug)
│   ├── price_tracker.py              ← forward price snapshots + resolution tracking
│   ├── edge.py                       ← edge calculator (quarter-Kelly)
│   ├── paper_trader.py               ← legacy paper trading (bot uses for /portfolio)
│   └── discord_cards.py              ← decision cards, health dashboard, alerts
├── backtest/
│   ├── backtest.py                   ← walk-forward backtester + grid search
│   ├── pnl_backtest.py               ← P&L backtester with realistic constraints
│   ├── market_comparison.py          ← model vs bookmaker vs Polymarket
│   └── manual_odds_entry.py          ← manual Pinnacle odds entry
├── scrapers/
│   ├── oe_scraper.py                 ← Oracle's Elixir match data (Google Drive)
│   ├── ttp_scraper.py                ← TrackingThePros soloq data
│   ├── roster_scraper.py             ← Leaguepedia Cargo API rosters
│   ├── riot_scraper.py               ← Riot API soloq backfill
│   └── team_matcher.py               ← shared fuzzy matching + alias DB
├── archive/                          ← abandoned code (preserved for history)
│   ├── features_v3.py                ← 107-feature extraction (0% gain)
│   ├── draft_features.py             ← champion/draft features (0% gain)
│   ├── v3_model_trainer.py           ← GBM trainer (abandoned)
│   └── odds_scraper.py               ← the-odds-api (no esports coverage)
├── db/
│   ├── schema.sql                    ← SQLite schema (16 tables)
│   ├── init_db.py                    ← creates/resets database
│   └── lol_model.db                  ← SQLite database (not in git)
├── data/
│   ├── raw/                          ← scraped data (not in git)
│   ├── backtest_trades.csv           ← 177-trade backtest log
│   └── upcoming_matches.csv          ← match tracking spreadsheet
├── docs/
│   ├── session_2026_06_18.md         ← full session report
│   ├── adversarial_and_feasibility.md ← adversarial validation + Step 3 no-go
│   ├── research_brief.md             ← research directions for beating markets
│   ├── audit_2026_06_18.md           ← codebase audit
│   ├── model_system_deep_dive.md     ← original model math docs
│   ├── context_handoff.md            ← partner handoff notes
│   └── scraper_roadmap.md            ← original planning doc
├── requirements.txt
└── .env.example
```

---

## Model Architecture (V1 ELO — the shipped model)

```
accounts → soloq_rating.py → player ratings
                                     ↓
rosters ──→ pro_elo.py ──→ team ELO (K=64, regional offsets, 270d decay)
                 ↑                   ↓
matches ─────────┘           blend.py (alpha = games / (games + 5))
                                     ↓
                              predict.py → P(win)
                                     ↓
                              + Platt calibration (a=0.910, b=0.045)
                              + Blue side offset (+22.3 ELO)
                              + Cross-region adjustment (80% of soloq offset)
                              + Bo series conversion (negative binomial)
```

**Parameters (frozen, grid-search optimized):**
- K = 64, blend_k = 5, scale = 400, half_life = 270 days
- Calibration: a=0.910, b=0.045
- Blue side: +22.3 ELO (from 53.2% blue WR across 10,372 matches)

---

## Polymarket Integration

**Scanner** discovers LoL markets via `tag_slug=league-of-legends` on the Gamma API. Parses team names from "LoL: X vs Y (BoN)" format. Filters resolved (100/0), per-game, and handicap markets.

**Price history** recovered from CLOB API: `GET /prices-history?market={token}&startTs=1&fidelity=10` returns 10-minute resolution for resolved markets.

**Live engine** runs every 5 minutes:
1. Detects new T2 moneyline markets
2. Computes model prob vs opening price
3. Gates: same-region, >10% edge, roster stability, liquidity
4. Sizes via quarter-Kelly (6.25%), 2% bankroll cap, depth-gated
5. Logs paper bet with entry price, edge, CLV tracking
6. On resolution: computes CLV vs pre-match close, realized P&L

---

## Discord Bot

**Commands:**
| Command | What it does |
|---|---|
| `/health` | **Edge health dashboard** — rolling CLV, promotion gate progress |
| `/predict <a> <b>` | Model prediction (with best_of, regions, calibration) |
| `/scan` | Force immediate Polymarket scan |
| `/portfolio` | Paper trading bankroll, P&L, win rate |
| `/trades` | Open positions + recent settled bets |
| `/settle` | Force check for resolved markets |
| `/leaderboard` | Top 20 teams by blended rating |
| `/status` | Bot uptime, scan count |
| `/reset` | Reset paper trading portfolio |

**Decision cards:** Every market evaluation posts a card showing the gate trace (✅/❌ for each gate) and whether a bet was placed or suppressed.

**Alerts:** CLV crossing zero, losing streak > 3, roster gate fires, data feed stale.

---

## Database Schema (SQLite — 16 tables)

| Table | Rows | Purpose |
|---|---|---|
| `players` | 2,291 | Pro player identities |
| `accounts` | 2,300 | Daily soloq snapshots |
| `teams` | 440 | Team ELO ratings |
| `matches` | 10,372 | Historical T2 match results |
| `rosters` | 598 | Current team rosters |
| `paper_trades` | 0+ | Legacy paper bets |
| `paper_portfolio` | 0+ | Legacy daily snapshots |
| `polymarket_markets` | 1+ | Market registry |
| `polymarket_prices` | 5+ | 5-min price snapshots |
| `bookmaker_odds` | 0 | Historical odds (no source yet) |
| `team_name_aliases` | 1+ | Fuzzy matching aliases |
| `live_signals` | 1+ | Detected market signals |
| `roster_checks` | 0+ | Roster stability audit |
| `live_bets` | 0+ | Paper/live bet log |
| `clv_log` | 0+ | CLV tracking per bet |

---

## Backtesting

**Walk-forward ELO backtest (8,888 matches):**
```
Accuracy:    63.5%
Brier:       0.2256
Best params: K=64, blend_k=5, scale=400, half_life=270d
```

**Opening-line P&L backtest (177 Polymarket trades, realistic liquidity):**
```
Strategy:    Same-region T2, >10% edge, bet at market open
Costs:       Volume-dependent (3% liquid → 8% thin markets)
Fillable:    1-3% of total volume at open ($20-$300 per bet)
Sizing:      Quarter-Kelly, 2% bankroll cap, depth-gated

Trades:      177 (125W / 52L)
Hit rate:    70.6% (CI: 63.8% – 77.4%)
ROI/bet:     +37.5% (CI: +23.4% – +51.4%)
P&L ($1K):   $1,000 → $3,288 (+229%)
P&L ($5K):   $5,000 → $15,110 (+202%)
Max DD:      12.9%
Max streak:  3 losses
CLV:         +0.150 (72% beat pre-match close)
```

**Trade logs:** `data/backtest_trades.csv` (default $1K), `data/backtest_trades_5000.csv` ($5K).
Generate for any bankroll: `python backtest/polymarket_backtest.py --bankroll 10000`

---

## Running Everything

```bash
# Scrapers (run periodically)
python scrapers/oe_scraper.py         # match data (weekly)
python scrapers/ttp_scraper.py        # soloq snapshots (daily)
python scrapers/roster_scraper.py     # rosters (match days)

# Model pipeline
python model/soloq_rating.py          # recompute player ratings
python model/pro_elo.py               # rebuild team ELOs

# Predictions
python model/predict.py "Solary" "Galions"
python model/predict.py "Solary" "Galions" --bo 5 --side blue

# Backtesting
python backtest/backtest.py --optimize                 # ELO grid search
python backtest/polymarket_backtest.py                 # P&L vs real Polymarket ($1K)
python backtest/polymarket_backtest.py --bankroll 5000 # P&L with $5K (own CSV)
python backtest/polymarket_backtest.py --bankroll 10000 # P&L with $10K
python backtest/pnl_backtest.py --sweep                # simulated market sweep

# Live engine
python polymarket/live_engine.py --status
python polymarket/live_engine.py --cycle

# Discord bot (runs continuously)
python polymarket/bot.py
```

---

## Coding Conventions

- **Python 3.9** (use `Optional[X]` not `X | None`)
- `loguru` for all logging
- Walk-forward only — never random splits, never lookahead
- Don't commit `data/`, `db/lol_model.db`, `.env`, or `*.pkl`
- **NEVER put secrets in `.env.example`**
- **Do not re-tune the model or thresholds** — the rule is frozen and validated

---

## Key Decisions Log

| Decision | Choice | Reason |
|---|---|---|
| Strategy | Bet opening lines | Opening 61% accurate, model 67% — edge exists at open |
| Edge threshold | >10% | Plateau from adversarial sweep; 5-10% eaten by costs |
| Region filter | Same-region only | Cross-region: 43% hit, negative ROI |
| Sizing | Quarter-Kelly | Conservative for thin markets + unvalidated live data |
| V3 features (107) | Abandoned | 0% accuracy gain — dragon/baron/vision/draft all redundant with ELO |
| Live in-game | No-go | Mid-game liquidity ($352) worse than open ($979) |
| Deployment | Paper first | 30+ bets with positive CLV before real capital |

---

## What's NOT Working / Known Issues

1. **Roster gate is partial** — checks roster exists but doesn't diff pre/post market creation timestamps.
2. **Only 29 of 440 teams have soloq baselines** — roster coverage is sparse.
3. **6 league mappings missing** from roster scraper: LCO, ESLOL, LEC, LTA N, LTA S, CBLOL Academy.
4. **Galions underestimation** — 5 of top 10 confident losses. ELO lags for teams on hot streaks.
5. **ELO-staleness suppressor not implemented** — flagged as recommended guardrail.
6. **No historical bookmaker odds source** — the-odds-api doesn't cover esports; OddsPortal blocks bots.
