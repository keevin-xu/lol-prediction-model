# LoL T2 Prediction Model — CLAUDE.md
*Two-person project. Read this before touching anything.*

---

## What This Project Does

Predicts win probabilities for League of Legends Tier 2 professional matches and bets against Polymarket opening lines where the model has edge.

**The strategy:** Polymarket opening lines on T2 LoL are only 54% accurate. Our V2 ELO model is ~62-64% accurate (walk-forward, no lookahead). By betting at market open on same-region matches where the model disagrees by >10%, we capture edge that decays as sharp money arrives. The edge ONLY exists at market open — by 2-4 hours before the match, the line has already corrected to 66-67% accuracy and the model loses money betting against it.

**Backtested P&L (220 trades, V2 model, walk-forward ELOs, realistic liquidity):**
```
Hit rate:     62.3%
ROI per bet:  +110% cumulative at $1K quarter-Kelly (backtest, one 4-month window)
P&L:          $1,000 → $2,104   |   $5,000 → $9,500
Max drawdown: 16.8%
Mean CLV:     +0.121 (line drifts toward our position 65% of the time)
```
Walk-forward: model uses only matches before each market's date. No lookahead.
Liquidity model: volume-dependent costs (3-8%), opening fillable capped at 1-3% of total volume.

**IMPORTANT: These P&L numbers are backtested, not realized.** The cost model (estimate_opening_cost, estimate_fillable_at_open) has never been validated against a real fill. At 2x modeled costs, ROI drops to +51%. Forward paper trading with logged fills is the next validation step.

---

## Project Status

| Component | Status | Notes |
|---|---|---|
| OE scraper | ✅ Done | 20,587 T2 matches (2021–2026) in SQLite |
| TTP scraper | ✅ Done | 2,291 players with soloq ratings |
| Roster scraper | ✅ Done | 598 roster entries across 97 teams |
| Riot API scraper | ✅ Done | Backfills soloq for unmatched roster players |
| V2 ELO model | ✅ Done | MOV scaling, no soloq, identity calibration |
| Platt calibration | ✅ Identity | V2 is self-calibrated; Platt adds nothing measurable |
| Blue side offset | ✅ Done | +22.3 ELO for blue side (53.2% WR) |
| Cross-region detector | ✅ Done | Flags + adjusts international matchups |
| Bo3/Bo5/Bo7 conversion | ✅ Done | Negative binomial series probability |
| Polymarket scanner | ✅ Done | tag_slug discovery, finds 43+ LoL events |
| Price tracker | ✅ Done | 5-min snapshots, resolution tracking |
| Live engine | ✅ Done | Opening-line detector, signal, gates, paper exec |
| Discord bot | ✅ Done | Alerts only on placed bets (not every scan) |
| Paper trading | ✅ Running | Needs V2 params update in live_engine.py |
| P&L backtester | ✅ Done | Walk-forward ELOs, realistic costs, frozen dataset |
| T1 backtester | ✅ Done | Monte Carlo sim for T1 feasibility analysis |
| Metric audit (2G) | ✅ Done | Removed lookahead bias, fixed blend_k, CLV anchor |
| Sizing analysis | ✅ Done | Quarter-Kelly conservative; Half-Kelly after live validation |
| Team matcher | ✅ Improved | Suffix expansion + aliases recover 105 lost Polymarket markets |
| Frozen PM dataset | ✅ Done | 662 markets with full price histories in data/polymarket_resolved.json |
| V3 features (107 cols) | ❌ Abandoned | 0% accuracy gain over V1 — archived |
| Draft ELO offsets | ❌ Dead | 3 signals tested rigorously, all null on 16K games (June 2026) |
| Roster gate | ❌ Dead | Model MORE accurate on roster-changed games (+3.7%) |
| Line movement filter | ❌ Shelved | n=28 eligible test markets, instrument can't resolve |
| Patch-freshness sizing | ❌ Dead | Zero P&L impact, identical hit rates across groups |
| Live in-game model | ❌ No-go | Mid-game liquidity worse than open |
| T1 model (LEC/LCK/LPL) | ❌ Not viable yet | Model 62.4% on LEC, line already 66-75% by 2-4h pre-match |

---

## The Shipped Strategy

**Rule (frozen, validated):**
- Same-region T2 LoL only
- Model disagrees with Polymarket opening price by >10%
- Bet at market open, before sharp money arrives
- Suppress: cross-region, edge 5-10% (costs eat it)

**Model (V2 — shipped June 2026):**
- K=32, blend_k=5, scale=400, half_life=270d
- MOV scaling: mov_weight=1.5 (big wins → bigger ELO gain, close wins → smaller)
- No soloq baselines (removed — slight negative with only 38/440 teams covered)
- Identity calibration: a=1.0, b=0.0 (V2 is self-calibrated, Platt adds nothing)

**Sizing:**
- Quarter-Kelly (6.25% max) — stay here until cost model is validated by live fills
- Per-market cap: 2% of bankroll
- Depth-gated: capped at estimated fillable size
- Do NOT move to Half-Kelly until 20+ forward trades confirm cost model accuracy
- Size as if forward CLV is +0.08, not the backtest +0.12

**Deployment:**
- Paper-trading now (LIVE_TRADING=False in live_engine.py)
- Promote to live after 30+ paper bets show positive CLV AND realized fills track the cost model
- The untested risk is execution, not prediction — cost model has never been validated against a real fill

---

## Key Findings

### June 18, 2026 — Model & Market Discovery
1. **Polymarket closing prices are 96% accurate** — but this is in-game contamination. Markets stay open during matches.
2. **Opening prices are only 54% accurate on T2** — soft lines, thin liquidity, no sharp money yet.
3. **The model beats the opening line** — 65.3% vs 54% accuracy (walk-forward, no lookahead).
4. **V3 model (107 features) gave 0% improvement** — dragon control, baron rates, vision, draft quality are all redundant with ELO. The accuracy ceiling from pre-match public data is ~64%.
5. **The edge is in WHEN you bet, not how good the model is.** A 63-65% model beats a 54% opening line.
6. **Live in-game betting is not viable** — mid-game T2 liquidity ($352) is worse than at open ($979).

### June 19, 2026 — Metric Audit & Lookahead Fix (Step 2G)
7. **Previous backtest had lookahead bias** — `predict_match()` used final ELOs from DB, not walk-forward. This inflated hit rate from 65.3% to 72% and ROI from +25% to +40%. Fixed by integrating walk-forward ELO tracker into Polymarket backtest.
8. **blend_k was wrong** — `predict_match` defaulted to blend_k=10 but grid search optimized blend_k=5. Fixed.
9. **Live engine CLV was contaminated** — used last-ever price snapshot instead of pre-match-start. Fixed to anchor on match_start_ts.
10. **36 NACL markets were missed** — keyword "nacl" didn't match "North American Challengers League" in Polymarket titles. Fixed.
11. **Half-Kelly is optimal sizing** — triples return vs quarter-Kelly (+224% vs +91% at $1K) with manageable 24% max drawdown.
12. **Capacity ceiling is $5-10K** — T2 market depth caps fillable at $20-500/bet. Above $10K bankroll, increasing Kelly fraction barely helps.

### June 19-20, 2026 — T1 Feasibility & Timing Analysis
13. **T1 opening lines are even softer than T2** — 52% accurate (coinflip) vs T2's 54%.
14. **But T1 lines correct fast** — by 2-4h before match, T1 lines are 68-75% accurate. The model (62.4% on LEC) can't beat a line that's already smarter than it.
15. **T2 edge ONLY exists at market open** — tested model vs line at open, 4h, 2h, 1h before match. Profitable only at open (+25% ROI). At 4h/2h/1h, model loses money (-23%, -8%, -8% ROI) because the line is already 66-67% accurate.
16. **T1 would need ~70% model accuracy** to profitably bet 2-4h before match. Current model gets 62-64%. Needs T1-specific match data (LCK, LPL, LCS not in DB).
17. **T1 has 50x more liquidity** — $713K avg volume vs $15K for T2. If a 70% model is achievable, T1 capacity could support $50K+ bankroll.

### June 28-29, 2026 — V2 Model, Signal Investigation, Execution Audit
18. **MOV scaling works (+0.5% accuracy)** — margin-of-victory K adjustment (K=32, mov_weight=1.5) captures strength info binary win/loss ELO discards. The only feature improvement that survived testing.
19. **Soloq baselines removed** — slight negative with only 38/440 teams covered. Model is better without them.
20. **Draft features are dead** — 3 signals (champion-role WR with Bayes shrinkage, player-champion mastery, champion pool depth) tested as ELO offsets on 20K games. All null on 16K tuning games. Pool depth spiked on 2K holdout (p=0.003) but year-by-year decomposition showed no trend — coin-flip flip accuracy in 2023-2024, isolated spike in 2026. Noise, not signal.
21. **Roster gate is counterproductive** — model is 67.3% accurate on roster-changed games vs 63.5% on stable rosters. T2 changes are mostly routine subs on losing teams. Suppressing these would discard correct predictions.
22. **V2 model is self-calibrated** — tested 4 Platt calibrations on 210 out-of-sample Polymarket trades. Identity (a=1.0, no correction) had 1.0% calibration error. All-data refit (a=1.06) looked great in-sample but was worst on actual trades (4.8% error). Platt `a` swings 0.10 across fitting eras — too unstable to tune.
23. **Edge is not correlated with thin depth** — r=-0.006 between edge and fillable depth across 220 trades. The +0.12 CLV is not an artifact of unfillable markets.
24. **Cost model is the untested load-bearing assumption** — P&L at 1x costs: +110%. At 2x costs: +51%. The cost model has never been validated against a real fill. This is the remaining risk.
25. **Team matcher improved** — suffix expansion + DB aliases recovered 105 of 190 lost Polymarket markets. Eligible set grew from 176 to 220 trades.
26. **662 Polymarket markets persisted** — frozen dataset with full 10-min price histories. All verified as point-in-time tradeable (price history starts within ~20 min of market creation, no backfill).

### June 29, 2026 — T1 Real-Model Deep Dive (replaces Monte Carlo estimates)
27. **The existing T1 P&L numbers (162 trades, +0.160 CLV, 64.2% hit) came from a Monte Carlo simulator, not the real model** — `backtest/t1_backtest.py` assigns a fixed 0.65 probability to "the model's side" rather than running `ELOTracker`. Per-league accuracy numbers were already correct (confirmed below), but the trade-level P&L/CLV figures were never produced by the actual walk-forward model. New script `backtest/t1_model_backtest.py` runs the real V2 `ELOTracker` against the real frozen T1 price histories and reproduces close-but-not-identical numbers (153 trades vs 162, +0.169 CLV vs +0.160) — directionally consistent, but the existing report's T1 trade table should be considered illustrative, not the audited figure.
28. **Per-league accuracy reproduced exactly from CLAUDE.md** — LCK 67.7% (n=1,386), LEC 64.3% (n=846), LCS 62.8% (n=349), LPL 61.6% (n=1,975), all walk-forward, eval 2024-01-01+. This part of the existing T1 claims is solid.
29. **Fine-grained timing decay (10 points between open and +24h, real model + real prices) shows the edge crosses to negative earlier than the coarse 6-point table implied.** ROI/bet (not just CLV) is the metric that matters: positive and CI-clear from open through ~+3h (+27% to +39%), CI straddles zero from +6h to +18h, and the point estimate is already negative by +24h (-5.4%, 95% CI [-30%, +22%]). The old coarse table's +24h point was reported as "CLV nearly zero" (+0.008) — the real-model ROI view is more pessimistic at the same time point.
30. **CLV and hit rate diverge past +6h — CLV alone is a misleading "is this still working" signal.** Mean CLV stays positive even at +24h (+0.083) while hit rate collapses from 63.0% (open) to 38.9% (+24h), well below the model's own ~62-68% raw accuracy. Consistent with adverse selection: once the line has had a day to correct, the trades the model still disagrees with the market on are disproportionately the cases the market got right. Confirmed broad across all three leagues (LCK 65.5%→37.5%, LPL 61.5%→38.7%, LCS 61.1%→42.9% hit rate, open vs +24h) — not one outlier league driving it.
31. **No rescue point exists late in the window** — re-ran the original hours-before-match grid (24h/12h/8h/6h/4h/3h/2h/1h before match) with the real model. Every single point is negative ROI/bet (-13.5% to -0.1%). The "bet later, accept less edge for more depth" idea does not work anywhere in the tested window — the only zone with a defensible positive-ROI case is open through roughly +3 to +6h after market creation.
32. **Apparent +1h/+2h ROI "bump" above the open-time estimate is not statistically real** — bootstrap 95% CIs (5,000 resamples) at open, +1h, +2h, +3h overlap heavily. Read this as one flat high-edge plateau through ~3-6h, not a precise timing optimum to chase.
33. **Bankroll sweep ($1K-$50K, real-model trades) shows the 2% per-market cap — not depth — binds at every level tested**, producing identical +126%/+105%/+87%/+56% ROI (at 1x/1.5x/2x/3x cost) regardless of bankroll size. This is a property of the fillable estimator (% of *lifetime* volume, already flagged as likely overstating real opening-hour depth), not evidence that real depth supports $50K. Depth-bound trades were only 0.7% of all trades at every bankroll level under the current proxy.
34. **Depth-sensitivity sweep quantifies how fragile the larger-bankroll case is.** Scaling the fillable proxy down: at $10K bankroll, ROI is fairly robust until real depth falls below ~10% of the proxy (~$800/market), where it drops from +126% to +120%. At $50K, the same 10%-of-proxy threshold cuts ROI roughly in half (+126% → +61%), and depth-bound trades jump to 100%. **The larger the proposed bankroll, the more the backtest's economics depend on an unmeasured number** — anything above ~$10-25K should be treated as contingent on direct opening-hour depth observation, not on this backtest.
35. **Full real-data writeup:** `docs/t1_deep_dive_report.html` — per-league accuracy, fine-grained timing curve with bootstrap CIs, bankroll/cost/depth sensitivity, all captioned with data source, n, and walk-forward status, and explicit about what's backtested-and-conditional vs validated.

### Key reusable diagnostics (from this investigation)
- **Residual variance without accuracy gain = redundant feature.** 1.39% R² on ELO residuals → 0% accuracy gain. Fast-kill gate for future candidates.
- **Brier improvement without accuracy improvement = calibration artifact.** Adding zero-mean noise softens overconfident probabilities toward 0.5, nudging Brier down without improving a single game's predicted winner.
- **In-sample improvement is not evidence.** The Platt refit halved gaps on 20K matches (in-sample) and was worst on actual trades (out-of-sample). Every future feature gets the betting-set out-of-sample check before shipping.
- **Era-dependent parameters should not be finely tuned.** When the parameter varies more across eras than it moves the metric, the parameter is noise.
- **Null in large sample + significant in small sample = small sample is the anomaly.** Pool depth p=0.455 on 16K games, p=0.003 on 2K holdout. The 16K result is trustworthy.

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
│   ├── calibration.py                ← Platt scaling (set to identity for V2)
│   ├── calibration_params.json       ← calibration params (a=1.0, b=0.0 — identity)
│   └── draft_elo.py                  ← draft signal trackers (tested, all null — reference only)
├── polymarket/
│   ├── bot.py                        ← Discord bot (alerts only on placed bets)
│   ├── live_engine.py                ← opening-line detector + paper execution + CLV
│   ├── scanner.py                    ← Polymarket market discovery (tag_slug)
│   ├── price_tracker.py              ← forward price snapshots + resolution tracking
│   ├── edge.py                       ← edge calculator (quarter-Kelly)
│   ├── paper_trader.py               ← legacy paper trading (bot uses for /portfolio)
│   └── discord_cards.py              ← decision cards, health dashboard, alerts
├── backtest/
│   ├── backtest.py                   ← walk-forward ELO backtester + grid search (V2: MOV, eval windows)
│   ├── polymarket_backtest.py        ← walk-forward P&L backtest vs real Polymarket markets
│   ├── draft_backtest.py             ← draft signal evaluation pipeline (residual test, McNemar, CI)
│   ├── persist_markets.py            ← fetches + persists resolved PM markets to JSON
│   ├── t1_backtest.py                ← T1 Monte Carlo feasibility backtester (assumed accuracy, not real model)
│   ├── t1_model_backtest.py          ← T1 REAL model backtester (walk-forward ELO + real PM prices, fine timing grid)
│   ├── pnl_backtest.py               ← simulated market P&L sweep
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
│   ├── polymarket_resolved.json      ← frozen T2 dataset: 662 markets + price histories (17 MB)
│   ├── polymarket_t1_resolved.json   ← frozen T1 dataset: 296 markets + price histories (16 MB)
│   ├── newmetrics/                   ← draft/game data (20K games, 2021-2026)
│   │   ├── draft_picks.csv           ← 205K player-champion-game rows
│   │   ├── team_pickbans.csv         ← 41K team draft rows (bans + picks + first_pick)
│   │   ├── games.csv                 ← 20,587 T2 game results
│   │   └── patches.csv               ← patch date ranges
│   ├── t1_trades_*.csv              ← T1 Monte Carlo trade logs
│   └── upcoming_matches.csv          ← match tracking spreadsheet
├── docs/
│   ├── project_summary_for_ian.md    ← full project summary for presentation
│   ├── sizing_and_capacity.md        ← Kelly analysis, capacity ceiling, monthly projections
│   ├── audit_step_2g.md              ← metric integrity audit + lookahead fix
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

## Model Architecture (V2 ELO — the shipped model)

**How team ELOs are calculated:**

1. **Initialization** — teams start at 1500 + regional offset (KR +140, EU +134, NA +59, etc.). Soloq baselines are NOT used (removed — only 38/440 teams had data, slight negative).

2. **Match ELO updates with margin-of-victory scaling** — `new_elo = old_elo + K_adj * (actual - expected)` where K_adj = K * MOV_multiplier. The MOV multiplier scales K based on kill differential and gold diff at 15 minutes: dominant wins get K up to ~1.5x, narrow wins get ~1.0x. This captures strength information that binary win/loss ELO throws away.

3. **Time decay** — ELO decays toward 1500 with 270-day half-life during inactivity.

4. **Blending** — `rating = alpha * match_elo + (1-alpha) * 1500` where `alpha = games / (games + 5)`. New teams regress toward 1500; established teams use match ELO.

5. **Prediction** — logistic function: `P(A wins) = 1 / (1 + 10^(-(rating_A - rating_B) / 400))`. No Platt correction — V2 is self-calibrated (identity: a=1.0, b=0.0).

```
matches ──→ pro_elo.py ──→ team ELO (K=32, MOV scaling, regional offsets, 270d decay)
                                     ↓
                              blend.py (alpha = games / (games + 5), no soloq)
                                     ↓
                              predict.py → P(win)
                                     ↓
                              + Identity calibration (pass-through)
                              + Blue side offset (+22.3 ELO)
                              + Cross-region adjustment (80% of regional offset)
                              + Bo series conversion (negative binomial)
```

**Parameters (frozen, grid-search optimized on 20K matches):**
- K = 32, blend_k = 5, scale = 400, half_life = 270 days, mov_weight = 1.5
- Calibration: identity (a=1.0, b=0.0)
- Blue side: +22.3 ELO (from 53.2% blue WR across 20,587 matches)
- No soloq baselines

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
6. On resolution: computes CLV vs pre-match close (anchored on match_start_ts), realized P&L

**Discord bot** only sends notifications when a bet is placed or resolved. Silent on suppressed evaluations and routine scans.

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

---

## Backtesting

**Walk-forward ELO backtest (V2, 20,587 matches, eval on 2024-2025):**
```
Accuracy:    63.8%
Brier:       0.2238
Best params: K=32, blend_k=5, scale=400, half_life=270d, mov_weight=1.5, no soloq
```

**Opening-line P&L backtest (220 trades, V2 model, frozen Polymarket dataset):**
```
Strategy:    Same-region T2, >10% edge, bet at market open
Dataset:     662 resolved markets persisted with full price histories (data/polymarket_resolved.json)
ELOs:        Walk-forward — only matches before each market's date (no lookahead)
Calibration: Identity (a=1.0, b=0.0)
Costs:       Volume-dependent (3% liquid → 8% thin markets) — UNVALIDATED against real fills
Fillable:    1-3% of total volume at open ($20-$300 per bet)
Sizing:      Quarter-Kelly, 2% bankroll cap, depth-gated

Trades:      220 (137W / 83L)
Hit rate:    62.3%
P&L ($1K):   $1,000 → $2,104 (+110%)
P&L ($5K):   $5,000 → $9,500 (+90%)
Max DD:      16.8%
CLV:         +0.121 (65% beat pre-match close)
Edge-depth:  r=-0.006 (edge NOT concentrated in thin markets)
```

**P&L sensitivity to cost assumptions (the untested risk):**
```
                   Final ($1K)   ROI    Max DD
0.5x cost model    $2,477       +148%   15.1%
1.0x (current)     $2,104       +110%   16.8%
1.5x pessimistic   $1,808        +81%   18.3%
2.0x worst case    $1,507        +51%   20.0%
```
Edge survives even at 2x costs, but the gap between +110% and +51% is the honest uncertainty range.

**Timing analysis (T2 model vs line at different time points):**
```
                Model Acc  Line Acc  Trades  Hit%   ROI/bet
Open              64.9%     54.2%     147    65%   +25.0%  ← ONLY profitable time
4h before         63.8%     66.0%     100    31%   -23.4%  ← line already smarter
2h before         63.8%     67.0%     101    34%    -8.4%
1h before         64.2%     66.3%     106    34%    -8.0%
```

**Frozen dataset:** `data/polymarket_resolved.json` — 662 markets, all integrity-verified (point-in-time, no backfill). All backtesting uses this frozen artifact, not live API calls.

Generate trade logs: `python backtest/polymarket_backtest.py --bankroll 10000`

---

## T1 Expansion (LCK/LPL/LCS — model validated, fill depth UNKNOWN)

**STATUS: Model edge is real. Economic viability is gated entirely on opening-book depth, which has NEVER been measured. Do not promote to live until depth logging confirms fillable size at open.**

**⚠️ The trade-level numbers immediately below (162 trades, +0.160 CLV) come from `backtest/t1_backtest.py`, a Monte Carlo simulator that assigns the model a fixed 0.65 win probability rather than running the real ELO model.** Per-league accuracy is confirmed correct against the real model (see finding #28). For real-model, real-price trade economics — including a fine-grained timing decay curve and bankroll/depth sensitivity — see `backtest/t1_model_backtest.py` and `docs/t1_deep_dive_report.html` (findings #27-35, June 29 2026). The real-model numbers are close but not identical (153 trades vs 162, +0.169 CLV vs +0.160) and the timing-decay zero-crossing is earlier than this section's table suggests.

**V2 model on T1 (walk-forward, audited, no lookahead):**
```
Dataset:     296 resolved T1 markets (data/polymarket_t1_resolved.json)
             LPL: 152, LCK: 99, LCS: 45 — integrity-verified, no backfill
DB matches:  4,556 T1 matches (LCK 1,386 + LPL 1,975 + LCS 349 + LEC already in DB)

Model accuracy (walk-forward ELO baseline, eval 2024+):
  LCK:      67.7%  (1,386 matches)
  LEC:      64.3%  (846 matches)
  LCS:      62.8%  (349 matches)
  LPL:      61.6%  (1,975 matches)
  Combined: 64.1%  (4,556 matches)

Polymarket opening accuracy: 53.0% (266 matched markets)
V2 model accuracy on PM markets: 66.9%

Edge trades (>10% edge): 162
Hit rate: 64.2%
Mean CLV: +0.160 (71% beat pre-match close)
```

**CRITICAL: The edge lives ONLY at the 6-day-out opening price.**
```
CLV by entry timing (162 edge trades):
  Open (6 days out):   +0.160  ← ALL the edge is here
  1h after open:       +0.099
  4h after open:       +0.054
  24h after open:      +0.008  ← nearly zero
  48h after open:      -0.011  ← negative
  4 days after open:   -0.023  ← losing money
```
The line corrects to zero-edge within 24 hours of market creation. This is the T2 pattern stretched across days. **Your edge is not "the model beats T1 lines." Your edge is "the model beats a 6-day-out price that nobody has corrected yet."**

**Cost sensitivity (at market open, 3% base cost):**
```
                   Final ($1K)   ROI    Max DD
0.5x cost (1.5%)  $2,742       +174%   11.5%
1.0x (3%)         $2,412       +141%   12.0%
2.0x (6%)         $1,911        +91%   13.0%
3.0x (9%)         $1,537        +54%   13.8%
```
Survives even at 3x costs — BUT this assumes fillable=$500, which has ZERO evidence. The $662K median volume is lifetime, not opening-hour volume. T2's fillable estimator already overstated depth. Six-days-out depth on a T1 market could be near zero.

**What determines T1 viability:** A few weeks of passive observation of live T1 Polymarket markets, logging actual order book depth in the first 1h/6h/24h after market creation. If fillable is $200+, the capacity ceiling blows past T2. If fillable is $20, T1's capacity advantage evaporates and it's a more complex path to the same ceiling.

**T1 paper trader MUST log depth-at-entry as the primary field, not P&L.** Standard paper trading that assumes fill at the quoted open reproduces +0.16 CLV and proves nothing.

---

## Running Everything

```bash
# Scrapers (run periodically)
python scrapers/oe_scraper.py              # T2 match data (weekly, default)
python scrapers/oe_scraper.py --tier t1    # T1 match data (weekly)
python scrapers/oe_scraper.py --tier all   # both tiers
python scrapers/ttp_scraper.py             # soloq snapshots (daily)
python scrapers/roster_scraper.py          # rosters (match days)

# Model pipeline
python model/soloq_rating.py              # recompute player ratings
python model/pro_elo.py                   # rebuild team ELOs (T1+T2)

# Predictions
python model/predict.py "Solary" "Galions"
python model/predict.py "Solary" "Galions" --bo 5 --side blue

# Backtesting
python backtest/backtest.py --optimize                              # T2 ELO grid search
python backtest/backtest.py --K 32 --blend-k 5 --scale 400 --half-life 270 --mov-weight 1.5 --no-soloq --league LCK --eval-start 2024-01-01  # T1 single league
python backtest/polymarket_backtest.py                              # T2 walk-forward P&L
python backtest/persist_markets.py --tier t1                        # persist T1 Polymarket markets
python backtest/persist_markets.py --tier t2                        # persist T2 Polymarket markets

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
- **Do not re-tune model params** — V2 config is frozen (K=32, mov_weight=1.5, identity cal)
- **Every new feature must pass the betting-set out-of-sample test** before shipping — in-sample improvement is not evidence
- **Features that restate team strength are redundant with ELO** — don't build them (draft, rosters, champion stats, rolling team stats all tested and null)

---

## Key Decisions Log

| Decision | Choice | Reason |
|---|---|---|
| Strategy | Bet opening lines | Opening 54% accurate, model ~62-64% walk-forward — edge exists at open |
| Edge threshold | >10% | Plateau from adversarial sweep; 5-10% eaten by costs |
| Region filter | Same-region only | Cross-region: 43% hit, negative ROI |
| ELO K-factor | K=32 (was 64) | Lower K + MOV scaling: routine wins move less, stomps move more |
| MOV scaling | mov_weight=1.5 | +0.5% accuracy; captures strength info binary ELO discards |
| Soloq baselines | Removed | Slight negative with only 38/440 teams; adds noise |
| Calibration | Identity (a=1.0) | V2 is self-calibrated; 4 calibrations tested, all within noise except all-data (broken). Zero free parameters. |
| Sizing | Quarter-Kelly, stay conservative | Cost model unvalidated; size as if CLV is +0.08, not +0.12 |
| Bet timing | Market open only | Model loses money at 4h/2h/1h before match — line already efficient |
| Walk-forward | Mandatory for all backtests | Earlier version used final ELOs — inflated hit rate 72% → 65.3% |
| Draft features | Dead (twice) | V3 GBM (107 features): 0%. Draft ELO offsets (3 signals, 16K games): all null. Redundant with ELO by construction. |
| Roster gate | Dead | Model MORE accurate on changed rosters (+3.7%). T2 subs mostly on losing teams. |
| Line movement | Shelved | n=28 eligible test markets. Instrument can't resolve. Revisit if eligible set grows >80. |
| Patch freshness | Dead | Zero P&L impact. ELO robust to patch transitions. |
| Live in-game | No-go | Mid-game liquidity ($352) worse than open ($979) |
| T1 expansion | Model validated, fill unknown | 66.9% accuracy, +0.16 CLV — but only at 6-day-out price. Fill depth unmeasured. |
| Deployment | Paper first + fill validation | 30+ bets with positive CLV AND realized fills tracking cost model |

---

## What's NOT Working / Known Issues

1. **Cost model is unvalidated** — estimate_opening_cost and estimate_fillable_at_open have NEVER been checked against a real fill. P&L swings from +110% to +51% at 2x costs. This is the single biggest risk. Forward paper trading must log predicted vs actual fills.
2. **Live engine still uses V1 params** — needs update to V2 (K=32, mov_weight=1.5, identity calibration, no soloq). The ELOTracker in polymarket_backtest.py also uses V1 params.
3. **All P&L numbers are from one 4-month window (Jan-May 2026)** — a single meta slice. Forward edge may be smaller.
4. **190 Polymarket markets have team-matching failures** — 105 recovered by matcher fix, 85 remaining are genuinely unknown teams (mostly Prime League, Hitpoint, Road of Legends sub-leagues).
5. **Fillable estimate uses total lifetime volume, not opening volume** — KC vs DCG showed $300 estimated fillable but only $1,460 traded in first 24h. The estimator may overstate available depth.
6. **6 league mappings missing** from roster scraper: LCO, ESLOL, LEC, LTA N, LTA S, CBLOL Academy.
7. **Galions underestimation** — 5 of top 10 confident losses. ELO lags for teams on hot streaks.
8. **No historical bookmaker odds source** — the-odds-api doesn't cover esports; OddsPortal blocks bots.
9. **T1 match data missing** — no LCK, LPL, or LCS matches in DB. Only LEC (846 matches). Blocks T1 model.
10. **Capacity ceiling** — T2 liquidity caps profitable deployment at ~$5-10K bankroll.

---

## Docs Index

| Document | What it covers |
|---|---|
| `docs/project_summary_for_ian.md` | Full project summary for external presentation |
| `docs/sizing_and_capacity.md` | Kelly fractions, capacity analysis, monthly projections |
| `docs/audit_step_2g.md` | Metric integrity audit — lookahead fix, blend_k, CLV, ordering |
| `docs/session_2026_06_18.md` | Original session report — model changes, scanner fixes, market discovery |
| `docs/adversarial_and_feasibility.md` | Adversarial validation + live in-game no-go |
| `docs/research_brief.md` | Research directions for improving accuracy |
| `docs/audit_2026_06_18.md` | Codebase audit |
| `docs/model_system_deep_dive.md` | Model math documentation |
| `docs/context_handoff.md` | Partner handoff notes |
| `docs/strategic_report.html` | Manager-facing visual summary (Chart.js), built from this file's headline numbers |
| `docs/t1_deep_dive_report.html` | Real-model T1 analysis — per-league accuracy, fine-grained timing decay, bankroll/depth sensitivity (not Monte Carlo) |
