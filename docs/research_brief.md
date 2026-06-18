# Research Brief: Beating Polymarket on LoL T2 Esports

## The Problem

We built an ELO + gradient boosted model to predict League of Legends Tier 2 professional match outcomes. The goal is to bet profitably against Polymarket prediction markets.

**The model fails.** Polymarket closing prices are 96% accurate on T2 LoL matches (91% market efficiency). Our best model reaches 63-64% accuracy regardless of complexity — V1 (pure ELO, 12 features) and V3 (107 features including objectives, draft, vision, economy) perform identically. Adding 94 features from Oracle's Elixir match data (dragon control rates, baron rates, vision scores, DPM, champion win rates, gold diff trajectories, turret plates) produced zero accuracy improvement over simple ELO + win rate.

**Key numbers:**
- Model accuracy: 63.4-63.8% (walk-forward, 8,064 test matches, no lookahead)
- Polymarket accuracy: 96.0% (251 resolved T2 moneyline markets)
- Model Brier score: 0.2224
- Polymarket Brier score: 0.0223
- Breakeven market efficiency: ~82%. Polymarket is at 91%. We lose money.

## What We Tried That Didn't Work

1. **More rolling features (27 new categories):** Dragon control, baron rate, first objective rates, void grubs, gold diff at 20 min, XP diffs, CS diffs, gold diff slope/acceleration, DPM, CSPM, earned gold, tower control, turret plates, vision score, wards placed/killed, team kills per minute, early kill share. All computed as 15-game rolling averages per team. Result: 0% accuracy gain.

2. **Champion/draft features:** Rolling champion win rate per role (300-game window), player-champion comfort score, meta score (% of top-20% WR picks), minimum champion WR on the team. Result: 0% accuracy gain.

3. **Stronger regularization:** max_depth 3-6, L2 regularization 0.1-1.0, max_features 50%, min_samples_leaf 15-30. Reduced overfitting but didn't improve accuracy.

4. **Platt calibration:** Improved Brier score by 0.0013 but didn't change which matches the model gets right.

5. **Regional adjustments for cross-region matches:** Helped specific matchups (KC vs DCG went from 47% to 85%) but doesn't affect same-region accuracy.

## Why We Think The Model Plateaus

The 63-64% ceiling persists across V1 (ELO only) and V3 (107 features). This suggests:

- **Rolling team stats are redundant with ELO.** Dragon control, baron rate, vision scores are *consequences* of team strength, not independent signals. A team with high ELO already implies they secure more objectives. The GBM can't extract additional signal beyond what ELO already captures.

- **Pre-match features may have a fundamental accuracy ceiling around 63-65% for T2 LoL.** The remaining 35% is determined by factors not in historical stats: draft execution in the specific game, player mental state, meta adaptation, in-game shotcalling, and variance.

- **Polymarket's 96% likely comes from in-game price movement, not pre-match prediction.** Markets stay open during the game. A 60/40 pre-match line shifts to 95/5 after first blood. The "closing price" we measured may include in-game information, not just pre-match consensus.

## What We Need Help Researching

### 1. Esports prediction models that beat 65% accuracy on pre-match data
- Are there published models for LoL, CS2, or Dota 2 that achieve >70% pre-match accuracy?
- What features or architectures do they use?
- Do any use player-level performance (not just team-level aggregates)?
- Specific interest: models that use individual player statistics per role (jungle pathing, laning stats per player, support roaming patterns) rather than team averages

### 2. Prediction market efficiency in esports
- How efficient are Polymarket, Betfair, and traditional bookmakers on esports?
- Is there academic literature on prediction market efficiency for niche sports/esports?
- Are closing prices or opening prices more efficient? (If opening prices are less efficient, that's when to bet)
- What is the typical Brier score of sharp bookmaker closing lines on esports?

### 3. Approaches to beat efficient markets with limited edge
- Kelly criterion variants for thin-edge situations (our edge if any is <3%)
- Market microstructure strategies: betting early before the line settles vs waiting for maximum information
- Contrarian signals: when does the crowd systematically misprice esports? (roster changes, meta shifts, regional bias?)
- Stale line detection: can we identify markets that haven't updated to reflect new information (e.g., roster swap announced after market opened)?

### 4. Alternative modeling approaches
- **Temporal models (LSTM/transformer):** Can sequence-based models capture momentum/form patterns that rolling averages miss?
- **Player-level models instead of team-level:** Model each player's contribution separately, then aggregate (like baseball WAR for esports)
- **Graph neural networks:** Model team composition as a graph (player interactions, role synergy)
- **Transfer learning from T1 to T2:** T1 leagues have more data and coverage — can a model trained on LCK/LPL/LEC transfer knowledge to T2?
- **Bayesian approaches:** Prior from ELO, updated with each new piece of information (roster change, patch, draft)
- **Meta-learning:** Different model for different confidence regimes — use ELO for uncertain matches, GBM for high-data matches

### 5. In-play/live betting models
- If Polymarket's edge comes from in-game information, should we shift to in-game prediction?
- Models that predict match outcome from game state (gold diff, dragon count, tower count) at 10/15/20 minutes
- The OE data has goldat10, goldat15, goldat20, goldat25 — these are in-game snapshots that could power a live model
- Live Polymarket prices during games could be compared against a live model

### 6. Data sources we're not using
- **Scrim results / bootcamp leaks:** Not publicly available but informed bettors may have this
- **Social media sentiment:** Player/team tweets, Reddit, Discord communities
- **Patch notes analysis:** Quantify which teams benefit from patch changes based on their champion pools
- **Solo queue form:** We have daily soloq snapshots — a player's recent soloq performance might predict upcoming match performance
- **Head-to-head history:** Some teams consistently beat/lose to specific opponents regardless of ELO

## Our Data

- 10,372 T2 professional matches (2024-2026) across 18 leagues, 440 teams
- 165 columns per match (team-level stats)
- 103,720 player-game rows with champion picks, individual stats
- 2,291 pro players with daily solo queue rating snapshots
- 598 roster entries with role assignments
- 486 resolved Polymarket T2 moneyline markets (closing prices recoverable via CLOB API with startTs=1)
- All data is Oracle's Elixir (public), TrackingThePros (public), Leaguepedia (public)

## The Core Question

Is 63-64% the fundamental ceiling for pre-match T2 LoL prediction from public data? Or is there a modeling approach — different architecture, different features, different granularity — that can reach 75%+ and make Polymarket betting profitable?

If the answer is "63% is the ceiling from public data," then the strategy shifts to either:
- **Live/in-game prediction** (bet during the match as odds shift)
- **Opening line exploitation** (bet before the market gets efficient)
- **Information edge** (find data sources the market doesn't have)
- **Different market** (find less efficient prediction markets or bookmakers)
