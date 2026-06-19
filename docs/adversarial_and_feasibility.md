# Adversarial Validation + Step 3 Feasibility — June 18, 2026

## Adversarial Validation (Step 2D)

### Tier 1: Out-of-sample holdout — SURVIVES
- 105 holdout markets (most recent, never seen during rule design)
- 60 bets triggered by frozen rule (same-region, >10% edge)
- Hit rate: **73.3%** (CI: 61.7% – 83.3%)
- Net ROI: **+39.2%** (CI: +15.9% – +62.2%) — **CI clears zero**
- CLV: +0.091 — line confirms us on fresh data

### Tier 2: Stress tests — ALL ROBUST
- **Threshold sweep:** smooth plateau from 5% to 20%. Not overfit to 10%.
- **Cost stress:** CI clears zero even at +5% extra costs (ROI +0.204)
- **Anchor perturbation:** insensitive to ±30 min shifts in match_start detection
- **Leave-one-league-out:** ROI stable +30% to +37% regardless of which league dropped

### Tier 3: Regime boundaries — NO DECAY
- Q2 2026 stronger than Q1 (72.8% vs 63.2%). Edge is growing.
- Works on Bo1, Bo3, Bo5. No format dependency.
- Losses distributed across leagues. No fatal cluster.

### Tier 4: Mechanism — COHERENT
- CLV-positive bets win 83.7%. CLV-negative bets win 34.2%. Signal is real.
- Top losses involve Galions (EM team model underrates). Fixable, not fatal.

---

## Step 3 Feasibility (Live In-Game) — NO-GO

### Gate 1: In-game model accuracy — MARGINAL PASS
```
@10 min: 67.0% accuracy, 0.2067 Brier
@15 min: 71.7% accuracy, 0.1819 Brier
@20 min: 83.9% accuracy, 0.1142 Brier
```
Model gets dramatically better with in-game state, but probably doesn't beat the live Polymarket price at the same timestamp — the market reprices in <60 seconds.

### Gate 2: Mid-game liquidity — FAIL
```
              At Open     Mid-Game
Median fill:  $979        $352
```
Mid-game fillable is **0.4x** the open, not higher. The capacity rationale for Step 3 collapses. T2 books stay thin mid-match.

### Gate 3: Latency — NOT TESTED (Gate 2 failed)
Known: Riot feeds delayed 2-3 min, licensed feeds $5K+/yr with partial T2 coverage, Polymarket reprices in <60s.

---

## Conclusion

**The opening-line bot is the product.** It survives adversarial validation on every test — holdout, cost stress, threshold perturbation, league removal, time decay, and CLV coherence. Step 3 (live in-game) fails on capacity grounds: mid-game T2 liquidity is worse than at open, not better.

**Ship the opening-line bot. Cap it at ~$500-700/market. Monitor CLV on live paper bets. Promote to live capital after 30+ paper bets confirm positive CLV.**
