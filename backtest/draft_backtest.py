"""
Draft signal evaluation pipeline.

Runs the full evaluation sequence from the plan:
  Phase 0: Residual test (fast kill gate)
  Phase 2: Individual signal testing with McNemar + CI
  Phase 3: Combination search (surviving signals only)
  Phase 4: Holdout test (2026, one shot)

Usage:
  python backtest/draft_backtest.py --residual-test
  python backtest/draft_backtest.py --test-signals
  python backtest/draft_backtest.py --full
  python backtest/draft_backtest.py --holdout
"""

import argparse
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from backtest.backtest import (
    BacktestResult,
    ELOTracker,
    load_matches,
    print_report,
)
from model.draft_elo import DraftTracker
from model.pro_elo import HALF_LIFE_DAYS, compute_regional_offsets, get_team_soloq_elos

DB_PATH = _ROOT / "db" / "lol_model.db"

# Baseline params (re-tuned on expanded 20K dataset, eval 2024-2025)
BASE_K = 32
BASE_BLEND_K = 5
BASE_SCALE = 400
BASE_HALF_LIFE = 270
BASE_MOV = 1.5

EVAL_START = "2024-01-01"
EVAL_END = "2025-12-31"
HOLDOUT_START = "2026-01-01"
HOLDOUT_END = "2026-12-31"


# ---------------------------------------------------------------------------
# Walk-forward with draft signals
# ---------------------------------------------------------------------------
def run_draft_backtest(
    draft_tracker: DraftTracker,
    champ_wr_coeff: float = 0.0,
    mastery_coeff: float = 0.0,
    pool_coeff: float = 0.0,
    eval_start: str = EVAL_START,
    eval_end: str = EVAL_END,
    K: float = BASE_K,
    blend_k: int = BASE_BLEND_K,
    scale: float = BASE_SCALE,
    half_life: float = BASE_HALF_LIFE,
    mov_weight: float = BASE_MOV,
) -> Tuple[BacktestResult, List[Tuple[str, float, float, float]]]:
    """
    Run walk-forward backtest with draft signal offsets.

    Returns (BacktestResult, detailed_predictions) where detailed_predictions
    is [(gameid, p_blue, actual, draft_offset), ...].
    """
    tracker = ELOTracker(
        K=K, blend_k=blend_k, scale=scale,
        half_life_days=half_life,
        soloq_baselines={}, regional_offsets={},
        mov_weight=mov_weight,
    )

    matches = load_matches()
    draft_tracker.reset()

    result = BacktestResult(K=K, blend_k=blend_k, scale=scale, half_life=half_life,
                            mov_weight=mov_weight, use_soloq=False)
    predictions = []
    detailed = []
    league_stats = defaultdict(lambda: [0, 0])

    for row in matches:
        gameid, date, league, blue, red, winner = row[:6]
        blue_kills, red_kills, blue_deaths, red_deaths = row[6], row[7], row[8], row[9]
        blue_gd15, red_gd15 = row[10], row[11]
        stats = dict(blue_kills=blue_kills, red_kills=red_kills,
                     blue_deaths=blue_deaths, red_deaths=red_deaths,
                     blue_gd15=blue_gd15, red_gd15=red_gd15)

        if date < eval_start:
            tracker.update(blue, red, winner, league, date, **stats)
            draft_tracker.advance_to(gameid)
            draft_tracker._update_state(gameid)
            result.warmup_matches += 1
            continue

        if date > eval_end:
            tracker.update(blue, red, winner, league, date, **stats)
            draft_tracker.advance_to(gameid)
            draft_tracker._update_state(gameid)
            continue

        # Advance draft tracker to just before this game
        draft_tracker.advance_to(gameid)

        # Compute base ELO prediction
        p_blue_base = tracker.predict(blue, red, league, date)

        # Compute draft offset
        signals = draft_tracker.compute_signals(gameid)
        draft_offset = 0.0
        if signals["a"] is not None and champ_wr_coeff > 0:
            draft_offset += champ_wr_coeff * signals["a"]
        if signals["b"] is not None and mastery_coeff > 0:
            draft_offset += mastery_coeff * signals["b"]
        if signals["d"] is not None and pool_coeff > 0:
            draft_offset += pool_coeff * signals["d"]

        # Apply offset to rating difference (convert back through logistic)
        if draft_offset != 0.0:
            logit = math.log(p_blue_base / (1.0 - p_blue_base + 1e-10))
            logit_scaled = logit * scale / math.log(10)
            logit_scaled += draft_offset
            p_blue = 1.0 / (1.0 + 10.0 ** (-logit_scaled / scale))
        else:
            p_blue = p_blue_base

        actual = 1.0 if winner == "blue" else 0.0
        predictions.append((p_blue, actual))
        result.predictions_with_ids.append((gameid, p_blue, actual))
        detailed.append((gameid, p_blue, actual, draft_offset))

        predicted_winner = p_blue >= 0.5
        actual_winner = actual == 1.0
        if predicted_winner == actual_winner:
            result.correct += 1
        league_stats[league][0] += 1
        if predicted_winner == actual_winner:
            league_stats[league][1] += 1

        # Update state AFTER prediction
        tracker.update(blue, red, winner, league, date, **stats)
        draft_tracker._update_state(gameid)

    result.test_matches = len(predictions)
    result.predictions = predictions

    if predictions:
        preds = np.array([p for p, _ in predictions])
        actuals = np.array([a for _, a in predictions])
        result.accuracy = result.correct / result.test_matches
        result.brier_score = float(np.mean((preds - actuals) ** 2))
        eps = 1e-10
        preds_clipped = np.clip(preds, eps, 1.0 - eps)
        result.log_loss = float(-np.mean(
            actuals * np.log(preds_clipped) + (1 - actuals) * np.log(1 - preds_clipped)
        ))

    result.league_accuracy = {
        league: (correct, total)
        for league, (total, correct) in league_stats.items()
    }

    return result, detailed


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def mcnemar_test(preds_a: List[Tuple[float, float]],
                 preds_b: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    McNemar's test on discordant predictions.
    Returns (chi2_statistic, p_value).
    """
    b_right_a_wrong = 0
    a_right_b_wrong = 0

    for (pa, actual_a), (pb, actual_b) in zip(preds_a, preds_b):
        a_correct = (pa >= 0.5) == (actual_a >= 0.5)
        b_correct = (pb >= 0.5) == (actual_b >= 0.5)
        if b_correct and not a_correct:
            b_right_a_wrong += 1
        elif a_correct and not b_correct:
            a_right_b_wrong += 1

    n_discord = b_right_a_wrong + a_right_b_wrong
    if n_discord == 0:
        return 0.0, 1.0

    chi2 = (abs(b_right_a_wrong - a_right_b_wrong) - 1) ** 2 / n_discord
    # chi2 with 1 df -> p-value approximation
    p = math.exp(-chi2 / 2.0) if chi2 < 20 else 0.0
    return chi2, p


def brier_ci(preds: List[Tuple[float, float]], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Bootstrap 95% CI on Brier score."""
    scores = np.array([(p - a) ** 2 for p, a in preds])
    n = len(scores)
    brier = float(scores.mean())

    n_boot = 2000
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means.append(sample.mean())

    alpha = (1 - confidence) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return brier, lo, hi


def accuracy_ci(preds: List[Tuple[float, float]], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Normal approximation CI on accuracy."""
    correct = sum(1 for p, a in preds if (p >= 0.5) == (a >= 0.5))
    n = len(preds)
    acc = correct / n
    z = 1.96
    se = math.sqrt(acc * (1 - acc) / n)
    return acc, acc - z * se, acc + z * se


# ---------------------------------------------------------------------------
# Phase 0: Residual test
# ---------------------------------------------------------------------------
def run_residual_test(draft_tracker: DraftTracker) -> None:
    """
    Test whether draft features explain residual variance after ELO.
    Fit logistic regression of outcome on ELO prediction + draft features.
    """
    logger.info("Phase 0: Residual test — do draft features add info beyond ELO?")

    tracker = ELOTracker(
        K=BASE_K, blend_k=BASE_BLEND_K, scale=BASE_SCALE,
        half_life_days=BASE_HALF_LIFE,
        soloq_baselines={}, regional_offsets={},
        mov_weight=BASE_MOV,
    )

    matches = load_matches()
    draft_tracker.reset()

    elo_preds = []
    actuals = []
    features_a = []
    features_b = []
    features_d = []

    for row in matches:
        gameid, date, league, blue, red, winner = row[:6]
        stats = dict(blue_kills=row[6], red_kills=row[7],
                     blue_deaths=row[8], red_deaths=row[9],
                     blue_gd15=row[10], red_gd15=row[11])

        if date < EVAL_START:
            tracker.update(blue, red, winner, league, date, **stats)
            draft_tracker.advance_to(gameid)
            draft_tracker._update_state(gameid)
            continue

        if date > EVAL_END:
            tracker.update(blue, red, winner, league, date, **stats)
            draft_tracker.advance_to(gameid)
            draft_tracker._update_state(gameid)
            continue

        draft_tracker.advance_to(gameid)

        p_blue = tracker.predict(blue, red, league, date)
        signals = draft_tracker.compute_signals(gameid)
        actual = 1.0 if winner == "blue" else 0.0

        if signals["a"] is not None and signals["b"] is not None and signals["d"] is not None:
            elo_preds.append(p_blue)
            actuals.append(actual)
            features_a.append(signals["a"])
            features_b.append(signals["b"])
            features_d.append(signals["d"])

        tracker.update(blue, red, winner, league, date, **stats)
        draft_tracker._update_state(gameid)

    n = len(actuals)
    logger.info("  %d games with complete draft features in eval window", n)

    actuals = np.array(actuals)
    elo_preds = np.array(elo_preds)
    features_a = np.array(features_a)
    features_b = np.array(features_b)
    features_d = np.array(features_d)

    # Compute ELO residuals
    residuals = actuals - elo_preds

    # Correlation of each feature with residuals
    print("\n" + "=" * 60)
    print("  PHASE 0: RESIDUAL TEST")
    print("=" * 60)
    print("  Do draft features predict what ELO gets wrong?\n")

    for name, feat in [("A (champ WR)", features_a),
                       ("B (mastery)", features_b),
                       ("D (pool)", features_d)]:
        corr = np.corrcoef(feat, residuals)[0, 1]
        # Binned analysis: when feature is positive vs negative
        pos_mask = feat > 0
        neg_mask = feat < 0
        pos_resid = residuals[pos_mask].mean() if pos_mask.sum() > 0 else 0
        neg_resid = residuals[neg_mask].mean() if neg_mask.sum() > 0 else 0
        print("  Signal %-15s  r=%.4f  resid_when_pos=%.4f  resid_when_neg=%.4f" %
              (name, corr, pos_resid, neg_resid))

    # Joint: OLS of residual on all features
    X = np.column_stack([features_a, features_b, features_d])
    X = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X, residuals, rcond=None)[0]
        y_hat = X @ beta
        ss_res = np.sum((residuals - y_hat) ** 2)
        ss_tot = np.sum((residuals - residuals.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except Exception:
        r_squared = 0.0
        beta = np.zeros(4)

    print("\n  Joint OLS on ELO residuals:")
    print("    R² = %.6f (%.4f%% of residual variance explained)" % (r_squared, r_squared * 100))
    print("    Coefficients: intercept=%.4f  A=%.4f  B=%.4f  D=%.4f" %
          (beta[0], beta[1], beta[2], beta[3]))

    if r_squared < 0.001:
        print("\n  VERDICT: Draft features explain <0.1%% of ELO residuals.")
        print("  The premise is VERY weak. Proceed with grid search but expect null results.")
    elif r_squared < 0.01:
        print("\n  VERDICT: Draft features explain %.2f%% — marginal signal exists." % (r_squared * 100))
        print("  Worth testing as ELO offsets, but effect size is small.")
    else:
        print("\n  VERDICT: Draft features explain %.2f%% — meaningful signal!" % (r_squared * 100))
        print("  Proceed with full grid search.")
    print()


# ---------------------------------------------------------------------------
# Phase 2: Individual signal testing
# ---------------------------------------------------------------------------
def test_individual_signals(draft_tracker: DraftTracker) -> Dict[str, Dict]:
    """Test each signal individually against baseline. Returns results dict."""
    logger.info("Phase 2: Individual signal testing with McNemar + CI")

    # Run baseline (no draft)
    baseline, _ = run_draft_backtest(draft_tracker, eval_start=EVAL_START, eval_end=EVAL_END)
    base_acc, base_acc_lo, base_acc_hi = accuracy_ci(baseline.predictions)
    base_brier, base_brier_lo, base_brier_hi = brier_ci(baseline.predictions)

    print("\n" + "=" * 70)
    print("  PHASE 2: INDIVIDUAL SIGNAL TESTING")
    print("=" * 70)
    print("\n  Baseline (no draft): Acc=%.1f%% [%.1f%%, %.1f%%]  Brier=%.4f [%.4f, %.4f]"
          % (base_acc * 100, base_acc_lo * 100, base_acc_hi * 100,
             base_brier, base_brier_lo, base_brier_hi))
    print("  Eval window: %s to %s  |  %d games\n" % (EVAL_START, EVAL_END, baseline.test_matches))

    signals = {
        "A (champ WR)": {"param": "champ_wr_coeff", "values": [50, 100, 150, 200, 300]},
        "B (mastery)": {"param": "mastery_coeff", "values": [10, 20, 30, 50]},
        "D (pool)": {"param": "pool_coeff", "values": [5, 10, 15, 20]},
    }

    results = {}
    print("  %-15s  %6s  %8s  %8s  %8s  %8s  %6s" %
          ("Signal", "Coeff", "Acc", "Acc Δ", "Brier", "Brier Δ", "McNem p"))
    print("  " + "-" * 73)

    for sig_name, cfg in signals.items():
        best_result = None
        best_coeff = 0
        best_brier = 999

        for coeff in cfg["values"]:
            kwargs = {"champ_wr_coeff": 0, "mastery_coeff": 0, "pool_coeff": 0}
            kwargs[cfg["param"]] = coeff
            r, _ = run_draft_backtest(draft_tracker, eval_start=EVAL_START, eval_end=EVAL_END, **kwargs)

            if r.brier_score < best_brier:
                best_brier = r.brier_score
                best_result = r
                best_coeff = coeff

        # Stats for best coefficient
        acc, acc_lo, acc_hi = accuracy_ci(best_result.predictions)
        brier, brier_lo, brier_hi = brier_ci(best_result.predictions)
        chi2, p = mcnemar_test(baseline.predictions, best_result.predictions)

        acc_delta = acc - base_acc
        brier_delta = brier - base_brier

        print("  %-15s  %6d  %7.1f%%  %+7.2f%%  %8.4f  %+8.4f  %6.3f" %
              (sig_name, best_coeff, acc * 100, acc_delta * 100, brier, brier_delta, p))

        # CI on deltas
        acc_delta_se = math.sqrt(2 * acc * (1 - acc) / best_result.test_matches)
        brier_delta_ci = (brier_lo - base_brier_hi, brier_hi - base_brier_lo)

        results[sig_name] = {
            "coeff": best_coeff,
            "acc": acc, "acc_delta": acc_delta,
            "brier": brier, "brier_delta": brier_delta,
            "mcnemar_p": p,
            "brier_ci": (brier_lo, brier_hi),
            "significant": p < 0.05 and brier_delta < 0,
            "result": best_result,
        }

    print()
    surviving = {k: v for k, v in results.items() if v["significant"]}
    if surviving:
        print("  Surviving signals (p<0.05, Brier improved): %s" % ", ".join(surviving.keys()))
    else:
        print("  No signals survived the significance gate.")
        # Still report which had best directionality
        best_sig = min(results.items(), key=lambda x: x[1]["brier_delta"])
        print("  Best directionality: %s (Brier Δ = %+.4f, p=%.3f)" %
              (best_sig[0], best_sig[1]["brier_delta"], best_sig[1]["mcnemar_p"]))
    print()

    return results


# ---------------------------------------------------------------------------
# Phase 3: Combination search
# ---------------------------------------------------------------------------
def combination_search(draft_tracker: DraftTracker,
                       surviving: Dict[str, Dict]) -> Optional[Tuple[Dict, BacktestResult]]:
    """Grid search over surviving signal coefficients."""
    if not surviving:
        print("  Phase 3: SKIPPED — no surviving signals.\n")
        return None

    logger.info("Phase 3: Combination search over %d surviving signals", len(surviving))
    print("\n" + "=" * 60)
    print("  PHASE 3: COMBINATION SEARCH")
    print("=" * 60)

    # Build grid from surviving signals
    param_grid = {}
    for sig_name, data in surviving.items():
        if "champ WR" in sig_name:
            param_grid["champ_wr_coeff"] = [0, 50, 100, 150, 200, 300]
        elif "mastery" in sig_name:
            param_grid["mastery_coeff"] = [0, 10, 20, 30, 50]
        elif "pool" in sig_name:
            param_grid["pool_coeff"] = [0, 5, 10, 15, 20]

    best_brier = 999
    best_params = {}
    best_result = None

    # Nested grid search
    a_vals = param_grid.get("champ_wr_coeff", [0])
    b_vals = param_grid.get("mastery_coeff", [0])
    d_vals = param_grid.get("pool_coeff", [0])

    total = len(a_vals) * len(b_vals) * len(d_vals)
    done = 0
    for a in a_vals:
        for b in b_vals:
            for d in d_vals:
                r, _ = run_draft_backtest(
                    draft_tracker, champ_wr_coeff=a, mastery_coeff=b, pool_coeff=d,
                    eval_start=EVAL_START, eval_end=EVAL_END,
                )
                done += 1
                if r.brier_score < best_brier:
                    best_brier = r.brier_score
                    best_params = {"champ_wr_coeff": a, "mastery_coeff": b, "pool_coeff": d}
                    best_result = r

    print("\n  Best combination: %s" % best_params)
    print("  Brier=%.4f  Acc=%.1f%%\n" % (best_result.brier_score, best_result.accuracy * 100))
    return best_params, best_result


# ---------------------------------------------------------------------------
# Phase 4: Holdout test
# ---------------------------------------------------------------------------
def holdout_test(draft_tracker: DraftTracker,
                 params: Dict) -> None:
    """One-shot test on 2026 holdout data."""
    logger.info("Phase 4: Holdout test (2026)")
    print("\n" + "=" * 60)
    print("  PHASE 4: HOLDOUT TEST (2026)")
    print("=" * 60)
    print("  NOTE: At n≈2,125, SE≈1.08%%. Can refute large effects,")
    print("  cannot confirm small ones. CI straddling zero means")
    print("  '6 months cant resolve it', not 'signal is fake'.\n")

    # Baseline on holdout
    baseline, _ = run_draft_backtest(
        draft_tracker, eval_start=HOLDOUT_START, eval_end=HOLDOUT_END,
    )
    base_acc, base_acc_lo, base_acc_hi = accuracy_ci(baseline.predictions)
    base_brier, base_brier_lo, base_brier_hi = brier_ci(baseline.predictions)

    # Draft model on holdout
    draft_result, _ = run_draft_backtest(
        draft_tracker, eval_start=HOLDOUT_START, eval_end=HOLDOUT_END, **params,
    )
    draft_acc, draft_acc_lo, draft_acc_hi = accuracy_ci(draft_result.predictions)
    draft_brier, draft_brier_lo, draft_brier_hi = brier_ci(draft_result.predictions)

    chi2, p = mcnemar_test(baseline.predictions, draft_result.predictions)

    print("  %-20s  %12s  %12s  %10s" % ("Metric", "Baseline", "Draft", "Delta"))
    print("  " + "-" * 56)
    print("  %-20s  %11.1f%%  %11.1f%%  %+9.2f%%" %
          ("Accuracy", base_acc * 100, draft_acc * 100, (draft_acc - base_acc) * 100))
    print("  %-20s  [%.1f%%, %.1f%%]  [%.1f%%, %.1f%%]" %
          ("  95% CI", base_acc_lo * 100, base_acc_hi * 100, draft_acc_lo * 100, draft_acc_hi * 100))
    print("  %-20s  %12.4f  %12.4f  %+10.4f" %
          ("Brier", base_brier, draft_brier, draft_brier - base_brier))
    print("  %-20s  [%.4f, %.4f]  [%.4f, %.4f]" %
          ("  95% CI", base_brier_lo, base_brier_hi, draft_brier_lo, draft_brier_hi))
    print("  %-20s  %12s  %12s  %10.3f" % ("McNemar p", "", "", p))
    print("  %-20s  %12d  %12d" % ("Games", baseline.test_matches, draft_result.test_matches))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Draft signal evaluation pipeline")
    parser.add_argument("--residual-test", action="store_true", help="Phase 0: residual test only")
    parser.add_argument("--test-signals", action="store_true", help="Phase 2: individual signal testing")
    parser.add_argument("--full", action="store_true", help="Run Phase 0 through Phase 4")
    parser.add_argument("--holdout", action="store_true", help="Phase 4: holdout test only (requires params)")
    args = parser.parse_args()

    # Load draft data
    dt = DraftTracker()
    dt.load()

    if args.residual_test:
        run_residual_test(dt)

    elif args.test_signals:
        test_individual_signals(dt)

    elif args.holdout:
        # Use best params from prior run or defaults
        params = {"champ_wr_coeff": 100, "mastery_coeff": 0, "pool_coeff": 0}
        holdout_test(dt, params)

    elif args.full:
        # Phase 0
        run_residual_test(dt)

        # Phase 2
        results = test_individual_signals(dt)

        # Phase 3
        surviving = {k: v for k, v in results.items() if v["significant"]}
        combo = combination_search(dt, surviving)

        # If no surviving signals, try best directional anyway for reporting
        if combo is None:
            best_sig = min(results.items(), key=lambda x: x[1]["brier_delta"])
            best_name, best_data = best_sig
            params = {"champ_wr_coeff": 0, "mastery_coeff": 0, "pool_coeff": 0}
            if "champ WR" in best_name:
                params["champ_wr_coeff"] = best_data["coeff"]
            elif "mastery" in best_name:
                params["mastery_coeff"] = best_data["coeff"]
            elif "pool" in best_name:
                params["pool_coeff"] = best_data["coeff"]

            print("  Running holdout with best directional signal (%s, coeff=%d)" %
                  (best_name, best_data["coeff"]))
            print("  (This is exploratory — signal did not pass significance gate)\n")
        else:
            params = combo[0]

        # Phase 4
        holdout_test(dt, params)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
