"""
V2 prediction model — gradient boosted classifier using team-level
features extracted from match history.

Uses the v1 ELO rating as one feature among many (recent form, gold
differentials, KDA, objective control). Walk-forward training ensures
no lookahead.

Run:
  python model/v2_model.py                # train + evaluate
  python model/v2_model.py --compare      # side-by-side v1 vs v2
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from backtest.backtest import ELOTracker, WARMUP_MONTHS
from model.calibration import PlattCalibrator
from model.features import build_feature_dataset
from model.pro_elo import compute_regional_offsets, get_team_soloq_elos

DB_PATH = _ROOT / "db" / "lol_model.db"
MODEL_PATH = _ROOT / "model" / "v2_model.pkl"

FEATURE_COLS = [
    "blue_win_rate", "blue_win_rate_last5", "blue_streak",
    "blue_avg_kills", "blue_avg_deaths", "blue_kda",
    "blue_avg_gamelength", "blue_avg_gd10", "blue_avg_gd15",
    "blue_fb_rate", "blue_ft_rate", "blue_games_played",
    "red_win_rate", "red_win_rate_last5", "red_streak",
    "red_avg_kills", "red_avg_deaths", "red_kda",
    "red_avg_gamelength", "red_avg_gd10", "red_avg_gd15",
    "red_fb_rate", "red_ft_rate", "red_games_played",
    "wr_diff", "kda_diff", "gd15_diff", "gd10_diff",
    "streak_diff", "fb_diff",
]


def walk_forward_evaluate(
    df: pd.DataFrame,
    warmup_frac: float = 0.15,
    retrain_every: int = 500,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Walk-forward evaluation with expanding training window.

    1. Use first warmup_frac of data for initial training
    2. Predict next batch, retrain every `retrain_every` matches
    3. Returns (v2_predictions, v1_predictions, actuals) — all on the test set
    """
    n = len(df)
    warmup_end = int(n * warmup_frac)

    if warmup_end < 200:
        warmup_end = min(200, n // 2)

    X = df[FEATURE_COLS].values
    y = df["result"].values

    # Also run v1 ELO for comparison
    soloq = get_team_soloq_elos()
    offsets = compute_regional_offsets()
    from backtest.backtest import load_matches
    matches = load_matches()
    first = matches[0][1]
    wy, wm = int(first[:4]), int(first[5:7]) + WARMUP_MONTHS
    while wm > 12: wm -= 12; wy += 1
    cutoff = f'{wy:04d}-{wm:02d}-01'
    tracker = ELOTracker(K=64, blend_k=5, scale=400, half_life_days=270,
                         soloq_baselines=soloq, regional_offsets=offsets)

    # Build v1 predictions aligned to feature dataset indices
    feature_gameids = set(df["gameid"].values)
    v1_preds_map: Dict[str, float] = {}
    for gameid, date, league, blue, red, winner in matches:
        if date < cutoff:
            tracker.update(blue, red, winner, league, date)
            continue
        p = tracker.predict(blue, red, league, date)
        if gameid in feature_gameids:
            v1_preds_map[gameid] = p
        tracker.update(blue, red, winner, league, date)

    v2_predictions: List[float] = []
    v1_predictions: List[float] = []
    actuals: List[float] = []

    model = None
    last_train_end = 0

    logger.info(f"Walk-forward: {warmup_end} warmup, {n - warmup_end} test, retrain every {retrain_every}")

    for i in range(warmup_end, n):
        # Retrain periodically
        if model is None or (i - last_train_end) >= retrain_every:
            X_train = X[:i]
            y_train = y[:i]
            # Handle NaN in features
            X_train = np.nan_to_num(X_train, nan=0.0)
            model = HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=20,
                random_state=42,
            )
            model.fit(X_train, y_train)
            last_train_end = i

        # Predict
        x_test = np.nan_to_num(X[i:i+1], nan=0.0)
        p_v2 = model.predict_proba(x_test)[0][1]  # P(blue wins)
        v2_predictions.append(float(p_v2))

        gid = df.iloc[i]["gameid"]
        p_v1 = v1_preds_map.get(gid, 0.5)
        v1_predictions.append(float(p_v1))

        actuals.append(float(y[i]))

    return v2_predictions, v1_predictions, actuals


def compute_metrics(preds: List[float], actuals: List[float]) -> Dict[str, float]:
    ps = np.array(preds)
    acts = np.array(actuals)
    correct = sum(1 for p, a in zip(preds, actuals) if (p >= 0.5) == (a == 1.0))
    accuracy = correct / len(preds)
    brier = float(np.mean((ps - acts) ** 2))
    eps = 1e-10
    ps_c = np.clip(ps, eps, 1 - eps)
    ll = float(-np.mean(acts * np.log(ps_c) + (1 - acts) * np.log(1 - ps_c)))
    return {"accuracy": accuracy, "brier": brier, "log_loss": ll, "n": len(preds)}


def calibration_table(preds: List[float], actuals: List[float]) -> None:
    bins = [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,0.85),(0.85,0.90),(0.90,1.01)]
    print(f"  {'Bucket':12} {'Predicted':>10} {'Actual':>10} {'Count':>7}")
    print(f"  {'-'*42}")
    for lo, hi in bins:
        fps, fas = [], []
        for p, a in zip(preds, actuals):
            fav_p = max(p, 1-p)
            fav_a = a if p >= 0.5 else 1-a
            if lo <= fav_p < hi:
                fps.append(fav_p)
                fas.append(fav_a)
        if fps:
            print(f"  {lo:.0%}-{hi:.0%}       {sum(fps)/len(fps):10.1%} {sum(fas)/len(fas):10.1%} {len(fps):7}")


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 model training and evaluation")
    parser.add_argument("--compare", action="store_true", help="Side-by-side v1 vs v2")
    args = parser.parse_args()

    logger.info("Building feature dataset…")
    df = build_feature_dataset()
    if df.empty:
        logger.error("No data")
        return

    logger.info("Running walk-forward evaluation…")
    v2_preds, v1_preds, actuals = walk_forward_evaluate(df)

    v2_m = compute_metrics(v2_preds, actuals)
    v1_m = compute_metrics(v1_preds, actuals)

    print(f"\n{'='*65}")
    print(f"  V1 (ELO) vs V2 (Gradient Boosting) — Walk-Forward Results")
    print(f"{'='*65}")
    print(f"  Test matches: {v2_m['n']}")
    print()
    print(f"  {'Metric':20} {'V1 ELO':>10} {'V2 GBM':>10} {'Change':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Accuracy':20} {v1_m['accuracy']:10.1%} {v2_m['accuracy']:10.1%} {(v2_m['accuracy']-v1_m['accuracy'])*100:+10.1f}pp")
    print(f"  {'Brier Score':20} {v1_m['brier']:10.4f} {v2_m['brier']:10.4f} {v2_m['brier']-v1_m['brier']:+10.4f}")
    print(f"  {'Log Loss':20} {v1_m['log_loss']:10.4f} {v2_m['log_loss']:10.4f} {v2_m['log_loss']-v1_m['log_loss']:+10.4f}")

    if args.compare:
        print(f"\n  V2 Calibration:")
        calibration_table(v2_preds, actuals)
        print(f"\n  V1 Calibration:")
        calibration_table(v1_preds, actuals)

    # Train final model on all data and save
    logger.info("Training final v2 model on all data…")
    X = np.nan_to_num(df[FEATURE_COLS].values, nan=0.0)
    y = df["result"].values
    final_model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=20, random_state=42,
    )
    final_model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    logger.info(f"V2 model saved → {MODEL_PATH}")

    # Feature importances
    importances = final_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n  Top 10 features:")
    for i in range(min(10, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"    {FEATURE_COLS[idx]:25} {importances[idx]:.4f}")
    print()


if __name__ == "__main__":
    main()
