"""
V3 prediction model — gradient boosted classifier using expanded
team-level features + draft quality features.

V3 uses ~90 features vs V2's 28:
- 39 rolling stats per side (objectives, laning, economy, vision, tempo)
- 4 draft features per side (champion WR, meta score, comfort)
- 12+ differentials
- ELO difference from V1

Walk-forward training ensures no lookahead.

Run:
  python model/v2_model.py                # train + evaluate
  python model/v2_model.py --compare      # side-by-side v1 vs v3
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
from model.draft_features import build_draft_features
from model.pro_elo import compute_regional_offsets, get_team_soloq_elos

DB_PATH = _ROOT / "db" / "lol_model.db"
MODEL_PATH = _ROOT / "model" / "v3_model.pkl"


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Dynamically get all feature columns (everything except metadata and target)."""
    exclude = {"gameid", "date", "league", "result"}
    return [c for c in df.columns if c not in exclude]


def walk_forward_evaluate(
    df: pd.DataFrame,
    warmup_frac: float = 0.15,
    retrain_every: int = 500,
) -> Tuple[List[float], List[float], List[float]]:
    n = len(df)
    warmup_end = int(n * warmup_frac)
    if warmup_end < 1000:
        warmup_end = min(1000, n // 3)

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values
    y = df["result"].values

    # V1 ELO baseline for comparison
    soloq = get_team_soloq_elos()
    offsets = compute_regional_offsets()
    from backtest.backtest import load_matches
    matches = load_matches()
    first = matches[0][1]
    wy, wm = int(first[:4]), int(first[5:7]) + WARMUP_MONTHS
    while wm > 12:
        wm -= 12
        wy += 1
    cutoff = f'{wy:04d}-{wm:02d}-01'
    tracker = ELOTracker(K=64, blend_k=5, scale=400, half_life_days=270,
                         soloq_baselines=soloq, regional_offsets=offsets)

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

    v3_predictions: List[float] = []
    v1_predictions: List[float] = []
    actuals: List[float] = []

    model = None
    last_train_end = 0

    logger.info(f"Walk-forward: {warmup_end} warmup, {n - warmup_end} test, "
                f"{len(feature_cols)} features, retrain every {retrain_every}")

    for i in range(warmup_end, n):
        if model is None or (i - last_train_end) >= retrain_every:
            X_train = np.nan_to_num(X[:i], nan=0.0)
            y_train = y[:i]
            model = HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=3,
                learning_rate=0.05,
                min_samples_leaf=30,
                l2_regularization=1.0,
                max_features=0.5,
                random_state=42,
            )
            model.fit(X_train, y_train)
            last_train_end = i

        x_test = np.nan_to_num(X[i:i+1], nan=0.0)
        p_v3 = model.predict_proba(x_test)[0][1]
        v3_predictions.append(float(p_v3))

        gid = df.iloc[i]["gameid"]
        p_v1 = v1_preds_map.get(gid, 0.5)
        v1_predictions.append(float(p_v1))

        actuals.append(float(y[i]))

    return v3_predictions, v1_predictions, actuals


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
    parser = argparse.ArgumentParser(description="V3 model training and evaluation")
    parser.add_argument("--compare", action="store_true", help="Side-by-side v1 vs v3")
    args = parser.parse_args()

    logger.info("Building V3 feature dataset…")
    df = build_feature_dataset()
    if df.empty:
        logger.error("No feature data")
        return

    logger.info("Building draft features…")
    draft_df = build_draft_features()
    if not draft_df.empty:
        df = df.merge(draft_df, on="gameid", how="left")
        logger.info(f"Merged draft features → {len(df.columns)} total columns")

    logger.info("Running walk-forward evaluation…")
    v3_preds, v1_preds, actuals = walk_forward_evaluate(df)

    v3_m = compute_metrics(v3_preds, actuals)
    v1_m = compute_metrics(v1_preds, actuals)

    feature_cols = get_feature_cols(df)

    print(f"\n{'='*65}")
    print(f"  V1 (ELO) vs V3 (GBM + {len(feature_cols)} features) — Walk-Forward")
    print(f"{'='*65}")
    print(f"  Test matches: {v3_m['n']}")
    print()
    print(f"  {'Metric':20} {'V1 ELO':>10} {'V3 GBM':>10} {'Change':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Accuracy':20} {v1_m['accuracy']:10.1%} {v3_m['accuracy']:10.1%} {(v3_m['accuracy']-v1_m['accuracy'])*100:+10.1f}pp")
    print(f"  {'Brier Score':20} {v1_m['brier']:10.4f} {v3_m['brier']:10.4f} {v3_m['brier']-v1_m['brier']:+10.4f}")
    print(f"  {'Log Loss':20} {v1_m['log_loss']:10.4f} {v3_m['log_loss']:10.4f} {v3_m['log_loss']-v1_m['log_loss']:+10.4f}")

    if args.compare:
        print(f"\n  V3 Calibration:")
        calibration_table(v3_preds, actuals)
        print(f"\n  V1 Calibration:")
        calibration_table(v1_preds, actuals)

    # Train final model on all data and save
    logger.info("Training final V3 model on all data…")
    X = np.nan_to_num(df[feature_cols].values, nan=0.0)
    y = df["result"].values
    final_model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=30, l2_regularization=1.0,
        max_features=0.5, random_state=42,
    )
    final_model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": feature_cols}, f)
    logger.info(f"V3 model saved → {MODEL_PATH}")

    # Feature importances (HistGBM uses a different attribute name)
    try:
        importances = final_model.feature_importances_
    except AttributeError:
        importances = np.zeros(len(feature_cols))

    if importances.sum() > 0:
        sorted_idx = np.argsort(importances)[::-1]
        print(f"\n  Top 20 features:")
        for i in range(min(20, len(sorted_idx))):
            idx = sorted_idx[i]
            print(f"    {feature_cols[idx]:35} {importances[idx]:.4f}")
    print()


if __name__ == "__main__":
    main()
