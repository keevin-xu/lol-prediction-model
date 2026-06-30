"""
Walk-forward backtester for the LoL T2 prediction model.

Processes matches chronologically — predicts BEFORE seeing the result,
updates ELO AFTER. No lookahead, no data leakage.

Run:
  python backtest/backtest.py                # default params, full report
  python backtest/backtest.py --optimize     # grid search over K/blend_k/scale
  python backtest/backtest.py --league LCKC  # backtest single league
"""

import argparse
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from model.pro_elo import (
    DEFAULT_ELO,
    HALF_LIFE_DAYS,
    LEAGUE_TO_REGION,
    _apply_decay,
    _parse_date,
    compute_regional_offsets,
    expected_score,
    get_team_soloq_elos,
)

DB_PATH = _ROOT / "db" / "lol_model.db"

WARMUP_MONTHS = 3  # process this many months before logging predictions


# ---------------------------------------------------------------------------
# Incremental ELO tracker
# ---------------------------------------------------------------------------
class ELOTracker:
    """
    Stateful ELO tracker for walk-forward backtesting.
    Predict BEFORE result, update AFTER. Applies time-decay between matches.
    """

    def __init__(
        self,
        K: float = 48,
        blend_k: int = 5,
        scale: float = 500.0,
        half_life_days: float = HALF_LIFE_DAYS,
        soloq_baselines: Optional[Dict[str, float]] = None,
        regional_offsets: Optional[Dict[str, float]] = None,
        mov_weight: float = 0.0,
    ) -> None:
        self.K = K
        self.blend_k = blend_k
        self.scale = scale
        self.half_life_days = half_life_days
        self.elos: Dict[str, float] = {}
        self.games: Dict[str, int] = {}
        self.last_played: Dict[str, datetime] = {}
        self.soloq_baselines = soloq_baselines or {}
        self.regional_offsets = regional_offsets or {}
        self.mov_weight = mov_weight

    def _init_team(self, team: str, league: str) -> None:
        if team in self.elos:
            return
        if team in self.soloq_baselines:
            self.elos[team] = self.soloq_baselines[team]
        else:
            region = LEAGUE_TO_REGION.get(league, "")
            offset = self.regional_offsets.get(region, 0.0)
            self.elos[team] = DEFAULT_ELO + offset
        self.games[team] = 0

    def _decay_team(self, team: str, match_dt: datetime) -> None:
        if team in self.last_played:
            days = (match_dt - self.last_played[team]).days
            self.elos[team] = _apply_decay(self.elos[team], days, self.half_life_days)

    def get_blended_rating(self, team: str, league: str) -> float:
        self._init_team(team, league)
        pro_elo = self.elos[team]
        soloq_elo = self.soloq_baselines.get(team, DEFAULT_ELO)
        gp = self.games[team]
        alpha = gp / (gp + self.blend_k)
        return alpha * pro_elo + (1.0 - alpha) * soloq_elo

    def predict(self, blue: str, red: str, league: str, date: str) -> float:
        """Return P(blue wins) without updating state."""
        self._init_team(blue, league)
        self._init_team(red, league)

        match_dt = _parse_date(date)
        blue_elo = _apply_decay(
            self.elos[blue],
            (match_dt - self.last_played[blue]).days if blue in self.last_played else 0,
            self.half_life_days,
        )
        red_elo = _apply_decay(
            self.elos[red],
            (match_dt - self.last_played[red]).days if red in self.last_played else 0,
            self.half_life_days,
        )

        soloq_blue = self.soloq_baselines.get(blue, DEFAULT_ELO)
        soloq_red = self.soloq_baselines.get(red, DEFAULT_ELO)
        gp_blue = self.games.get(blue, 0)
        gp_red = self.games.get(red, 0)
        alpha_blue = gp_blue / (gp_blue + self.blend_k)
        alpha_red = gp_red / (gp_red + self.blend_k)

        rating_a = alpha_blue * blue_elo + (1.0 - alpha_blue) * soloq_blue
        rating_b = alpha_red * red_elo + (1.0 - alpha_red) * soloq_red

        return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / self.scale))

    def _mov_multiplier(
        self,
        winner_kills: Optional[int],
        loser_kills: Optional[int],
        winner_gd15: Optional[float],
    ) -> float:
        """
        Margin-of-victory K multiplier.
        Combines kill differential and gold diff at 15 into a scaling factor.
        Returns 1.0 when mov_weight is 0 or stats are missing.
        """
        if self.mov_weight == 0:
            return 1.0

        signals = []

        if winner_kills is not None and loser_kills is not None:
            kill_diff = winner_kills - loser_kills
            kill_signal = math.log1p(max(kill_diff, 0)) / math.log1p(20)
            signals.append(min(kill_signal, 1.5))

        if winner_gd15 is not None:
            gd_signal = max(winner_gd15, 0) / 5000.0
            signals.append(min(gd_signal, 1.5))

        if not signals:
            return 1.0

        avg_signal = sum(signals) / len(signals)
        return 1.0 + self.mov_weight * avg_signal

    def update(
        self,
        blue: str,
        red: str,
        winner: str,
        league: str,
        date: str,
        blue_kills: Optional[int] = None,
        red_kills: Optional[int] = None,
        blue_deaths: Optional[int] = None,
        red_deaths: Optional[int] = None,
        blue_gd15: Optional[float] = None,
        red_gd15: Optional[float] = None,
    ) -> None:
        """Update ELOs after seeing result, with time decay and optional MOV scaling."""
        self._init_team(blue, league)
        self._init_team(red, league)

        match_dt = _parse_date(date)

        self._decay_team(blue, match_dt)
        self._decay_team(red, match_dt)

        blue_elo = self.elos[blue]
        red_elo = self.elos[red]

        blue_exp = expected_score(blue_elo, red_elo)
        blue_actual = 1.0 if winner == "blue" else 0.0

        if winner == "blue":
            mov_mult = self._mov_multiplier(blue_kills, red_kills, blue_gd15)
        else:
            mov_mult = self._mov_multiplier(red_kills, blue_kills, red_gd15)

        k_adj = self.K * mov_mult

        self.elos[blue] = blue_elo + k_adj * (blue_actual - blue_exp)
        self.elos[red] = red_elo + k_adj * ((1.0 - blue_actual) - (1.0 - blue_exp))
        self.games[blue] = self.games.get(blue, 0) + 1
        self.games[red] = self.games.get(red, 0) + 1
        self.last_played[blue] = match_dt
        self.last_played[red] = match_dt


# ---------------------------------------------------------------------------
# Backtest result
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    K: float
    blend_k: int
    scale: float
    half_life: float = HALF_LIFE_DAYS
    mov_weight: float = 0.0
    use_soloq: bool = True
    warmup_matches: int = 0
    test_matches: int = 0
    correct: int = 0
    accuracy: float = 0.0
    brier_score: float = 0.0
    log_loss: float = 0.0
    calibration: Dict[str, Tuple[float, float, int]] = field(default_factory=dict)
    league_accuracy: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    predictions: List[Tuple[float, float]] = field(default_factory=list)
    predictions_with_ids: List[Tuple[str, float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------
def load_matches(league_filter: Optional[str] = None) -> List[Tuple]:
    cols = "gameid, date, league, blue_team, red_team, winner, blue_kills, red_kills, blue_deaths, red_deaths, blue_golddiffat15, red_golddiffat15"
    conn = sqlite3.connect(DB_PATH)
    if league_filter:
        query = f"SELECT {cols} FROM matches WHERE league = ? ORDER BY date ASC, gameid ASC"
        rows = conn.execute(query, (league_filter,)).fetchall()
    else:
        query = f"SELECT {cols} FROM matches ORDER BY date ASC, gameid ASC"
        rows = conn.execute(query).fetchall()
    conn.close()
    return rows


def run_backtest(
    K: float = 48,
    blend_k: int = 5,
    scale: float = 500.0,
    half_life_days: float = HALF_LIFE_DAYS,
    warmup_months: int = WARMUP_MONTHS,
    league_filter: Optional[str] = None,
    verbose: bool = False,
    use_soloq: bool = True,
    mov_weight: float = 0.0,
    eval_start: Optional[str] = None,
    eval_end: Optional[str] = None,
) -> BacktestResult:
    """
    Run walk-forward backtest over all matches.

    Returns BacktestResult with all metrics.
    """
    soloq_baselines = get_team_soloq_elos() if use_soloq else {}
    regional_offsets = compute_regional_offsets() if use_soloq else {}

    tracker = ELOTracker(
        K=K,
        blend_k=blend_k,
        scale=scale,
        half_life_days=half_life_days,
        soloq_baselines=soloq_baselines,
        regional_offsets=regional_offsets,
        mov_weight=mov_weight,
    )

    matches = load_matches(league_filter)
    if not matches:
        logger.error("No matches found")
        return BacktestResult(K=K, blend_k=blend_k, scale=scale, half_life=half_life_days, mov_weight=mov_weight, use_soloq=use_soloq)

    # Determine warmup cutoff date
    if eval_start:
        warmup_cutoff = eval_start
    else:
        first_date = matches[0][1]
        warmup_year = int(first_date[:4])
        warmup_month = int(first_date[5:7]) + warmup_months
        while warmup_month > 12:
            warmup_month -= 12
            warmup_year += 1
        warmup_cutoff = f"{warmup_year:04d}-{warmup_month:02d}-01"

    # Walk-forward
    result = BacktestResult(K=K, blend_k=blend_k, scale=scale, half_life=half_life_days, mov_weight=mov_weight, use_soloq=use_soloq)
    predictions: List[Tuple[float, float]] = []
    league_stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    for row in matches:
        gameid, date, league, blue, red, winner = row[:6]
        blue_kills, red_kills, blue_deaths, red_deaths = row[6], row[7], row[8], row[9]
        blue_gd15, red_gd15 = row[10], row[11]

        stats = dict(
            blue_kills=blue_kills, red_kills=red_kills,
            blue_deaths=blue_deaths, red_deaths=red_deaths,
            blue_gd15=blue_gd15, red_gd15=red_gd15,
        )

        if date < warmup_cutoff:
            tracker.update(blue, red, winner, league, date, **stats)
            result.warmup_matches += 1
            continue

        if eval_end and date > eval_end:
            tracker.update(blue, red, winner, league, date, **stats)
            continue

        # Predict BEFORE seeing result
        p_blue = tracker.predict(blue, red, league, date)
        actual = 1.0 if winner == "blue" else 0.0

        predictions.append((p_blue, actual))
        result.predictions_with_ids.append((gameid, p_blue, actual))

        # Track accuracy
        predicted_winner = p_blue >= 0.5
        actual_winner = actual == 1.0
        if predicted_winner == actual_winner:
            result.correct += 1
        league_stats[league][0] += 1  # total
        if predicted_winner == actual_winner:
            league_stats[league][1] += 1  # correct

        # Update AFTER seeing result
        tracker.update(blue, red, winner, league, date, **stats)

    result.test_matches = len(predictions)
    result.predictions = predictions

    if not predictions:
        return result

    # Compute metrics
    preds = np.array([p for p, _ in predictions])
    actuals = np.array([a for _, a in predictions])

    result.accuracy = result.correct / result.test_matches

    # Brier score: mean(( p - actual )^2)
    result.brier_score = float(np.mean((preds - actuals) ** 2))

    # Log loss: -mean(actual*log(p) + (1-actual)*log(1-p))
    eps = 1e-10
    preds_clipped = np.clip(preds, eps, 1.0 - eps)
    result.log_loss = float(-np.mean(
        actuals * np.log(preds_clipped) + (1 - actuals) * np.log(1 - preds_clipped)
    ))

    # Calibration: bin predictions into buckets
    bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
            (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.01)]
    for lo, hi in bins:
        # Use the "favorite" probability (always >= 0.5)
        mask = []
        fav_actuals = []
        for p, a in predictions:
            fav_p = max(p, 1.0 - p)
            fav_a = a if p >= 0.5 else 1.0 - a
            if lo <= fav_p < hi:
                mask.append(fav_p)
                fav_actuals.append(fav_a)
        if mask:
            avg_pred = sum(mask) / len(mask)
            avg_actual = sum(fav_actuals) / len(fav_actuals)
            label = f"{lo:.0%}-{hi:.0%}"
            result.calibration[label] = (avg_pred, avg_actual, len(mask))

    # Per-league accuracy
    result.league_accuracy = {
        league: (correct, total)
        for league, (total, correct) in league_stats.items()
    }

    return result


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def grid_search(
    k_values: List[float] = None,
    blend_k_values: List[int] = None,
    scale_values: List[float] = None,
    half_life_values: List[float] = None,
    mov_weight_values: List[float] = None,
    league_filter: Optional[str] = None,
    use_soloq: bool = True,
    eval_start: Optional[str] = None,
    eval_end: Optional[str] = None,
) -> List[BacktestResult]:
    """
    Sweep parameter combinations. Returns results sorted by Brier score (best first).
    """
    k_values = k_values or [32, 48, 64]
    blend_k_values = blend_k_values or [5, 10, 20]
    scale_values = scale_values or [400, 500]
    half_life_values = half_life_values or [90, 180, 270, 365, 9999]
    mov_weight_values = mov_weight_values or [0.0]

    total = len(k_values) * len(blend_k_values) * len(scale_values) * len(half_life_values) * len(mov_weight_values)
    logger.info(f"Grid search: {total} combinations")

    results: List[BacktestResult] = []
    done = 0

    for k in k_values:
        for bk in blend_k_values:
            for sc in scale_values:
                for hl in half_life_values:
                    for mw in mov_weight_values:
                        r = run_backtest(
                            K=k, blend_k=bk, scale=sc, half_life_days=hl,
                            league_filter=league_filter, use_soloq=use_soloq,
                            mov_weight=mw, eval_start=eval_start, eval_end=eval_end,
                        )
                        results.append(r)
                        done += 1
                        if done % 20 == 0:
                            logger.info(f"  {done}/{total} completed")

    results.sort(key=lambda r: r.brier_score)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(result: BacktestResult) -> None:
    print(f"\n{'='*60}")
    print(f"  BACKTEST REPORT")
    print(f"{'='*60}")
    hl_str = "off" if result.half_life >= 9000 else f"{result.half_life:.0f}d"
    mov_str = f", mov_weight={result.mov_weight}" if result.mov_weight > 0 else ""
    soloq_str = ", soloq=OFF" if not result.use_soloq else ""
    print(f"  Parameters: K={result.K}, blend_k={result.blend_k}, scale={result.scale}, half_life={hl_str}{mov_str}{soloq_str}")
    print(f"  Warmup:     {result.warmup_matches} matches")
    print(f"  Test:       {result.test_matches} matches")
    print()
    print(f"  Accuracy:    {result.accuracy:.1%}  ({result.correct}/{result.test_matches})")
    print(f"  Brier Score: {result.brier_score:.4f}  (lower=better, random=0.2500)")
    print(f"  Log Loss:    {result.log_loss:.4f}  (lower=better, random=0.6931)")
    print()

    # Calibration table
    print(f"  {'Bucket':12} {'Predicted':>10} {'Actual':>10} {'Count':>8}")
    print(f"  {'-'*42}")
    for label, (pred, actual, count) in sorted(result.calibration.items()):
        diff = actual - pred
        arrow = "+" if diff > 0.02 else ("-" if diff < -0.02 else " ")
        print(f"  {label:12} {pred:10.1%} {actual:10.1%} {count:8d}  {arrow}")
    print()

    # Per-league breakdown
    league_data = sorted(result.league_accuracy.items(), key=lambda x: x[1][0] / max(x[1][1], 1), reverse=True)
    print(f"  {'League':10} {'Accuracy':>10} {'Correct':>8} {'Total':>8}")
    print(f"  {'-'*38}")
    for league, (correct, total) in league_data:
        acc = correct / total if total > 0 else 0
        print(f"  {league:10} {acc:10.1%} {correct:8d} {total:8d}")
    print()


def print_grid_results(results: List[BacktestResult], top_n: int = 15) -> None:
    print(f"\n{'='*85}")
    print(f"  GRID SEARCH RESULTS (top {top_n} of {len(results)} by Brier Score)")
    print(f"{'='*85}")
    print(f"  {'#':>3}  {'K':>4}  {'blend_k':>7}  {'scale':>6}  {'half_life':>9}  {'mov_w':>6}  {'Brier':>8}  {'LogLoss':>8}  {'Accuracy':>9}")
    print(f"  {'-'*73}")
    for i, r in enumerate(results[:top_n]):
        hl = "off" if r.half_life >= 9000 else f"{r.half_life:.0f}d"
        print(f"  {i+1:3}  {r.K:4.0f}  {r.blend_k:7d}  {r.scale:6.0f}  {hl:>9}  {r.mov_weight:6.2f}  {r.brier_score:8.4f}  {r.log_loss:8.4f}  {r.accuracy:9.1%}")

    print(f"\n  Worst:")
    worst = results[-1]
    hl = "off" if worst.half_life >= 9000 else f"{worst.half_life:.0f}d"
    print(f"       {worst.K:4.0f}  {worst.blend_k:7d}  {worst.scale:6.0f}  {hl:>9}  {worst.mov_weight:6.2f}  {worst.brier_score:8.4f}  {worst.log_loss:8.4f}  {worst.accuracy:9.1%}")
    print()

    best = results[0]
    hl = "off" if best.half_life >= 9000 else f"{best.half_life:.0f}d"
    mov_str = f", mov_weight={best.mov_weight}" if best.mov_weight > 0 else ""
    print(f"  BEST: K={best.K:.0f}, blend_k={best.blend_k}, scale={best.scale}, half_life={hl}{mov_str}")
    print(f"    Brier={best.brier_score:.4f}  LogLoss={best.log_loss:.4f}  Accuracy={best.accuracy:.1%}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LoL T2 model backtester")
    parser.add_argument("--optimize", action="store_true", help="Run grid search over K/blend_k/scale/half_life")
    parser.add_argument("--league", type=str, default=None, help="Filter to a single league")
    parser.add_argument("--K", type=float, default=48, help="ELO K-factor")
    parser.add_argument("--blend-k", type=int, default=5, help="Blend denominator")
    parser.add_argument("--scale", type=float, default=500, help="ELO scale factor")
    parser.add_argument("--half-life", type=float, default=HALF_LIFE_DAYS, help="ELO decay half-life in days (9999=off)")
    parser.add_argument("--mov-weight", type=float, default=0.0, help="Margin-of-victory K scaling weight (0=off, try 0.5-2.0)")
    parser.add_argument("--no-soloq", action="store_true", help="Disable soloq baselines (pro ELO only)")
    parser.add_argument("--ablation", action="store_true", help="Run soloq ablation: compare with/without soloq baselines")
    parser.add_argument("--eval-start", type=str, default=None, help="Start of evaluation window (YYYY-MM-DD). Earlier matches used for warmup only.")
    parser.add_argument("--eval-end", type=str, default=None, help="End of evaluation window (YYYY-MM-DD). Later matches still update ELO but aren't scored.")
    args = parser.parse_args()

    eval_kw = dict(eval_start=args.eval_start, eval_end=args.eval_end)

    if args.ablation:
        logger.info("Running soloq ablation study…")
        print(f"\n{'='*60}")
        print(f"  SOLOQ ABLATION STUDY")
        print(f"{'='*60}")
        r_with = run_backtest(K=args.K, blend_k=args.blend_k, scale=args.scale,
                              half_life_days=args.half_life, league_filter=args.league,
                              use_soloq=True, mov_weight=args.mov_weight, **eval_kw)
        r_without = run_backtest(K=args.K, blend_k=args.blend_k, scale=args.scale,
                                 half_life_days=args.half_life, league_filter=args.league,
                                 use_soloq=False, mov_weight=args.mov_weight, **eval_kw)
        print(f"\n  {'Metric':<15} {'With SoloQ':>12} {'Without SoloQ':>14} {'Delta':>10}")
        print(f"  {'-'*53}")
        print(f"  {'Accuracy':<15} {r_with.accuracy:12.1%} {r_without.accuracy:14.1%} {r_with.accuracy - r_without.accuracy:+10.1%}")
        print(f"  {'Brier Score':<15} {r_with.brier_score:12.4f} {r_without.brier_score:14.4f} {r_with.brier_score - r_without.brier_score:+10.4f}")
        print(f"  {'Log Loss':<15} {r_with.log_loss:12.4f} {r_without.log_loss:14.4f} {r_with.log_loss - r_without.log_loss:+10.4f}")
        print()
        winner = "With SoloQ" if r_with.brier_score < r_without.brier_score else "Without SoloQ"
        print(f"  Winner (by Brier): {winner}")
        print()

    elif args.optimize:
        logger.info("Starting parameter optimization…")
        mov_values = [0.0, 0.25, 0.5, 1.0, 1.5] if args.mov_weight == 0.0 else [args.mov_weight]
        results = grid_search(
            league_filter=args.league,
            use_soloq=not args.no_soloq,
            mov_weight_values=mov_values,
            **eval_kw,
        )
        print_grid_results(results)

        best = results[0]
        hl = "off" if best.half_life >= 9000 else f"{best.half_life:.0f}d"
        logger.info(f"Running detailed report with best params: K={best.K}, blend_k={best.blend_k}, scale={best.scale}, half_life={hl}")
        detailed = run_backtest(K=best.K, blend_k=best.blend_k, scale=best.scale,
                                half_life_days=best.half_life, league_filter=args.league,
                                use_soloq=not args.no_soloq, mov_weight=best.mov_weight, **eval_kw)
        print_report(detailed)
    else:
        hl = "off" if args.half_life >= 9000 else f"{args.half_life:.0f}d"
        logger.info(f"Running backtest (K={args.K}, blend_k={args.blend_k}, scale={args.scale}, half_life={hl})…")
        result = run_backtest(K=args.K, blend_k=args.blend_k, scale=args.scale,
                              half_life_days=args.half_life, league_filter=args.league,
                              use_soloq=not args.no_soloq, mov_weight=args.mov_weight, **eval_kw)
        print_report(result)


if __name__ == "__main__":
    main()
