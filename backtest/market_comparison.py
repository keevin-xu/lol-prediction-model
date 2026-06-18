"""
Model vs Market comparison — three independent analyses:

1. Model vs Bookmaker: Is our model sharper than Pinnacle closing lines?
2. Model vs Polymarket: Can we find +EV on Polymarket specifically?
3. Bookmaker vs Polymarket: Is Polymarket mispriced vs sharp book lines?

Run:
  python backtest/market_comparison.py                    # all comparisons
  python backtest/market_comparison.py --venue bookmaker   # model vs book only
  python backtest/market_comparison.py --venue polymarket   # model vs PM only
  python backtest/market_comparison.py --venue cross        # book vs PM only
  python backtest/market_comparison.py --league LCKC        # filter by league
"""

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from backtest.backtest import run_backtest
from model.pro_elo import HALF_LIFE_DAYS

DB_PATH = _ROOT / "db" / "lol_model.db"


@dataclass
class ComparisonRow:
    gameid: str
    date: str
    league: str
    blue_team: str
    red_team: str
    actual: float
    model_prob: float
    market_prob: Optional[float]
    market_source: str


# ---------------------------------------------------------------------------
# Build comparison datasets
# ---------------------------------------------------------------------------
def _gameid_to_match_id() -> Dict[str, int]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, gameid FROM matches").fetchall()
    conn.close()
    return {gid: mid for mid, gid in rows}


def build_model_vs_bookmaker(
    league_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
) -> List[ComparisonRow]:
    """Join walk-forward model predictions with bookmaker odds."""
    logger.info("Running walk-forward backtest for model predictions…")
    result = run_backtest(
        K=64, blend_k=5, scale=400, half_life_days=HALF_LIFE_DAYS,
        league_filter=league_filter,
    )

    if not result.predictions_with_ids:
        logger.warning("No predictions from backtest")
        return []

    gameid_map = _gameid_to_match_id()
    model_by_match: Dict[int, Tuple[str, float, float, str, str, str]] = {}

    conn = sqlite3.connect(DB_PATH)
    for gameid, p_blue, actual in result.predictions_with_ids:
        match_id = gameid_map.get(gameid)
        if not match_id:
            continue
        row = conn.execute(
            "SELECT date, league, blue_team, red_team FROM matches WHERE id = ?",
            (match_id,),
        ).fetchone()
        if row:
            model_by_match[match_id] = (gameid, p_blue, actual, row[0], row[1], row[2], row[3])

    query = """
        SELECT match_id, no_vig_prob_a, source, team_a_db, team_b_db
        FROM bookmaker_odds
        WHERE match_id IS NOT NULL AND no_vig_prob_a IS NOT NULL
    """
    params = []
    if source_filter:
        query += " AND source = ?"
        params.append(source_filter)

    book_rows = conn.execute(query, params).fetchall()
    conn.close()

    book_by_match: Dict[int, Tuple[float, str]] = {}
    for match_id, prob_a, source, team_a_db, team_b_db in book_rows:
        book_by_match[match_id] = (prob_a, source)

    rows = []
    for match_id in set(model_by_match.keys()) & set(book_by_match.keys()):
        gameid, p_blue, actual, date, league, blue, red = model_by_match[match_id]
        book_prob, source = book_by_match[match_id]

        # bookmaker_odds stores prob for team_a which may be blue or red
        conn2 = sqlite3.connect(DB_PATH)
        bo = conn2.execute(
            "SELECT team_a_db FROM bookmaker_odds WHERE match_id = ? LIMIT 1",
            (match_id,),
        ).fetchone()
        conn2.close()

        if bo and bo[0] == blue:
            book_prob_blue = book_prob
        elif bo and bo[0] == red:
            book_prob_blue = 1.0 - book_prob
        else:
            book_prob_blue = book_prob

        rows.append(ComparisonRow(
            gameid=gameid, date=date, league=league,
            blue_team=blue, red_team=red, actual=actual,
            model_prob=p_blue, market_prob=book_prob_blue,
            market_source=source,
        ))

    logger.info(f"Model vs Bookmaker: {len(rows)} matched rows")
    return rows


def build_model_vs_polymarket() -> List[ComparisonRow]:
    """Join model predictions with Polymarket closing prices."""
    conn = sqlite3.connect(DB_PATH)
    resolved = conn.execute(
        """
        SELECT market_id, db_team_a, db_team_b,
               closing_price_a, closing_price_b, resolution_winner
        FROM polymarket_markets
        WHERE status = 'resolved' AND closing_price_a IS NOT NULL
        """
    ).fetchall()
    conn.close()

    if not resolved:
        logger.info("No resolved Polymarket markets to compare")
        return []

    from model.predict import predict_match
    rows = []
    for market_id, team_a, team_b, close_a, close_b, winner in resolved:
        result = predict_match(team_a, team_b)
        actual = 1.0 if winner == team_a else 0.0

        rows.append(ComparisonRow(
            gameid=market_id, date="", league="polymarket",
            blue_team=team_a, red_team=team_b, actual=actual,
            model_prob=result["p_a"], market_prob=close_a,
            market_source="polymarket",
        ))

    logger.info(f"Model vs Polymarket: {len(rows)} resolved markets")
    return rows


def build_book_vs_polymarket() -> List[ComparisonRow]:
    """Compare bookmaker odds against Polymarket closing prices for same matches."""
    # This requires matches that appear on both platforms — likely very rare
    # Placeholder for when data accumulates
    logger.info("Bookmaker vs Polymarket: not enough overlapping data yet")
    return []


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    probs: List[float],
    actuals: List[float],
    label: str,
) -> Dict[str, float]:
    ps = np.array(probs)
    acts = np.array(actuals)
    n = len(ps)

    correct = sum(1 for p, a in zip(probs, actuals) if (p >= 0.5) == (a == 1.0))
    accuracy = correct / n if n > 0 else 0
    brier = float(np.mean((ps - acts) ** 2)) if n > 0 else 0

    eps = 1e-10
    ps_c = np.clip(ps, eps, 1 - eps)
    log_loss = float(-np.mean(acts * np.log(ps_c) + (1 - acts) * np.log(1 - ps_c))) if n > 0 else 0

    return {"label": label, "n": n, "accuracy": accuracy, "brier": brier, "log_loss": log_loss}


def calibration_table(probs: List[float], actuals: List[float]) -> List[Tuple]:
    bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
            (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.01)]
    table = []
    for lo, hi in bins:
        fps, fas = [], []
        for p, a in zip(probs, actuals):
            fav_p = max(p, 1 - p)
            fav_a = a if p >= 0.5 else 1 - a
            if lo <= fav_p < hi:
                fps.append(fav_p)
                fas.append(fav_a)
        if fps:
            table.append((f"{lo:.0%}-{hi:.0%}", sum(fps)/len(fps), sum(fas)/len(fas), len(fps)))
    return table


# ---------------------------------------------------------------------------
# Edge simulation
# ---------------------------------------------------------------------------
def simulate_edges(
    rows: List[ComparisonRow],
    thresholds: Optional[List[float]] = None,
) -> List[Dict]:
    thresholds = thresholds or [0.03, 0.05, 0.07, 0.10, 0.15]
    results = []

    for min_edge in thresholds:
        bets = 0
        wins = 0
        total_pnl = 0.0

        for r in rows:
            if r.market_prob is None:
                continue
            edge = r.model_prob - r.market_prob
            if abs(edge) < min_edge:
                continue

            # Bet on model's side
            if edge > 0:
                bet_on_blue = True
                entry_price = r.market_prob
            else:
                bet_on_blue = False
                entry_price = 1.0 - r.market_prob

            if entry_price <= 0 or entry_price >= 1:
                continue

            won = (bet_on_blue and r.actual == 1.0) or (not bet_on_blue and r.actual == 0.0)
            pnl = (1.0 / entry_price - 1.0) if won else -1.0

            bets += 1
            if won:
                wins += 1
            total_pnl += pnl

        roi = total_pnl / bets if bets > 0 else 0
        win_rate = wins / bets if bets > 0 else 0

        results.append({
            "min_edge": min_edge,
            "bets": bets,
            "wins": wins,
            "win_rate": win_rate,
            "total_pnl_units": round(total_pnl, 2),
            "roi": roi,
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_comparison(rows: List[ComparisonRow], venue: str) -> None:
    if not rows:
        print(f"\n  No data for {venue} comparison.\n")
        return

    model_probs = [r.model_prob for r in rows]
    market_probs = [r.market_prob for r in rows if r.market_prob is not None]
    actuals = [r.actual for r in rows]
    actuals_mkt = [r.actual for r in rows if r.market_prob is not None]

    model_m = compute_metrics(model_probs, actuals, "Model")
    market_m = compute_metrics(market_probs, actuals_mkt, venue.capitalize())

    source = rows[0].market_source if rows else "?"
    dates = [r.date for r in rows if r.date]
    date_range = f"{min(dates)} → {max(dates)}" if dates else "N/A"

    print(f"\n{'='*65}")
    print(f"  MODEL vs {venue.upper()}")
    print(f"{'='*65}")
    print(f"  Matched rows:  {len(rows)}")
    print(f"  Source:         {source}")
    print(f"  Date range:     {date_range}")
    print()
    print(f"  {'Metric':20} {'Model':>10} {venue.capitalize():>12} {'Diff':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Accuracy':20} {model_m['accuracy']:10.1%} {market_m['accuracy']:12.1%} {(model_m['accuracy']-market_m['accuracy'])*100:+10.1f}pp")
    print(f"  {'Brier Score':20} {model_m['brier']:10.4f} {market_m['brier']:12.4f} {model_m['brier']-market_m['brier']:+10.4f}")
    print(f"  {'Log Loss':20} {model_m['log_loss']:10.4f} {market_m['log_loss']:12.4f} {model_m['log_loss']-market_m['log_loss']:+10.4f}")

    # Calibration
    print(f"\n  Calibration (model):")
    print(f"  {'Bucket':12} {'Predicted':>10} {'Actual':>10} {'Count':>7}")
    print(f"  {'-'*42}")
    for label, pred, actual, count in calibration_table(model_probs, actuals):
        print(f"  {label:12} {pred:10.1%} {actual:10.1%} {count:7}")

    print(f"\n  Calibration ({venue}):")
    print(f"  {'Bucket':12} {'Predicted':>10} {'Actual':>10} {'Count':>7}")
    print(f"  {'-'*42}")
    for label, pred, actual, count in calibration_table(market_probs, actuals_mkt):
        print(f"  {label:12} {pred:10.1%} {actual:10.1%} {count:7}")

    # Edge simulation
    edge_results = simulate_edges(rows)
    print(f"\n  Edge Simulation (betting model vs {venue} when disagreement > threshold):")
    print(f"  {'Edge >':>8} {'Bets':>7} {'Wins':>7} {'Win%':>8} {'ROI':>8} {'P&L (units)':>12}")
    print(f"  {'-'*55}")
    for er in edge_results:
        if er["bets"] > 0:
            print(
                f"  {er['min_edge']:8.0%} {er['bets']:7} {er['wins']:7} "
                f"{er['win_rate']:8.1%} {er['roi']:8.1%} {er['total_pnl_units']:12.2f}"
            )
        else:
            print(f"  {er['min_edge']:8.0%}       0       -        -        -            -")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Model vs Market comparison")
    parser.add_argument("--venue", choices=["bookmaker", "polymarket", "cross", "all"], default="all")
    parser.add_argument("--league", type=str, default=None)
    parser.add_argument("--source", type=str, default=None, help="Bookmaker source filter")
    args = parser.parse_args()

    if args.venue in ("bookmaker", "all"):
        rows = build_model_vs_bookmaker(league_filter=args.league, source_filter=args.source)
        print_comparison(rows, "bookmaker")

    if args.venue in ("polymarket", "all"):
        rows = build_model_vs_polymarket()
        print_comparison(rows, "polymarket")

    if args.venue in ("cross", "all"):
        rows = build_book_vs_polymarket()
        if rows:
            print_comparison(rows, "cross-venue")
        elif args.venue == "cross":
            print("\n  Bookmaker vs Polymarket: not enough overlapping data yet.\n")


if __name__ == "__main__":
    main()
