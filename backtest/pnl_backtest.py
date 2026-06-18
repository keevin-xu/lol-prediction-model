"""
P&L-focused backtester — simulates paper trading the model's predictions
as if they were bet against Polymarket-style markets.

Assumes the market price equals the "true" probability (since we don't have
historical bookmaker odds). Tests what happens when the model disagrees with
a hypothetical market at various edge thresholds.

For each match:
  1. Model produces P(blue wins)
  2. Simulate a market price (use actual base rate as proxy)
  3. If model disagrees with market by > min_edge, place a bet
  4. Size using quarter-Kelly
  5. Track P&L, bankroll, drawdown, streaks

Run:
  python backtest/pnl_backtest.py                    # default params
  python backtest/pnl_backtest.py --edge 0.05        # 5% min edge
  python backtest/pnl_backtest.py --league LCKC      # single league
  python backtest/pnl_backtest.py --bankroll 5000    # custom starting bankroll
"""

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from backtest.backtest import (
    ELOTracker,
    WARMUP_MONTHS,
    load_matches,
)
from model.pro_elo import (
    HALF_LIFE_DAYS,
    compute_regional_offsets,
    get_team_soloq_elos,
)


@dataclass
class PnLResult:
    starting_bankroll: float
    final_bankroll: float
    total_pnl: float
    roi: float
    total_bets: int
    wins: int
    losses: int
    win_rate: float
    avg_edge: float
    avg_bet_size: float
    max_drawdown: float
    max_drawdown_pct: float
    peak_bankroll: float
    longest_win_streak: int
    longest_loss_streak: int
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    league_pnl: Dict[str, Dict] = field(default_factory=dict)
    edge_bucket_pnl: Dict[str, Dict] = field(default_factory=dict)
    bankroll_history: List[float] = field(default_factory=list)
    bet_log: List[Dict] = field(default_factory=list)


def kelly_fraction(model_prob: float, market_prob: float, max_kelly: float = 0.0625) -> float:
    if market_prob <= 0 or market_prob >= 1:
        return 0.0
    b = (1.0 / market_prob) - 1.0
    q = 1.0 - model_prob
    f = (model_prob * b - q) / b
    return max(0.0, min(f, max_kelly))


def run_pnl_backtest(
    K: float = 64,
    blend_k: int = 5,
    scale: float = 400,
    half_life_days: float = HALF_LIFE_DAYS,
    min_edge: float = 0.03,
    max_kelly: float = 0.0625,
    starting_bankroll: float = 1000.0,
    league_filter: Optional[str] = None,
    market_vig: float = 0.02,
) -> PnLResult:
    """
    Walk-forward backtest with full P&L simulation.

    Since we don't have historical market prices, we simulate the market
    as: market_prob = actual_base_rate + noise. In practice, we use a
    simple model: market_prob ≈ model_prob shifted by a random offset,
    simulating disagreement. But the cleanest approach is to assume the
    market is "perfect" (market_prob = long-run base rate for that
    confidence bucket) and see if our model's deviations from perfection
    are profitable.

    Simpler approach used here: for each match, the "market" price is
    the model's own prediction dampened toward 50% by the vig factor.
    This simulates a slightly less confident market that the model
    tries to beat.
    """
    soloq = get_team_soloq_elos()
    offsets = compute_regional_offsets()
    tracker = ELOTracker(
        K=K, blend_k=blend_k, scale=scale, half_life_days=half_life_days,
        soloq_baselines=soloq, regional_offsets=offsets,
    )

    matches = load_matches(league_filter)
    if not matches:
        logger.error("No matches found")
        return PnLResult(starting_bankroll=starting_bankroll, final_bankroll=starting_bankroll,
                         total_pnl=0, roi=0, total_bets=0, wins=0, losses=0, win_rate=0,
                         avg_edge=0, avg_bet_size=0, max_drawdown=0, max_drawdown_pct=0,
                         peak_bankroll=starting_bankroll, longest_win_streak=0, longest_loss_streak=0)

    first_date = matches[0][1]
    wy, wm = int(first_date[:4]), int(first_date[5:7]) + WARMUP_MONTHS
    while wm > 12:
        wm -= 12
        wy += 1
    warmup_cutoff = f"{wy:04d}-{wm:02d}-01"

    bankroll = starting_bankroll
    peak = starting_bankroll
    max_dd = 0.0
    max_dd_pct = 0.0
    wins = 0
    losses = 0
    total_edge = 0.0
    total_bet_size = 0.0
    win_streak = 0
    loss_streak = 0
    longest_win = 0
    longest_loss = 0
    monthly_pnl: Dict[str, float] = defaultdict(float)
    league_stats: Dict[str, Dict] = defaultdict(lambda: {"bets": 0, "pnl": 0.0, "wins": 0})
    edge_buckets: Dict[str, Dict] = defaultdict(lambda: {"bets": 0, "pnl": 0.0, "wins": 0})
    bankroll_history = [bankroll]
    bet_log: List[Dict] = []

    for gameid, date, league, blue, red, winner in matches:
        if date < warmup_cutoff:
            tracker.update(blue, red, winner, league, date)
            continue

        model_prob = tracker.predict(blue, red, league, date)
        actual = 1.0 if winner == "blue" else 0.0

        # The model's edge over a naive 50/50 market.
        # In reality, Polymarket prices converge toward the true probability,
        # so we simulate the market as 50% (no information) and the model
        # provides the edge. This tests: "if you could buy the favorite at
        # 50c whenever the model is confident, would you profit?"
        #
        # With market_vig > 0, the market is shifted away from 50/50 toward
        # the model's direction, simulating a partially-informed market.
        # vig=0.0 → market is 50/50 (maximum edge)
        # vig=0.5 → market is halfway between 50% and model (moderate edge)
        # vig=1.0 → market equals model (zero edge)
        market_blue = 0.5 * (1 - market_vig) + model_prob * market_vig
        market_red = 1.0 - market_blue

        edge_blue = model_prob - market_blue
        edge_red = (1.0 - model_prob) - market_red

        bet_side = None
        edge = 0.0
        entry_price = 0.0
        model_p = 0.0

        if edge_blue >= min_edge and edge_blue >= edge_red:
            bet_side = "blue"
            edge = edge_blue
            entry_price = market_blue
            model_p = model_prob
        elif edge_red >= min_edge:
            bet_side = "red"
            edge = edge_red
            entry_price = market_red
            model_p = 1.0 - model_prob

        if bet_side and bankroll > 1.0:
            fraction = kelly_fraction(model_p, entry_price, max_kelly)
            bet_amount = round(bankroll * fraction, 2)

            if bet_amount < 1.0:
                tracker.update(blue, red, winner, league, date)
                continue

            bet_won = (bet_side == "blue" and actual == 1.0) or (bet_side == "red" and actual == 0.0)

            if bet_won:
                pnl = bet_amount * (1.0 / entry_price - 1.0)
                wins += 1
                win_streak += 1
                loss_streak = 0
                longest_win = max(longest_win, win_streak)
            else:
                pnl = -bet_amount
                losses += 1
                loss_streak += 1
                win_streak = 0
                longest_loss = max(longest_loss, loss_streak)

            pnl = round(pnl, 2)
            bankroll += pnl
            total_edge += edge
            total_bet_size += bet_amount

            if bankroll > peak:
                peak = bankroll
            dd = peak - bankroll
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd / peak if peak > 0 else 0

            month_key = date[:7]
            monthly_pnl[month_key] += pnl
            league_stats[league]["bets"] += 1
            league_stats[league]["pnl"] += pnl
            if bet_won:
                league_stats[league]["wins"] += 1

            bucket = f"{int(edge * 100) // 3 * 3}-{int(edge * 100) // 3 * 3 + 3}%"
            edge_buckets[bucket]["bets"] += 1
            edge_buckets[bucket]["pnl"] += pnl
            if bet_won:
                edge_buckets[bucket]["wins"] += 1

            bankroll_history.append(bankroll)
            bet_log.append({
                "date": date[:10], "league": league,
                "blue": blue, "red": red,
                "bet_side": bet_side, "edge": edge,
                "bet_amount": bet_amount, "entry_price": entry_price,
                "won": bet_won, "pnl": pnl, "bankroll": bankroll,
            })

        tracker.update(blue, red, winner, league, date)

    total_bets = wins + losses
    return PnLResult(
        starting_bankroll=starting_bankroll,
        final_bankroll=round(bankroll, 2),
        total_pnl=round(bankroll - starting_bankroll, 2),
        roi=round((bankroll - starting_bankroll) / starting_bankroll, 4) if starting_bankroll > 0 else 0,
        total_bets=total_bets,
        wins=wins,
        losses=losses,
        win_rate=wins / total_bets if total_bets > 0 else 0,
        avg_edge=total_edge / total_bets if total_bets > 0 else 0,
        avg_bet_size=total_bet_size / total_bets if total_bets > 0 else 0,
        max_drawdown=round(max_dd, 2),
        max_drawdown_pct=round(max_dd_pct, 4),
        peak_bankroll=round(peak, 2),
        longest_win_streak=longest_win,
        longest_loss_streak=longest_loss,
        monthly_returns=dict(monthly_pnl),
        league_pnl=dict(league_stats),
        edge_bucket_pnl=dict(edge_buckets),
        bankroll_history=bankroll_history,
        bet_log=bet_log,
    )


def print_report(r: PnLResult, min_edge: float, max_kelly: float) -> None:
    print(f"\n{'='*70}")
    print(f"  P&L BACKTEST REPORT")
    print(f"{'='*70}")
    print(f"  Kelly cap: {max_kelly:.1%} (quarter-Kelly)  |  Min edge: {min_edge:.0%}")
    print(f"  Starting bankroll: ${r.starting_bankroll:,.2f}")
    print()

    print(f"  SUMMARY")
    print(f"  {'-'*50}")
    print(f"  Final bankroll:     ${r.final_bankroll:,.2f}")
    print(f"  Total P&L:          ${r.total_pnl:+,.2f}")
    print(f"  ROI:                {r.roi:+.1%}")
    print(f"  Total bets:         {r.total_bets}")
    print(f"  Record:             {r.wins}W / {r.losses}L ({r.win_rate:.1%})")
    print(f"  Avg edge per bet:   {r.avg_edge:.1%}")
    print(f"  Avg bet size:       ${r.avg_bet_size:,.2f}")
    print()

    print(f"  RISK")
    print(f"  {'-'*50}")
    print(f"  Peak bankroll:      ${r.peak_bankroll:,.2f}")
    print(f"  Max drawdown:       ${r.max_drawdown:,.2f} ({r.max_drawdown_pct:.1%})")
    print(f"  Longest win streak: {r.longest_win_streak}")
    print(f"  Longest loss streak:{r.longest_loss_streak}")
    print()

    if r.monthly_returns:
        print(f"  MONTHLY P&L")
        print(f"  {'-'*50}")
        for month in sorted(r.monthly_returns.keys()):
            pnl = r.monthly_returns[month]
            bar = "+" * int(min(pnl / 10, 30)) if pnl > 0 else "-" * int(min(-pnl / 10, 30))
            print(f"  {month}  ${pnl:+8,.2f}  {bar}")
        print()

    if r.league_pnl:
        print(f"  P&L BY LEAGUE")
        print(f"  {'League':10} {'Bets':>6} {'Wins':>6} {'Win%':>7} {'P&L':>10} {'Avg P&L':>10}")
        print(f"  {'-'*55}")
        for league in sorted(r.league_pnl.keys(), key=lambda x: r.league_pnl[x]["pnl"], reverse=True):
            s = r.league_pnl[league]
            wr = s["wins"] / s["bets"] if s["bets"] > 0 else 0
            avg = s["pnl"] / s["bets"] if s["bets"] > 0 else 0
            print(f"  {league:10} {s['bets']:6} {s['wins']:6} {wr:7.1%} ${s['pnl']:+9,.2f} ${avg:+9,.2f}")
        print()

    if r.edge_bucket_pnl:
        print(f"  P&L BY EDGE SIZE")
        print(f"  {'Edge':10} {'Bets':>6} {'Wins':>6} {'Win%':>7} {'P&L':>10} {'Avg P&L':>10}")
        print(f"  {'-'*55}")
        for bucket in sorted(r.edge_bucket_pnl.keys()):
            s = r.edge_bucket_pnl[bucket]
            wr = s["wins"] / s["bets"] if s["bets"] > 0 else 0
            avg = s["pnl"] / s["bets"] if s["bets"] > 0 else 0
            print(f"  {bucket:10} {s['bets']:6} {s['wins']:6} {wr:7.1%} ${s['pnl']:+9,.2f} ${avg:+9,.2f}")
        print()

    # Show last 20 bets
    if r.bet_log:
        print(f"  RECENT BETS (last 20)")
        print(f"  {'Date':12} {'League':6} {'Bet On':20} {'Edge':>6} {'Size':>8} {'Result':>8} {'Bankroll':>10}")
        print(f"  {'-'*75}")
        for b in r.bet_log[-20:]:
            team = b["blue"] if b["bet_side"] == "blue" else b["red"]
            result = f"${b['pnl']:+.2f}"
            print(f"  {b['date']:12} {b['league']:6} {team:20} {b['edge']:6.1%} ${b['bet_amount']:7,.2f} {result:>8} ${b['bankroll']:9,.2f}")
        print()


def run_edge_sweep(league_filter: Optional[str] = None, bankroll: float = 1000.0) -> None:
    """Run backtests at multiple edge thresholds to find optimal."""
    print(f"\n{'='*70}")
    print(f"  EDGE THRESHOLD SWEEP")
    print(f"{'='*70}")
    print(f"  {'Min Edge':>10} {'Bets':>7} {'Win%':>7} {'ROI':>8} {'Final $':>10} {'Max DD':>8} {'Avg Bet':>9}")
    print(f"  {'-'*65}")

    for edge in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
        r = run_pnl_backtest(min_edge=edge, starting_bankroll=bankroll, league_filter=league_filter)
        if r.total_bets == 0:
            print(f"  {edge:10.0%} {'0':>7}       -        -          -        -         -")
            continue
        print(
            f"  {edge:10.0%} {r.total_bets:7} {r.win_rate:7.1%} {r.roi:+8.1%} "
            f"${r.final_bankroll:9,.2f} {r.max_drawdown_pct:8.1%} ${r.avg_bet_size:8,.2f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="P&L-focused backtest")
    parser.add_argument("--edge", type=float, default=0.03, help="Minimum edge to bet (default: 0.03)")
    parser.add_argument("--kelly", type=float, default=0.0625, help="Max Kelly fraction (default: 0.0625)")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll (default: 1000)")
    parser.add_argument("--league", type=str, default=None, help="Filter to single league")
    parser.add_argument("--sweep", action="store_true", help="Run edge threshold sweep")
    parser.add_argument("--vig", type=float, default=0.02, help="Simulated market vig (default: 0.02)")
    args = parser.parse_args()

    if args.sweep:
        run_edge_sweep(league_filter=args.league, bankroll=args.bankroll)
        return

    r = run_pnl_backtest(
        min_edge=args.edge, max_kelly=args.kelly,
        starting_bankroll=args.bankroll, league_filter=args.league,
        market_vig=args.vig,
    )
    print_report(r, args.edge, args.kelly)


if __name__ == "__main__":
    main()
