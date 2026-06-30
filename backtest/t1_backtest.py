"""
T1 LoL opening-line backtest — tests whether a model with X% accuracy
can profitably bet against Polymarket T1 lines at different time points
before match start.

Unlike the T2 backtest (which uses our walk-forward ELO model), this is a
Monte Carlo simulation: we don't have a T1 model yet, so we simulate one
at varying accuracy levels to determine the minimum viable accuracy.

Run:
  python backtest/t1_backtest.py                          # full grid
  python backtest/t1_backtest.py --accuracy 0.70          # single accuracy
  python backtest/t1_backtest.py --time-point 2h          # single time point
  python backtest/t1_backtest.py --bankroll 5000          # custom bankroll
  python backtest/t1_backtest.py --league LPL             # single league
  python backtest/t1_backtest.py --trials 1000            # more MC trials
  python backtest/t1_backtest.py --csv                    # write trade logs
"""

import argparse
import csv
import json
import sys
import time as time_mod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

T1_KEYWORDS = ["lec", "lck:", "lck cup", "lpl", "lcs:", "lcs 2026"]
T1_EXCLUDE = ["challengers", "academy"]

TIME_POINTS = ["open", "24h", "12h", "8h", "6h", "4h", "3h", "2h", "1h"]
DEFAULT_ACCURACIES = [0.60, 0.65, 0.70, 0.75]
DEFAULT_BANKROLLS = [1000, 5000, 10000]
DEFAULT_TRIALS = 500
DEFAULT_THRESHOLD = 0.10
DEFAULT_KELLY = 0.0625
DEFAULT_CAP = 0.02


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def estimate_t1_cost(volume: float) -> float:
    if volume < 50000:
        return 0.05
    elif volume < 200000:
        return 0.04
    elif volume < 500000:
        return 0.03
    else:
        return 0.02


def estimate_t1_fillable(volume: float) -> float:
    if volume < 50000:
        return min(volume * 0.02, 500)
    elif volume < 200000:
        return min(volume * 0.03, 2000)
    elif volume < 500000:
        return min(volume * 0.04, 5000)
    else:
        return min(volume * 0.05, 10000)


def detect_match_start(prices: List[float]) -> int:
    n = len(prices)
    for i in range(n):
        if prices[i] >= 0.90 or prices[i] <= 0.10:
            return max(i - 1, 0)
    if n > 5:
        diffs = [abs(prices[i + 1] - prices[i]) for i in range(n - 1)]
        if max(diffs) > 0.05:
            big_move = next(i for i, d in enumerate(diffs) if d > 0.05)
            return max(big_move - 1, 0)
    return n - 1


def classify_league(title: str) -> str:
    tl = title.lower()
    if "lec" in tl:
        return "LEC"
    elif "lck" in tl:
        return "LCK"
    elif "lpl" in tl:
        return "LPL"
    elif "lcs" in tl:
        return "LCS"
    return "Other"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def fetch_t1_markets(
    session: requests.Session,
    league_filter: Optional[str] = None,
) -> List[Dict]:
    """Fetch resolved T1 LoL markets with price trajectories at each time point."""
    all_events = []
    offset = 0
    for _ in range(25):
        r = session.get(
            f"{GAMMA_API}/events",
            params={"tag_slug": "league-of-legends", "closed": "true",
                    "limit": "100", "offset": str(offset)},
            timeout=15,
        )
        if r.status_code != 200:
            break
        batch = r.json()
        all_events.extend(batch)
        if len(batch) < 100:
            break
        offset += 100

    logger.info(f"Fetched {len(all_events)} closed LoL events from Polymarket")

    markets = []
    checked = 0

    for e in all_events:
        title = e.get("title", "")
        tl = title.lower()
        if not any(k in tl for k in T1_KEYWORDS):
            continue
        if "lol:" not in tl or "vs" not in tl:
            continue
        if any(k in tl for k in T1_EXCLUDE):
            continue

        league = classify_league(title)
        if league_filter and league != league_filter.upper():
            continue

        for m in e.get("markets", []):
            q = m.get("question", "")
            ql = q.lower()
            if "(bo" not in ql or "vs" not in ql:
                continue
            if any(x in ql for x in ["game 1", "game 2", "game 3", "game 4", "game 5", "handicap"]):
                continue

            outcomes = m.get("outcomes", "[]")
            prices_raw = m.get("outcomePrices", "[]")
            tokens = m.get("clobTokenIds", "[]")
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            if isinstance(prices_raw, str):
                prices_raw = json.loads(prices_raw)
            if isinstance(tokens, str):
                tokens = json.loads(tokens)
            if len(outcomes) < 2 or len(tokens) < 2:
                continue

            try:
                pa, pb = float(prices_raw[0]), float(prices_raw[1])
            except (ValueError, TypeError):
                continue
            if not (pa >= 0.99 or pb >= 0.99):
                continue

            winner_idx = 0 if pa >= 0.99 else 1

            try:
                r2 = session.get(
                    f"{CLOB_API}/prices-history",
                    params={"market": tokens[0], "startTs": "1", "fidelity": "10"},
                    timeout=10,
                )
                if r2.status_code != 200:
                    continue
                hist = r2.json().get("history", [])
                if len(hist) < 20:
                    continue
            except requests.RequestException:
                continue

            prices_arr = [h["p"] for h in hist]
            times_arr = [h["t"] for h in hist]
            match_start_idx = detect_match_start(prices_arr)
            actual = 1.0 if winner_idx == 0 else 0.0
            volume = float(m.get("volumeNum", 0) or m.get("volume", 0) or 0)

            snapshots = {"open": prices_arr[0]}
            for hours_before in [24, 12, 8, 6, 4, 3, 2, 1]:
                idx = match_start_idx - (hours_before * 6)
                if 0 <= idx < len(prices_arr):
                    snapshots[f"{hours_before}h"] = prices_arr[idx]
            snapshots["pre_match"] = prices_arr[match_start_idx]

            match_date = datetime.fromtimestamp(times_arr[0]).strftime("%Y-%m-%d")

            markets.append({
                "date": match_date,
                "league": league,
                "title": title,
                "team_a": outcomes[0].strip(),
                "team_b": outcomes[1].strip(),
                "winner": outcomes[winner_idx].strip(),
                "actual": actual,
                "volume": volume,
                "snapshots": snapshots,
                "match_start_idx": match_start_idx,
                "total_points": len(prices_arr),
            })

            checked += 1
            if checked % 50 == 0:
                logger.info(f"  {checked} markets collected…")
            time_mod.sleep(0.1)

        if checked >= 500:
            break

    markets.sort(key=lambda m: m["date"])
    logger.info(f"  {len(markets)} T1 markets collected and sorted")
    return markets


# ---------------------------------------------------------------------------
# Line accuracy analysis
# ---------------------------------------------------------------------------
def print_line_accuracy(markets: List[Dict]) -> None:
    """Print line accuracy at each time point — the bar a model must clear."""
    print(f"\n{'='*80}")
    print(f"  T1 LINE ACCURACY BY TIME POINT ({len(markets)} markets)")
    print(f"{'='*80}")
    print(f"  {'Time Point':<18} {'Accuracy':>9} {'Brier':>8} {'Fav Price':>10} {'Markets':>8}")
    print(f"  {'-'*57}")

    labels = {
        "open": "Market open", "24h": "24h before", "12h": "12h before",
        "8h": "8h before", "6h": "6h before", "4h": "4h before",
        "3h": "3h before", "2h": "2h before", "1h": "1h before",
        "pre_match": "Pre-match close",
    }
    for tp in TIME_POINTS + ["pre_match"]:
        subset = [(m["snapshots"][tp], m["actual"]) for m in markets if tp in m["snapshots"]]
        if not subset:
            continue
        n = len(subset)
        correct = sum(1 for p, a in subset if (p >= 0.5) == (a == 1.0))
        brier = np.mean([(p - a) ** 2 for p, a in subset])
        avg_fav = np.mean([max(p, 1 - p) for p, _ in subset])
        print(f"  {labels.get(tp, tp):<18} {correct / n:>9.1%} {brier:>8.4f} {avg_fav:>10.3f} {n:>8}")

    print()
    by_league = {}
    for m in markets:
        by_league.setdefault(m["league"], []).append(m)

    for tp in ["4h", "2h", "1h"]:
        print(f"  {labels[tp]} by league:")
        for league in ["LEC", "LCK", "LPL", "LCS"]:
            lg = by_league.get(league, [])
            subset = [(m["snapshots"][tp], m["actual"]) for m in lg if tp in m["snapshots"]]
            if len(subset) < 5:
                continue
            n = len(subset)
            correct = sum(1 for p, a in subset if (p >= 0.5) == (a == 1.0))
            brier = np.mean([(p - a) ** 2 for p, a in subset])
            print(f"    {league:<6} {correct / n:.1%} acc  Brier={brier:.4f}  n={n}")
        print()


# ---------------------------------------------------------------------------
# Monte Carlo P&L simulation
# ---------------------------------------------------------------------------
def run_simulation(
    markets: List[Dict],
    time_point: str,
    model_accuracy: float,
    starting_bankroll: float,
    kelly_max: float = DEFAULT_KELLY,
    cap_pct: float = DEFAULT_CAP,
    threshold: float = DEFAULT_THRESHOLD,
    n_trials: int = DEFAULT_TRIALS,
    rng_seed: int = 42,
) -> Optional[Dict]:
    eligible = []
    for mkt in markets:
        if time_point not in mkt["snapshots"]:
            continue
        eligible.append({
            "date": mkt["date"],
            "league": mkt["league"],
            "team_a": mkt["team_a"],
            "team_b": mkt["team_b"],
            "winner": mkt["winner"],
            "line_price": mkt["snapshots"][time_point],
            "pre_match": mkt["snapshots"].get("pre_match", mkt["snapshots"][time_point]),
            "actual": mkt["actual"],
            "cost": estimate_t1_cost(mkt["volume"]),
            "fillable": estimate_t1_fillable(mkt["volume"]),
            "volume": mkt["volume"],
        })

    if not eligible:
        return None

    rng = np.random.RandomState(rng_seed)
    all_finals = []
    all_max_dds = []
    all_trade_counts = []
    all_win_counts = []
    all_pnls = []
    best_trades = None
    median_final = None

    for trial in range(n_trials):
        bankroll = float(starting_bankroll)
        peak = bankroll
        max_dd = 0.0
        trades = []

        for mkt in eligible:
            line = mkt["line_price"]
            actual = mkt["actual"]

            correct = rng.random() < model_accuracy
            if correct:
                model_says_a_wins = (actual == 1.0)
            else:
                model_says_a_wins = (actual != 1.0)

            model_prob_a = 0.65 if model_says_a_wins else 0.35

            edge_a = model_prob_a - line
            edge_b = (1.0 - model_prob_a) - (1.0 - line)
            if abs(edge_a) >= abs(edge_b):
                edge = abs(edge_a)
                bet_on_a = edge_a > 0
            else:
                edge = abs(edge_b)
                bet_on_a = edge_b < 0

            if edge < threshold:
                continue

            model_p = model_prob_a if bet_on_a else (1.0 - model_prob_a)
            raw_entry = line if bet_on_a else 1.0 - line
            entry = min(raw_entry + mkt["cost"], 0.99)

            if not (0 < entry < 1):
                continue

            b_odds = (1.0 / entry) - 1.0
            kelly = max(0.0, min((model_p * b_odds - (1.0 - model_p)) / b_odds, kelly_max))
            kelly_size = bankroll * kelly
            cap_size = bankroll * cap_pct
            size = round(min(kelly_size, cap_size, mkt["fillable"]), 2)
            if size < 1.0:
                continue

            won = (bet_on_a and actual == 1.0) or (not bet_on_a and actual == 0.0)
            pnl = round(size * (1.0 / entry - 1.0) if won else -size, 2)
            bankroll = round(bankroll + pnl, 2)

            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

            pmc = mkt["pre_match"] if bet_on_a else 1.0 - mkt["pre_match"]
            op = line if bet_on_a else 1.0 - line
            clv = round(pmc - op, 4)

            trades.append({
                "date": mkt["date"],
                "league": mkt["league"],
                "bet_team": mkt["team_a"] if bet_on_a else mkt["team_b"],
                "opponent": mkt["team_b"] if bet_on_a else mkt["team_a"],
                "model_prob": round(model_p, 3),
                "market_line": round(raw_entry, 3),
                "edge": round(edge, 3),
                "cost": round(mkt["cost"], 3),
                "entry_price": round(entry, 3),
                "fillable_est": round(mkt["fillable"], 2),
                "stake": size,
                "result": "WIN" if won else "LOSS",
                "pnl": pnl,
                "bankroll_after": bankroll,
                "clv": clv,
                "volume": round(mkt["volume"], 0),
                "winner": mkt["winner"],
            })

            if bankroll < 10:
                break

        all_finals.append(bankroll)
        all_max_dds.append(max_dd)
        all_trade_counts.append(len(trades))
        all_win_counts.append(sum(1 for t in trades if t["result"] == "WIN"))
        all_pnls.append(bankroll - starting_bankroll)

        if best_trades is None or abs(bankroll - np.median(all_finals[:max(trial, 1)])) < abs(
            (best_trades[-1]["bankroll_after"] if best_trades else starting_bankroll) - np.median(all_finals[:max(trial, 1)])
        ):
            best_trades = trades

    median_idx = int(np.argsort(all_finals)[len(all_finals) // 2])

    return {
        "n_eligible": len(eligible),
        "avg_trades": np.mean(all_trade_counts),
        "avg_wins": np.mean(all_win_counts),
        "avg_final": np.mean(all_finals),
        "median_final": np.median(all_finals),
        "p10_final": np.percentile(all_finals, 10),
        "p25_final": np.percentile(all_finals, 25),
        "p75_final": np.percentile(all_finals, 75),
        "p90_final": np.percentile(all_finals, 90),
        "avg_roi": (np.mean(all_finals) - starting_bankroll) / starting_bankroll * 100,
        "median_roi": (np.median(all_finals) - starting_bankroll) / starting_bankroll * 100,
        "avg_max_dd": np.mean(all_max_dds) * 100,
        "p90_max_dd": np.percentile(all_max_dds, 90) * 100,
        "bust_pct": sum(1 for f in all_finals if f < 50) / n_trials * 100,
        "avg_hit_rate": np.mean(all_win_counts) / np.mean(all_trade_counts) if np.mean(all_trade_counts) > 0 else 0,
        "median_trades": best_trades,
    }


def write_trade_csv(trades: List[Dict], path: Path) -> str:
    if not trades:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trades[0].keys()))
        writer.writeheader()
        for t in trades:
            writer.writerow(t)
    return str(path)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_grid(
    markets: List[Dict],
    time_points: List[str],
    accuracies: List[float],
    bankrolls: List[float],
    n_trials: int,
    write_csv_flag: bool = False,
) -> None:
    for starting in bankrolls:
        print(f"\n{'='*110}")
        print(f"  STARTING BANKROLL: ${starting:,}  |  Half-Kelly ({DEFAULT_KELLY:.1%}), {DEFAULT_CAP:.0%} cap, >{DEFAULT_THRESHOLD:.0%} edge  |  {n_trials} MC trials")
        print(f"{'='*110}")
        print(f"  {'Time':>6} {'Acc':>5} {'Mkts':>5} {'Trades':>6} {'Win%':>5} {'Avg Final':>11} {'Med Final':>11} {'ROI':>7}"
              f" {'P10':>10} {'P90':>10} {'Avg DD':>7} {'P90 DD':>7} {'Bust':>5}")
        print(f"  {'-'*108}")

        for tp in time_points:
            for acc in accuracies:
                r = run_simulation(markets, tp, acc, starting, n_trials=n_trials)
                if not r:
                    continue
                print(
                    f"  {tp:>6} {acc:>5.0%} {r['n_eligible']:>5} {r['avg_trades']:>6.0f} {r['avg_hit_rate']:>5.0%}"
                    f" ${r['avg_final']:>10,.0f} ${r['median_final']:>10,.0f} {r['avg_roi']:>+6.0f}%"
                    f" ${r['p10_final']:>9,.0f} ${r['p90_final']:>9,.0f}"
                    f" {r['avg_max_dd']:>6.1f}% {r['p90_max_dd']:>6.1f}% {r['bust_pct']:>4.1f}%"
                )

                if write_csv_flag and r["median_trades"]:
                    csv_name = f"t1_trades_{tp}_{int(acc * 100)}pct_{int(starting)}.csv"
                    csv_path = _ROOT / "data" / csv_name
                    write_trade_csv(r["median_trades"], csv_path)

            if tp != time_points[-1]:
                print()

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T1 LoL Polymarket backtest (Monte Carlo)")
    parser.add_argument("--accuracy", type=float, default=None,
                        help="Single model accuracy to test (e.g. 0.70)")
    parser.add_argument("--time-point", type=str, default=None,
                        help="Single time point (e.g. 2h, 4h, open)")
    parser.add_argument("--bankroll", type=float, default=None,
                        help="Single bankroll level")
    parser.add_argument("--league", type=str, default=None,
                        help="Filter to single league (LEC, LCK, LPL, LCS)")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"Monte Carlo trials (default: {DEFAULT_TRIALS})")
    parser.add_argument("--csv", action="store_true",
                        help="Write trade logs for median trial of each config")
    parser.add_argument("--no-accuracy", action="store_true",
                        help="Skip grid, just show line accuracy analysis")
    args = parser.parse_args()

    session = _make_session()

    logger.info("Fetching T1 markets from Polymarket…")
    markets = fetch_t1_markets(session, league_filter=args.league)

    if not markets:
        logger.error("No T1 markets found")
        return

    print_line_accuracy(markets)

    if args.no_accuracy:
        return

    time_points = [args.time_point] if args.time_point else ["open", "4h", "2h", "1h"]
    accuracies = [args.accuracy] if args.accuracy else DEFAULT_ACCURACIES
    bankrolls = [args.bankroll] if args.bankroll else DEFAULT_BANKROLLS

    logger.info(
        f"Running grid: {len(time_points)} time points × {len(accuracies)} accuracies "
        f"× {len(bankrolls)} bankrolls × {args.trials} trials"
    )
    print_grid(markets, time_points, accuracies, bankrolls, args.trials, args.csv)

    print("  Key:")
    print("  - Acc = simulated model accuracy (fraction of correct predictions)")
    print("  - Win% = realized win rate on placed bets (higher than Acc due to edge filter)")
    print("  - P10/P90 = 10th/90th percentile final bankroll across MC trials")
    print("  - P90 DD = 90th percentile max drawdown (worst-case-ish)")
    print("  - Bust = % of trials ending below $50")
    print()
    if args.league:
        print(f"  Filtered to: {args.league.upper()}")
    print(f"  T1 costs: 2-5% (volume-dependent, lower than T2)")
    print(f"  T1 fillable: $500-$10,000 per market (50x T2 depth)")
    print()


if __name__ == "__main__":
    main()
