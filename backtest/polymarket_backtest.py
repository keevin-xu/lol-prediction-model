"""
Backtest the opening-line strategy against real resolved Polymarket markets.

Pulls actual price trajectories from the CLOB API, applies the frozen
rule (same-region, >10% edge, bet at open), and reports P&L with a
full trade log CSV.

Run:
  python backtest/polymarket_backtest.py                # full backtest + CSV
  python backtest/polymarket_backtest.py --threshold 0.08  # custom edge threshold
  python backtest/polymarket_backtest.py --bankroll 5000   # custom starting bankroll
"""

import argparse
import csv
import json
import sqlite3
import sys
import time
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

from model.predict import predict_match
from scrapers.team_matcher import match_team_name

DB_PATH = _ROOT / "db" / "lol_model.db"
CSV_PATH = _ROOT / "data" / "backtest_trades.csv"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

T2_KEYWORDS = [
    "lck challengers", "tcl", "ljl", "nacl", "lfl", "nlc", "emea masters",
    "pcs", "vcs", "superliga", "prime league", "hitpoint", "road of legends",
]

DEFAULT_THRESHOLD = 0.10
DEFAULT_COST = 0.03
DEFAULT_KELLY = 0.0625
DEFAULT_BANKROLL_CAP = 0.02
DEFAULT_BANKROLL = 1000.0
MIN_TEAM_GAMES = 10


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_resolved_markets(session: requests.Session) -> List[Dict]:
    """Fetch all resolved T2 LoL moneyline markets from Polymarket."""
    all_events = []
    offset = 0
    for _ in range(20):
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

    logger.info(f"Fetched {len(all_events)} resolved LoL events from Polymarket")
    return all_events


def detect_match_start(prices: List[float]) -> int:
    """Detect where in-game price movement begins."""
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


def run_backtest(
    threshold: float = DEFAULT_THRESHOLD,
    cost: float = DEFAULT_COST,
    kelly_max: float = DEFAULT_KELLY,
    bankroll_cap: float = DEFAULT_BANKROLL_CAP,
    starting_bankroll: float = DEFAULT_BANKROLL,
) -> Tuple[List[Dict], Dict]:
    """Run full backtest. Returns (trades, summary)."""
    session = _make_session()

    conn = sqlite3.connect(DB_PATH)
    db_teams = [r[0] for r in conn.execute("SELECT team_name FROM teams").fetchall()]
    team_games = {
        r[0]: r[1]
        for r in conn.execute("SELECT team_name, games_played FROM teams").fetchall()
    }
    conn.close()

    events = fetch_resolved_markets(session)

    trades = []
    bankroll = starting_bankroll
    peak = starting_bankroll
    max_dd = 0.0
    streak = 0
    max_streak = 0
    markets_checked = 0

    for event in events:
        title = event.get("title", "").lower()
        if not any(kw in title for kw in T2_KEYWORDS):
            continue

        for m in event.get("markets", []):
            q = m.get("question", "")
            ql = q.lower()
            if "(bo" not in ql or " vs " not in ql:
                continue
            if any(x in ql for x in ["game 1", "game 2", "game 3", "game 4", "game 5", "handicap"]):
                continue

            outcomes = m.get("outcomes", "[]")
            prices = m.get("outcomePrices", "[]")
            tokens = m.get("clobTokenIds", "[]")
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            if isinstance(prices, str):
                prices = json.loads(prices)
            if isinstance(tokens, str):
                tokens = json.loads(tokens)
            if len(outcomes) < 2 or len(tokens) < 2:
                continue

            try:
                pa, pb = float(prices[0]), float(prices[1])
            except (ValueError, TypeError):
                continue
            if not (pa >= 0.99 or pb >= 0.99):
                continue

            team_a = outcomes[0].strip()
            team_b = outcomes[1].strip()
            winner = team_a if pa >= 0.99 else team_b

            db_a = match_team_name(team_a, db_teams, source="polymarket")
            db_b = match_team_name(team_b, db_teams, source="polymarket")
            if not db_a or not db_b:
                continue
            if team_games.get(db_a, 0) < MIN_TEAM_GAMES or team_games.get(db_b, 0) < MIN_TEAM_GAMES:
                continue

            # Pull price trajectory
            try:
                r2 = session.get(
                    f"{CLOB_API}/prices-history",
                    params={"market": tokens[0], "startTs": "1", "fidelity": "10"},
                    timeout=10,
                )
                if r2.status_code != 200:
                    continue
                hist = r2.json().get("history", [])
                if len(hist) < 10:
                    continue
            except requests.RequestException:
                continue

            prices_arr = [h["p"] for h in hist]
            times_arr = [h["t"] for h in hist]
            match_start_idx = detect_match_start(prices_arr)

            pred = predict_match(db_a, db_b)
            model_a = pred["p_a"]
            actual = 1.0 if winner == team_a else 0.0

            if pred.get("cross_region", False):
                continue

            open_price = prices_arr[0]
            pre_match_close = prices_arr[match_start_idx]

            edge_a = model_a - open_price
            edge_b = (1.0 - model_a) - (1.0 - open_price)
            if abs(edge_a) >= abs(edge_b):
                edge = abs(edge_a)
                bet_on_a = edge_a > 0
            else:
                edge = abs(edge_b)
                bet_on_a = False

            if edge < threshold:
                continue

            entry = (open_price if bet_on_a else 1.0 - open_price) + cost
            entry = min(entry, 0.99)
            model_p = model_a if bet_on_a else (1.0 - model_a)

            # Kelly sizing
            if 0 < entry < 1:
                b_odds = (1.0 / entry) - 1.0
                kelly = max(0.0, min((model_p * b_odds - (1.0 - model_p)) / b_odds, kelly_max))
            else:
                kelly = 0.0

            volume = float(m.get("volumeNum", 0) or m.get("volume", 0) or 0)
            size = min(bankroll * kelly, bankroll * bankroll_cap, volume * 0.03, 700.0)
            if size < 1.0:
                continue

            won = (bet_on_a and actual == 1.0) or (not bet_on_a and actual == 0.0)
            pnl = round(size * (1.0 / entry - 1.0) if won else -size, 2)
            bankroll = round(bankroll + pnl, 2)

            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

            if not won:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

            pmc = pre_match_close if bet_on_a else 1.0 - pre_match_close
            op = open_price if bet_on_a else 1.0 - open_price
            clv = round(pmc - op, 4)

            match_date = datetime.fromtimestamp(times_arr[0]).strftime("%Y-%m-%d")
            bet_team = db_a if bet_on_a else db_b
            opp_team = db_b if bet_on_a else db_a
            league = next((kw for kw in T2_KEYWORDS if kw in title), "")
            bo_type = "bo3" if "(bo3)" in ql else ("bo5" if "(bo5)" in ql else "bo1")

            trades.append({
                "trade_num": len(trades) + 1,
                "date": match_date,
                "league": league,
                "format": bo_type,
                "bet_team": bet_team,
                "opponent": opp_team,
                "model_prob": round(model_p, 3),
                "market_open": round(op, 3),
                "edge": round(edge, 3),
                "entry_price": round(entry, 3),
                "stake": round(size, 2),
                "result": "WIN" if won else "LOSS",
                "pnl": pnl,
                "bankroll_after": bankroll,
                "clv": clv,
                "prematch_close": round(pmc, 3),
                "beat_close": "Y" if clv > 0 else "N",
                "market_volume": round(volume, 0),
                "winner": winner,
            })

            markets_checked += 1
            if markets_checked % 50 == 0:
                logger.info(f"  {markets_checked} bets placed…")
            time.sleep(0.15)

    # Compute summary
    if not trades:
        return trades, {"error": "No trades triggered"}

    wins = sum(1 for t in trades if t["result"] == "WIN")
    n = len(trades)
    pnls = [t["pnl"] / max(t["stake"], 0.01) for t in trades]
    clvs = [t["clv"] for t in trades]
    total_pnl = sum(t["pnl"] for t in trades)
    total_wagered = sum(t["stake"] for t in trades)

    rng = np.random.RandomState(42)
    boot_rois = [np.mean(rng.choice(pnls, n, replace=True)) for _ in range(10000)]
    boot_hits = [
        np.mean(rng.choice([1 if t["result"] == "WIN" else 0 for t in trades], n, replace=True))
        for _ in range(10000)
    ]

    summary = {
        "trades": n,
        "wins": wins,
        "losses": n - wins,
        "hit_rate": wins / n,
        "hit_ci": (np.percentile(boot_hits, 2.5), np.percentile(boot_hits, 97.5)),
        "total_wagered": total_wagered,
        "total_pnl": total_pnl,
        "final_bankroll": bankroll,
        "roi_capital": total_pnl / starting_bankroll,
        "roi_per_bet": np.mean(pnls),
        "roi_ci": (np.percentile(boot_rois, 2.5), np.percentile(boot_rois, 97.5)),
        "max_drawdown": max_dd,
        "max_loss_streak": max_streak,
        "mean_clv": np.mean(clvs),
        "pct_beat_close": sum(1 for c in clvs if c > 0) / n,
        "ci_clears_zero": np.percentile(boot_rois, 2.5) > 0,
    }

    return trades, summary


def write_csv(trades: List[Dict], path: Optional[Path] = None) -> str:
    path = path or CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trades[0].keys()))
        writer.writeheader()
        for t in trades:
            writer.writerow(t)
    return str(path)


def print_report(summary: Dict, threshold: float) -> None:
    s = summary
    print(f"\n{'='*60}")
    print(f"  POLYMARKET OPENING-LINE BACKTEST")
    print(f"{'='*60}")
    print(f"  Rule:      Same-region T2, >{threshold:.0%} edge, bet at open")
    print(f"  Costs:     3% (spread + slippage)")
    print(f"  Sizing:    Quarter-Kelly, 2% cap, depth-gated")
    print()
    print(f"  Trades:    {s['trades']}")
    print(f"  Record:    {s['wins']}W / {s['losses']}L")
    print(f"  Hit rate:  {s['hit_rate']:.1%}  (CI: {s['hit_ci'][0]:.1%} – {s['hit_ci'][1]:.1%})")
    print(f"  Wagered:   ${s['total_wagered']:,.2f}")
    print(f"  P&L:       ${s['total_pnl']:+,.2f}")
    print(f"  Final:     ${s['final_bankroll']:,.2f}")
    print(f"  ROI:       {s['roi_capital']:+.0%} on capital")
    print(f"  ROI/bet:   {s['roi_per_bet']:+.1%}  (CI: {s['roi_ci'][0]:+.1%} – {s['roi_ci'][1]:+.1%})")
    print(f"  Max DD:    {s['max_drawdown']:.1%}")
    print(f"  Max streak:{s['max_loss_streak']} losses")
    print(f"  CLV:       {s['mean_clv']:+.3f}")
    print(f"  Beat close:{s['pct_beat_close']:.0%}")
    print(f"  CI > 0:    {'YES' if s['ci_clears_zero'] else 'NO'}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest against real Polymarket markets")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Min edge threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL,
                        help=f"Starting bankroll (default: {DEFAULT_BANKROLL})")
    parser.add_argument("--cost", type=float, default=DEFAULT_COST,
                        help=f"Total cost per trade (default: {DEFAULT_COST})")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output")
    args = parser.parse_args()

    logger.info("Running Polymarket opening-line backtest…")
    trades, summary = run_backtest(
        threshold=args.threshold,
        starting_bankroll=args.bankroll,
        cost=args.cost,
    )

    if "error" in summary:
        logger.error(summary["error"])
        return

    print_report(summary, args.threshold)

    if not args.no_csv:
        path = write_csv(trades)
        print(f"  Trade log: {path}")
        print(f"  Import to Google Sheets: File → Import → Upload")
    print()


if __name__ == "__main__":
    main()
