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

from backtest.backtest import ELOTracker, load_matches
from model.calibration import PlattCalibrator
from model.predict import check_cross_region
from model.pro_elo import HALF_LIFE_DAYS
from scrapers.team_matcher import match_team_name

DB_PATH = _ROOT / "db" / "lol_model.db"
CSV_PATH = _ROOT / "data" / "backtest_trades.csv"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

T2_KEYWORDS = [
    "lck challengers", "tcl", "ljl", "nacl", "north american challengers",
    "lfl", "nlc", "emea masters",
    "pcs", "vcs", "superliga", "prime league", "hitpoint", "road of legends",
]

DEFAULT_THRESHOLD = 0.10
DEFAULT_KELLY = 0.0625
DEFAULT_BANKROLL_CAP = 0.02
DEFAULT_BANKROLL = 1000.0
MIN_TEAM_GAMES = 10

# Realistic execution model
# Opening liquidity is a small fraction of total volume — most volume
# trades near match time, not at market creation. These estimates are
# conservative to ensure live results don't undershoot the backtest.

def estimate_opening_cost(total_volume: float) -> float:
    """Volume-dependent spread + slippage. Thin markets cost more."""
    if total_volume < 2000:
        return 0.08  # 8% — very thin, wide spread, bad fills
    elif total_volume < 5000:
        return 0.06  # 6%
    elif total_volume < 15000:
        return 0.05  # 5%
    elif total_volume < 50000:
        return 0.04  # 4%
    else:
        return 0.03  # 3% — liquid enough for reasonable fills

def estimate_fillable_at_open(total_volume: float) -> float:
    """Estimate max fillable stake at open within ~2 cents of quoted price.
    Opening hour typically has 2-5% of total volume available as resting depth.
    Thin markets are worse — might have $10-50 on the book at open."""
    if total_volume < 2000:
        return min(total_volume * 0.01, 20.0)   # 1% of volume, max $20
    elif total_volume < 5000:
        return min(total_volume * 0.015, 50.0)   # 1.5%, max $50
    elif total_volume < 15000:
        return min(total_volume * 0.02, 150.0)   # 2%, max $150
    elif total_volume < 50000:
        return min(total_volume * 0.025, 300.0)  # 2.5%, max $300
    else:
        return min(total_volume * 0.03, 500.0)   # 3%, max $500


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
    kelly_max: float = DEFAULT_KELLY,
    bankroll_cap: float = DEFAULT_BANKROLL_CAP,
    starting_bankroll: float = DEFAULT_BANKROLL,
) -> Tuple[List[Dict], Dict]:
    """Run full backtest with walk-forward ELOs (no lookahead). Returns (trades, summary)."""
    session = _make_session()

    conn = sqlite3.connect(DB_PATH)
    db_teams = [r[0] for r in conn.execute("SELECT team_name FROM teams").fetchall()]
    team_leagues = {
        r[0]: r[1] or ""
        for r in conn.execute("SELECT team_name, league FROM teams").fetchall()
    }
    conn.close()

    events = fetch_resolved_markets(session)

    # Phase 1: collect raw market data from API (no model predictions yet)
    raw_markets = []

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
            match_date = datetime.fromtimestamp(times_arr[0]).strftime("%Y-%m-%d")

            raw_markets.append({
                "date": match_date,
                "title": title,
                "ql": ql,
                "db_a": db_a,
                "db_b": db_b,
                "team_a_raw": team_a,
                "winner": winner,
                "open_price": prices_arr[0],
                "pre_match_close": prices_arr[match_start_idx],
                "volume": float(m.get("volumeNum", 0) or m.get("volume", 0) or 0),
            })
            time.sleep(0.15)

    logger.info(f"  {len(raw_markets)} raw markets collected from API")

    # Phase 2: walk-forward prediction — advance ELO tracker through matches,
    # predict each market using ONLY prior match data (no lookahead)
    raw_markets.sort(key=lambda m: m["date"])
    all_matches = load_matches()
    calibrator = PlattCalibrator()
    calibrator.load()

    # V2 params: K=32, MOV=1.5, no soloq, identity calibration
    tracker = ELOTracker(
        K=32, blend_k=5, scale=400, half_life_days=HALF_LIFE_DAYS,
        soloq_baselines={}, regional_offsets={},
        mov_weight=1.5,
    )

    match_idx = 0
    raw_candidates = []

    for market in raw_markets:
        # Advance tracker through all matches BEFORE this market's date
        while match_idx < len(all_matches) and all_matches[match_idx][1] < market["date"]:
            row = all_matches[match_idx]
            gameid, date, league, blue, red, winner = row[:6]
            stats = dict(blue_kills=row[6], red_kills=row[7],
                         blue_deaths=row[8], red_deaths=row[9],
                         blue_gd15=row[10], red_gd15=row[11])
            tracker.update(blue, red, winner, league, date, **stats)
            match_idx += 1

        db_a, db_b = market["db_a"], market["db_b"]

        # Walk-forward game count check (not final DB count)
        if tracker.games.get(db_a, 0) < MIN_TEAM_GAMES or tracker.games.get(db_b, 0) < MIN_TEAM_GAMES:
            continue

        region_check = check_cross_region(db_a, db_b)
        if region_check["cross_region"]:
            continue

        # Point-in-time prediction using walk-forward ELOs
        league_a = team_leagues.get(db_a, "")
        p_a_raw = tracker.predict(db_a, db_b, league_a, market["date"])
        model_a = calibrator.calibrate(p_a_raw) if calibrator.fitted else p_a_raw

        actual = 1.0 if market["winner"] == market["team_a_raw"] else 0.0
        open_price = market["open_price"]
        pre_match_close = market["pre_match_close"]

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

        volume = market["volume"]
        model_p = model_a if bet_on_a else (1.0 - model_a)
        cost = estimate_opening_cost(volume)
        fillable = estimate_fillable_at_open(volume)
        raw_open = open_price if bet_on_a else 1.0 - open_price
        entry = min(raw_open + cost, 0.99)
        won = (bet_on_a and actual == 1.0) or (not bet_on_a and actual == 0.0)
        pmc = pre_match_close if bet_on_a else 1.0 - pre_match_close
        clv = round(pmc - raw_open, 4)

        raw_candidates.append({
            "date": market["date"],
            "league": next((kw for kw in T2_KEYWORDS if kw in market["title"]), ""),
            "format": "bo3" if "(bo3)" in market["ql"] else ("bo5" if "(bo5)" in market["ql"] else "bo1"),
            "bet_team": db_a if bet_on_a else db_b,
            "opponent": db_b if bet_on_a else db_a,
            "model_prob": round(model_p, 3),
            "market_open": round(raw_open, 3),
            "edge": round(edge, 3),
            "cost": round(cost, 3),
            "entry_price": round(entry, 3),
            "fillable_est": round(fillable, 2),
            "won": won,
            "clv": clv,
            "prematch_close": round(pmc, 3),
            "market_volume": round(volume, 0),
            "winner": market["winner"],
        })

    logger.info(
        f"  {len(raw_candidates)} trades pass gates (walk-forward, {match_idx}/{len(all_matches)} matches processed)"
    )

    # Phase 2: sort by date, then simulate with bankroll-dependent sizing
    raw_candidates.sort(key=lambda c: c["date"])

    trades = []
    bankroll = starting_bankroll
    peak = starting_bankroll
    max_dd = 0.0
    streak = 0
    max_streak = 0

    for c in raw_candidates:
        entry = c["entry_price"]
        model_p = c["model_prob"]
        fillable = c["fillable_est"]

        if 0 < entry < 1:
            b_odds = (1.0 / entry) - 1.0
            kelly = max(0.0, min((model_p * b_odds - (1.0 - model_p)) / b_odds, kelly_max))
        else:
            kelly = 0.0

        kelly_size = bankroll * kelly
        cap_size = bankroll * bankroll_cap
        size = round(min(kelly_size, cap_size, fillable), 2)
        if size < 1.0:
            continue

        won = c["won"]
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

        trades.append({
            "trade_num": len(trades) + 1,
            "date": c["date"],
            "league": c["league"],
            "format": c["format"],
            "bet_team": c["bet_team"],
            "opponent": c["opponent"],
            "model_prob": c["model_prob"],
            "market_open": c["market_open"],
            "edge": c["edge"],
            "cost": c["cost"],
            "entry_price": c["entry_price"],
            "fillable_est": c["fillable_est"],
            "stake": round(size, 2),
            "result": "WIN" if won else "LOSS",
            "pnl": pnl,
            "bankroll_after": bankroll,
            "clv": c["clv"],
            "prematch_close": c["prematch_close"],
            "beat_close": "Y" if c["clv"] > 0 else "N",
            "market_volume": c["market_volume"],
            "winner": c["winner"],
        })

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
    print(f"  POLYMARKET OPENING-LINE BACKTEST (walk-forward, no lookahead)")
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
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output")
    args = parser.parse_args()

    logger.info("Running Polymarket opening-line backtest (realistic liquidity model)…")
    trades, summary = run_backtest(
        threshold=args.threshold,
        starting_bankroll=args.bankroll,
    )

    if "error" in summary:
        logger.error(summary["error"])
        return

    print_report(summary, args.threshold)

    if not args.no_csv:
        bankroll_tag = f"_{int(args.bankroll)}" if args.bankroll != DEFAULT_BANKROLL else ""
        out_path = _ROOT / "data" / f"backtest_trades{bankroll_tag}.csv"
        path = write_csv(trades, out_path)
        print(f"  Trade log: {path}")
        print(f"  Import to Google Sheets: File → Import → Upload")
    print()


if __name__ == "__main__":
    main()
