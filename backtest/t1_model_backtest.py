"""
T1 model-based backtest using the REAL V2 walk-forward ELO model against the
frozen Polymarket T1 dataset (data/polymarket_t1_resolved.json).

Unlike backtest/t1_backtest.py (a Monte Carlo simulation over an assumed
accuracy parameter), this script:
  1. Runs the actual ELOTracker (same V2 params as T2: K=32, blend_k=5,
     scale=400, half_life=270d, mov_weight=1.5, no soloq) walk-forward
     through the full match DB.
  2. Matches Polymarket T1 team names to DB team names.
  3. Re-samples each market's real price history at fine time resolution
     (30m, 1h, 2h, 3h, 6h, 9h, 12h, 18h, 24h after market creation, plus
     hours-before-match where available).
  4. At each entry point, computes real model edge vs the real quoted price,
     real CLV, real hit rate, and runs real bankroll sweeps with quarter-Kelly
     sizing, 2% cap, and depth gating.

Run:
  python backtest/t1_model_backtest.py                  # full report
  python backtest/t1_model_backtest.py --csv             # write trade logs
"""

import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from backtest.backtest import ELOTracker, load_matches
from model.pro_elo import HALF_LIFE_DAYS, LEAGUE_TO_REGION
from scrapers.team_matcher import match_team_name

DB_PATH = _ROOT / "db" / "lol_model.db"
T1_DATA_PATH = _ROOT / "data" / "polymarket_t1_resolved.json"

DEFAULT_THRESHOLD = 0.10
DEFAULT_KELLY = 0.0625
DEFAULT_CAP = 0.02
MIN_TEAM_GAMES = 10

# Fine-grained entry points: minutes after market creation
ENTRY_POINTS_MIN = [0, 30, 60, 120, 180, 360, 540, 720, 1080, 1440]
ENTRY_LABELS = {
    0: "open (0m)", 30: "+30m", 60: "+1h", 120: "+2h", 180: "+3h",
    360: "+6h", 540: "+9h", 720: "+12h", 1080: "+18h", 1440: "+24h",
}

# Hours-before-match grid (mirrors t1_backtest.py for comparability)
HOURS_BEFORE = [24, 12, 8, 6, 4, 3, 2, 1]

BANKROLLS = [1000, 5000, 10000, 25000, 50000]
COST_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]


def classify_league(title: str) -> str:
    tl = title.lower()
    if "lck" in tl:
        return "LCK"
    elif "lpl" in tl:
        return "LPL"
    elif "lcs" in tl:
        return "LCS"
    elif "lec" in tl:
        return "LEC"
    return "Other"


def estimate_t1_cost(volume: float, multiplier: float = 1.0) -> float:
    if volume < 50000:
        base = 0.05
    elif volume < 200000:
        base = 0.04
    elif volume < 500000:
        base = 0.03
    else:
        base = 0.02
    return min(base * multiplier, 0.95)


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


def price_at_offset(times: List[int], prices: List[float], t0: int, offset_min: int) -> Optional[float]:
    """Find price at the closest available sample at or after t0 + offset_min minutes."""
    target = t0 + offset_min * 60
    if target > times[-1]:
        return None
    # find first index with time >= target
    for i, t in enumerate(times):
        if t >= target:
            return prices[i]
    return None


def price_at_hours_before(times: List[int], prices: List[float], match_start_idx: int, hours_before: int) -> Optional[float]:
    """Walk backward in real time (not index count) from match start."""
    target = times[match_start_idx] - hours_before * 3600
    if target < times[0]:
        return None
    for i in range(match_start_idx, -1, -1):
        if times[i] <= target:
            return prices[i]
    return None


# ---------------------------------------------------------------------------
# Phase 1: load + match markets to DB teams, compute walk-forward predictions
# ---------------------------------------------------------------------------
def load_and_match_markets() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    db_teams = [r[0] for r in conn.execute("SELECT team_name FROM teams").fetchall()]
    team_leagues = {
        r[0]: r[1] or "" for r in conn.execute("SELECT team_name, league FROM teams").fetchall()
    }
    conn.close()

    raw = json.loads(T1_DATA_PATH.read_text())
    logger.info(f"Loaded {len(raw)} raw T1 markets from {T1_DATA_PATH.name}")

    matched = []
    unmatched_a, unmatched_b = 0, 0

    for m in raw:
        league = classify_league(m["event_title"])
        if league == "Other":
            continue

        db_a = match_team_name(m["team_a"], db_teams, source="polymarket")
        db_b = match_team_name(m["team_b"], db_teams, source="polymarket")
        if not db_a:
            unmatched_a += 1
        if not db_b:
            unmatched_b += 1
        if not db_a or not db_b:
            continue

        hist = m["price_history"]
        if len(hist) < 10:
            continue
        times = [h["t"] for h in hist]
        prices = [h["p"] for h in hist]
        match_start_idx = detect_match_start(prices)

        try:
            created_dt = datetime.fromisoformat(m["created_at"].replace("Z", "+00:00"))
            market_date = created_dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            market_date = datetime.fromtimestamp(times[0], tz=timezone.utc).strftime("%Y-%m-%d")

        actual = 1.0 if m["winner"] == m["team_a"] else 0.0

        matched.append({
            "league": league,
            "db_a": db_a,
            "db_b": db_b,
            "team_a_raw": m["team_a"],
            "team_b_raw": m["team_b"],
            "winner": m["winner"],
            "actual": actual,
            "volume": m["volume"],
            "market_date": market_date,
            "times": times,
            "prices": prices,
            "match_start_idx": match_start_idx,
            "t0": times[0],
        })

    logger.info(
        f"Matched {len(matched)}/{len(raw)} markets to DB teams "
        f"(unmatched_a={unmatched_a}, unmatched_b={unmatched_b})"
    )
    return matched


def build_walkforward_predictions(matched: List[Dict]) -> List[Dict]:
    """Advance the V2 ELO tracker chronologically and predict each market
    using ONLY matches before its date — no lookahead. Returns markets
    enriched with model_prob_a (walk-forward) and team game counts."""
    matched.sort(key=lambda m: m["market_date"])
    all_matches = load_matches()

    conn = sqlite3.connect(DB_PATH)
    team_leagues = {
        r[0]: r[1] or "" for r in conn.execute("SELECT team_name, league FROM teams").fetchall()
    }
    conn.close()

    # V2 frozen params, identical to T2 production model
    tracker = ELOTracker(
        K=32, blend_k=5, scale=400, half_life_days=HALF_LIFE_DAYS,
        soloq_baselines={}, regional_offsets={}, mov_weight=1.5,
    )

    match_idx = 0
    enriched = []
    dropped_min_games = 0
    dropped_cross_region = 0

    for mkt in matched:
        while match_idx < len(all_matches) and all_matches[match_idx][1] < mkt["market_date"]:
            row = all_matches[match_idx]
            gameid, date, league, blue, red, winner = row[:6]
            stats = dict(blue_kills=row[6], red_kills=row[7],
                         blue_deaths=row[8], red_deaths=row[9],
                         blue_gd15=row[10], red_gd15=row[11])
            tracker.update(blue, red, winner, league, date, **stats)
            match_idx += 1

        db_a, db_b = mkt["db_a"], mkt["db_b"]

        if tracker.games.get(db_a, 0) < MIN_TEAM_GAMES or tracker.games.get(db_b, 0) < MIN_TEAM_GAMES:
            dropped_min_games += 1
            continue

        region_a = LEAGUE_TO_REGION.get(team_leagues.get(db_a, ""), "")
        region_b = LEAGUE_TO_REGION.get(team_leagues.get(db_b, ""), "")
        if region_a and region_b and region_a != region_b:
            dropped_cross_region += 1
            continue

        league_a = team_leagues.get(db_a, mkt["league"])
        p_a = tracker.predict(db_a, db_b, league_a, mkt["market_date"])

        mkt["model_prob_a"] = p_a
        mkt["games_a"] = tracker.games.get(db_a, 0)
        mkt["games_b"] = tracker.games.get(db_b, 0)
        enriched.append(mkt)

    logger.info(
        f"Walk-forward predictions: {len(enriched)} markets kept "
        f"({dropped_min_games} dropped <{MIN_TEAM_GAMES} games, {dropped_cross_region} cross-region), "
        f"{match_idx}/{len(all_matches)} historical matches processed"
    )
    return enriched


# ---------------------------------------------------------------------------
# Phase 2: per-league walk-forward accuracy (model only, no market data)
# ---------------------------------------------------------------------------
def per_league_accuracy(eval_start: str = "2024-01-01") -> Dict[str, Dict]:
    from backtest.backtest import run_backtest as elo_run_backtest

    results = {}
    for league in ["LCK", "LPL", "LCS", "LEC"]:
        r = elo_run_backtest(
            K=32, blend_k=5, scale=400, half_life_days=HALF_LIFE_DAYS,
            mov_weight=1.5, use_soloq=False, league_filter=league,
            eval_start=eval_start,
        )
        results[league] = {
            "accuracy": r.accuracy, "brier": r.brier_score,
            "log_loss": r.log_loss, "n": r.test_matches,
        }
    r_all = elo_run_backtest(
        K=32, blend_k=5, scale=400, half_life_days=HALF_LIFE_DAYS,
        mov_weight=1.5, use_soloq=False, eval_start=eval_start,
    )
    results["ALL_T2T1_COMBINED"] = {
        "accuracy": r_all.accuracy, "brier": r_all.brier_score,
        "log_loss": r_all.log_loss, "n": r_all.test_matches,
    }
    return results


# ---------------------------------------------------------------------------
# Phase 3: fine-grained entry-timing analysis (real model, real prices)
# ---------------------------------------------------------------------------
def build_trade_candidates_at_entry(
    enriched: List[Dict], entry_offset_min: Optional[int] = None,
    hours_before: Optional[int] = None, threshold: float = DEFAULT_THRESHOLD,
) -> List[Dict]:
    """For each market, sample the real quoted price at the given entry point
    (either minutes-after-open or hours-before-match), compute real model
    edge, and return trade candidates that clear the edge threshold."""
    candidates = []
    n_priced = 0

    for mkt in enriched:
        times, prices = mkt["times"], mkt["prices"]
        t0 = mkt["t0"]
        msi = mkt["match_start_idx"]

        if entry_offset_min is not None:
            price_a = price_at_offset(times, prices, t0, entry_offset_min)
        else:
            price_a = price_at_hours_before(times, prices, msi, hours_before)

        if price_a is None:
            continue
        n_priced += 1

        model_a = mkt["model_prob_a"]
        edge_a = model_a - price_a
        edge_b = (1.0 - model_a) - (1.0 - price_a)

        if abs(edge_a) >= abs(edge_b):
            edge = abs(edge_a)
            bet_on_a = edge_a > 0
        else:
            edge = abs(edge_b)
            bet_on_a = edge_b < 0

        pre_match = prices[msi]
        pmc = pre_match if bet_on_a else 1.0 - pre_match
        open_p = price_a if bet_on_a else 1.0 - price_a
        clv = round(pmc - open_p, 4)

        model_p = model_a if bet_on_a else 1.0 - model_a
        actual = mkt["actual"]
        won = (bet_on_a and actual == 1.0) or (not bet_on_a and actual == 0.0)

        candidates.append({
            "league": mkt["league"], "market_date": mkt["market_date"],
            "bet_team": mkt["db_a"] if bet_on_a else mkt["db_b"],
            "opponent": mkt["db_b"] if bet_on_a else mkt["db_a"],
            "model_prob": round(model_p, 3), "market_price": round(open_p, 3),
            "edge": round(edge, 3), "passes_threshold": edge >= threshold,
            "won": won, "clv": clv, "volume": mkt["volume"],
        })

    return candidates, n_priced


def _trade_roi_fraction(c: Dict, cost_multiplier: float = 1.0) -> float:
    """Per-bet ROI as a fraction of stake, using the real T1 cost model
    (entry = quoted price + cost, payout = 1/entry if win, -1 if loss)."""
    cost = estimate_t1_cost(c["volume"], cost_multiplier)
    entry = min(c["market_price"] + cost, 0.99)
    if not (0 < entry < 1):
        return 0.0
    return (1.0 / entry - 1.0) if c["won"] else -1.0


def summarize_timing_point(candidates: List[Dict], n_priced: int, n_total_markets: int) -> Dict:
    eligible = [c for c in candidates if c["passes_threshold"]]
    n = len(eligible)
    if n == 0:
        return {
            "n_priced": n_priced, "n_total": n_total_markets, "n_trades": 0,
            "pct_with_edge": 0.0, "hit_rate": None, "mean_clv": None, "mean_edge": None,
            "roi_per_bet": None,
        }
    wins = sum(1 for c in eligible if c["won"])
    rois = np.array([_trade_roi_fraction(c) for c in eligible])

    rng = np.random.RandomState(42)
    boot_roi = [np.mean(rng.choice(rois, n, replace=True)) for _ in range(5000)]
    boot_hit = [
        np.mean(rng.choice([1 if c["won"] else 0 for c in eligible], n, replace=True))
        for _ in range(5000)
    ]

    return {
        "n_priced": n_priced,
        "n_total": n_total_markets,
        "n_trades": n,
        "pct_with_edge": n / n_priced * 100 if n_priced else 0.0,
        "hit_rate": wins / n,
        "hit_rate_ci": (float(np.percentile(boot_hit, 2.5)), float(np.percentile(boot_hit, 97.5))),
        "mean_clv": float(np.mean([c["clv"] for c in eligible])),
        "mean_edge": float(np.mean([c["edge"] for c in eligible])),
        "roi_per_bet": float(np.mean(rois)) * 100,
        "roi_per_bet_ci": (float(np.percentile(boot_roi, 2.5)) * 100, float(np.percentile(boot_roi, 97.5)) * 100),
        "wins": wins,
        "losses": n - wins,
    }


def run_fine_timing_grid(enriched: List[Dict], threshold: float = DEFAULT_THRESHOLD) -> Dict:
    n_total = len(enriched)
    results = {}

    for offset in ENTRY_POINTS_MIN:
        cands, n_priced = build_trade_candidates_at_entry(enriched, entry_offset_min=offset, threshold=threshold)
        results[ENTRY_LABELS[offset]] = summarize_timing_point(cands, n_priced, n_total)
        results[ENTRY_LABELS[offset]]["candidates"] = cands

    for hb in HOURS_BEFORE:
        cands, n_priced = build_trade_candidates_at_entry(enriched, hours_before=hb, threshold=threshold)
        results[f"{hb}h before match"] = summarize_timing_point(cands, n_priced, n_total)
        results[f"{hb}h before match"]["candidates"] = cands

    return results


# ---------------------------------------------------------------------------
# Phase 4: bankroll sweep with depth gating + cost sensitivity
# ---------------------------------------------------------------------------
def run_bankroll_sweep(
    candidates: List[Dict], market_volumes: Dict, starting_bankroll: float,
    cost_multiplier: float = 1.0, kelly_max: float = DEFAULT_KELLY,
    cap_pct: float = DEFAULT_CAP, depth_scale: float = 1.0,
) -> Dict:
    eligible = [c for c in candidates if c["passes_threshold"]]
    eligible.sort(key=lambda c: c["market_date"])

    bankroll = float(starting_bankroll)
    peak = bankroll
    max_dd = 0.0
    trades = []
    depth_bound_count = 0
    kelly_bound_count = 0
    cap_bound_count = 0

    for c in eligible:
        volume = c["volume"]
        cost = estimate_t1_cost(volume, cost_multiplier)
        fillable = estimate_t1_fillable(volume) * depth_scale

        raw_entry = c["market_price"]
        entry = min(raw_entry + cost, 0.99)
        if not (0 < entry < 1):
            continue

        model_p = c["model_prob"]
        b_odds = (1.0 / entry) - 1.0
        kelly = max(0.0, min((model_p * b_odds - (1.0 - model_p)) / b_odds, kelly_max))
        kelly_size = bankroll * kelly
        cap_size = bankroll * cap_pct

        size = min(kelly_size, cap_size, fillable)
        if size == fillable and fillable < kelly_size and fillable < cap_size:
            depth_bound_count += 1
        elif size == cap_size and cap_size < kelly_size:
            cap_bound_count += 1
        elif size == kelly_size:
            kelly_bound_count += 1

        size = round(size, 2)
        if size < 1.0:
            continue

        won = c["won"]
        pnl = round(size * (1.0 / entry - 1.0) if won else -size, 2)
        bankroll = round(bankroll + pnl, 2)

        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        trades.append({"size": size, "fillable": fillable, "won": won, "pnl": pnl, "bankroll": bankroll})

        if bankroll < 10:
            break

    n_trades = len(trades)
    wins = sum(1 for t in trades if t["won"])

    return {
        "starting_bankroll": starting_bankroll,
        "cost_multiplier": cost_multiplier,
        "n_eligible": len(eligible),
        "n_trades": n_trades,
        "wins": wins,
        "hit_rate": wins / n_trades if n_trades else 0.0,
        "final_bankroll": bankroll,
        "total_pnl": bankroll - starting_bankroll,
        "roi": (bankroll - starting_bankroll) / starting_bankroll * 100,
        "max_dd": max_dd * 100,
        "depth_bound_pct": depth_bound_count / n_trades * 100 if n_trades else 0.0,
        "cap_bound_pct": cap_bound_count / n_trades * 100 if n_trades else 0.0,
        "kelly_bound_pct": kelly_bound_count / n_trades * 100 if n_trades else 0.0,
        "avg_stake": float(np.mean([t["size"] for t in trades])) if trades else 0.0,
        "avg_fillable": float(np.mean([t["fillable"] for t in trades])) if trades else 0.0,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_timing_grid(results: Dict) -> None:
    print(f"\n{'='*100}")
    print(f"  FINE-GRAINED ENTRY-TIMING BACKTEST (real V2 model, real Polymarket T1 prices)")
    print(f"{'='*100}")
    print(f"  {'Entry Point':<20} {'Priced':>7} {'Trades':>7} {'%w/Edge':>8} {'Hit%':>6} {'Mean CLV':>9} {'Mean Edge':>10} {'ROI/bet':>8} {'ROI 95% CI':>16}")
    print(f"  {'-'*106}")

    for offset in ENTRY_POINTS_MIN:
        label = ENTRY_LABELS[offset]
        r = results[label]
        hit_str = f"{r['hit_rate']:.1%}" if r['hit_rate'] is not None else "  n/a"
        clv_str = f"{r['mean_clv']:+.4f}" if r['mean_clv'] is not None else "    n/a"
        edge_str = f"{r['mean_edge']:.3f}" if r['mean_edge'] is not None else "  n/a"
        roi_str = f"{r['roi_per_bet']:+.1f}%" if r.get('roi_per_bet') is not None else "  n/a"
        ci = r.get('roi_per_bet_ci')
        ci_str = f"[{ci[0]:+.0f}%, {ci[1]:+.0f}%]" if ci else "n/a"
        print(f"  {label:<20} {r['n_priced']:>7} {r['n_trades']:>7} {r['pct_with_edge']:>7.1f}% {hit_str:>6} {clv_str:>9} {edge_str:>10} {roi_str:>8} {ci_str:>16}")

    print(f"\n  Hours-before-match grid (for comparison to coarse CLAUDE.md numbers):")
    print(f"  {'-'*88}")
    for hb in HOURS_BEFORE:
        label = f"{hb}h before match"
        r = results[label]
        hit_str = f"{r['hit_rate']:.1%}" if r['hit_rate'] is not None else "  n/a"
        clv_str = f"{r['mean_clv']:+.4f}" if r['mean_clv'] is not None else "    n/a"
        edge_str = f"{r['mean_edge']:.3f}" if r['mean_edge'] is not None else "  n/a"
        roi_str = f"{r['roi_per_bet']:+.1f}%" if r.get('roi_per_bet') is not None else "  n/a"
        print(f"  {label:<20} {r['n_priced']:>7} {r['n_trades']:>7} {r['pct_with_edge']:>7.1f}% {hit_str:>6} {clv_str:>9} {edge_str:>10} {roi_str:>8}")
    print()


DEPTH_SCALES = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]


def run_depth_sensitivity(candidates: List[Dict], bankroll: float) -> List[Dict]:
    """Scale the (known-unmeasured) fillable estimate down to find the point
    where real-world depth, if thinner than the lifetime-volume proxy assumes,
    would start constraining stake size and eroding ROI."""
    results = []
    for scale in DEPTH_SCALES:
        r = run_bankroll_sweep(candidates, {}, bankroll, cost_multiplier=1.0, depth_scale=scale)
        r["depth_scale"] = scale
        results.append(r)
    return results


def print_depth_sensitivity(results: List[Dict], bankroll: float) -> None:
    print(f"\n{'='*100}")
    print(f"  DEPTH SENSITIVITY at ${bankroll:,.0f} bankroll — fillable estimate scaled down from the lifetime-volume proxy")
    print(f"  (the proxy itself is UNVALIDATED — this shows what happens to returns IF real opening depth is thinner)")
    print(f"{'='*100}")
    print(f"  {'Depth Scale':>12} {'Implied AvgFillable':>20} {'Trades':>7} {'Final':>12} {'ROI':>8} {'AvgStake':>9} {'DepthBnd%':>10}")
    print(f"  {'-'*86}")
    for r in results:
        print(
            f"  {r['depth_scale']:>11.0%} ${r['avg_fillable']:>18,.0f} {r['n_trades']:>7}"
            f" ${r['final_bankroll']:>10,.0f} {r['roi']:>+7.0f}% ${r['avg_stake']:>7,.0f} {r['depth_bound_pct']:>9.1f}%"
        )
    print()


def print_bankroll_sweep(results: List[Dict]) -> None:
    print(f"\n{'='*120}")
    print(f"  BANKROLL SWEEP — real model trades at market open, quarter-Kelly, 2% cap, depth-gated")
    print(f"{'='*120}")
    print(f"  {'Bankroll':>10} {'Cost×':>6} {'Trades':>7} {'Hit%':>6} {'Final':>12} {'ROI':>8} {'MaxDD':>7} {'AvgStake':>9} {'DepthBnd%':>10} {'CapBnd%':>9}")
    print(f"  {'-'*100}")
    for r in results:
        print(
            f"  ${r['starting_bankroll']:>9,.0f} {r['cost_multiplier']:>5.1f}x {r['n_trades']:>7} {r['hit_rate']:>5.0%}"
            f" ${r['final_bankroll']:>10,.0f} {r['roi']:>+7.0f}% {r['max_dd']:>6.1f}% ${r['avg_stake']:>7,.0f}"
            f" {r['depth_bound_pct']:>9.1f}% {r['cap_bound_pct']:>8.1f}%"
        )
    print()


def write_trades_csv(candidates: List[Dict], path: Path) -> None:
    if not candidates:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        keys = [k for k in candidates[0].keys() if k != "candidates"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for c in candidates:
            writer.writerow({k: c[k] for k in keys})


def main() -> None:
    parser = argparse.ArgumentParser(description="Real T1 model-based backtest")
    parser.add_argument("--csv", action="store_true", help="Write trade CSVs")
    parser.add_argument("--json-out", type=str, default=None, help="Write full results to JSON")
    args = parser.parse_args()

    logger.info("Loading and matching T1 markets to DB teams…")
    matched = load_and_match_markets()

    logger.info("Building walk-forward ELO predictions…")
    enriched = build_walkforward_predictions(matched)

    logger.info("Computing per-league walk-forward accuracy (model only)…")
    league_acc = per_league_accuracy()
    print(f"\n{'='*60}")
    print(f"  PER-LEAGUE WALK-FORWARD ACCURACY (eval since 2024-01-01)")
    print(f"{'='*60}")
    for league, r in league_acc.items():
        print(f"  {league:<22} acc={r['accuracy']:.1%}  brier={r['brier']:.4f}  n={r['n']}")
    print()

    logger.info("Running fine-grained entry-timing grid…")
    timing_results = run_fine_timing_grid(enriched)
    print_timing_grid(timing_results)

    logger.info("Running bankroll sweep with cost sensitivity…")
    open_candidates, _ = build_trade_candidates_at_entry(enriched, entry_offset_min=0)
    volumes = {}

    sweep_results = []
    for bankroll in BANKROLLS:
        for mult in COST_MULTIPLIERS:
            r = run_bankroll_sweep(open_candidates, volumes, bankroll, cost_multiplier=mult)
            sweep_results.append(r)
    print_bankroll_sweep(sweep_results)

    logger.info("Running depth sensitivity scan…")
    depth_results = {}
    for bankroll in [10000, 50000]:
        dr = run_depth_sensitivity(open_candidates, bankroll)
        depth_results[bankroll] = dr
        print_depth_sensitivity(dr, bankroll)

    if args.csv:
        out_dir = _ROOT / "data"
        write_trades_csv(timing_results[ENTRY_LABELS[0]]["candidates"], out_dir / "t1_model_trades_open.csv")
        write_trades_csv(timing_results[ENTRY_LABELS[1440]]["candidates"], out_dir / "t1_model_trades_24h.csv")
        logger.info(f"Trade CSVs written to {out_dir}")

    if args.json_out:
        out = {
            "league_accuracy": league_acc,
            "timing": {k: {kk: vv for kk, vv in v.items() if kk != "candidates"} for k, v in timing_results.items()},
            "bankroll_sweep": sweep_results,
            "depth_sensitivity": depth_results,
            "n_matched_markets": len(matched),
            "n_enriched_markets": len(enriched),
        }
        Path(args.json_out).write_text(json.dumps(out, indent=2, default=str))
        logger.info(f"Full results written to {args.json_out}")


if __name__ == "__main__":
    main()
