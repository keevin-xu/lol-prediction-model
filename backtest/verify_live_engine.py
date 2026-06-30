"""
Live-path verification: proves the live engine's compute_signal() produces
the same point-in-time predictions as the backtest, anchored to market
open time (not scan time), with each match excluded from its own ELO.

Calls the REAL live_engine.compute_signal() — does not rebuild a tracker
to stand in for it. See conversation spec for why this matters: a prior
version of this test built two trackers by hand and compared them to
themselves, which can only ever pass and proves nothing about the engine.

Run:  python backtest/verify_live_engine.py
"""

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from backtest.backtest import ELOTracker, load_matches
from polymarket import live_engine

DB_PATH = _ROOT / "db" / "lol_model.db"


def build_reference_tracker(cutoff_date: str) -> ELOTracker:
    """The backtest's own construction — independent of the engine.
    This is the ONLY hand-built tracker in this test."""
    tracker = ELOTracker(
        K=32, blend_k=5, scale=400, half_life_days=270,
        soloq_baselines={}, regional_offsets={}, mov_weight=1.5,
    )
    for row in load_matches():
        date = row[1]
        if date >= cutoff_date:
            break
        stats = dict(blue_kills=row[6], red_kills=row[7],
                     blue_deaths=row[8], red_deaths=row[9],
                     blue_gd15=row[10], red_gd15=row[11])
        tracker.update(row[3], row[4], row[5], row[2], row[1], **stats)
    return tracker


def db_game_count_before(team: str, cutoff_date: str) -> int:
    """Ground truth: count of team's matches strictly before cutoff_date."""
    conn = sqlite3.connect(DB_PATH)
    n = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE (blue_team = ? OR red_team = ?) AND date < ?",
        (team, team, cutoff_date),
    ).fetchone()[0]
    conn.close()
    return n


def run_market_check(team_a: str, team_b: str, market_open_iso: str, league_a: str) -> bool:
    cutoff_date = market_open_iso[:10]

    market = {
        "db_team_a": team_a,
        "db_team_b": team_b,
        "open_price_a": 0.5,
        "market_create_ts": market_open_iso,
    }

    print("Market: %s vs %s  (open: %s)" % (team_a, team_b, market_open_iso))

    # --- Reference: hand-built backtest tracker (the only stand-in allowed) ---
    ref_tracker = build_reference_tracker(cutoff_date)
    reference_p = ref_tracker.predict(team_a, team_b, league_a, cutoff_date)

    # --- Live: call the REAL engine function ---
    live_engine._tracker_cache.clear()  # force rebuild so this test is hermetic
    signal = live_engine.compute_signal(market)
    live_p = signal["model_prob"]

    diff1 = abs(live_p - reference_p)
    pass1 = diff1 < 0.001

    # --- Assertion 2: open-anchored, not scan-time-anchored ---
    # Simulate two different scan times (wall-clock "now") while the
    # market's creation timestamp stays fixed. If compute_signal still used
    # datetime.now() for the cutoff, these would differ. With the fix, the
    # cutoff comes only from market_create_ts, so they must match exactly.
    open_dt = datetime.fromisoformat(market_open_iso.replace("Z", "+00:00"))
    scanned_at_open = open_dt
    scanned_5d_later = open_dt + timedelta(days=5)

    live_engine._tracker_cache.clear()
    with patch.object(live_engine, "datetime") as mock_dt:
        mock_dt.now.return_value = scanned_at_open
        mock_dt.fromisoformat = datetime.fromisoformat
        signal_open = live_engine.compute_signal(market)
    p_at_open = signal_open["model_prob"]

    live_engine._tracker_cache.clear()
    with patch.object(live_engine, "datetime") as mock_dt:
        mock_dt.now.return_value = scanned_5d_later
        mock_dt.fromisoformat = datetime.fromisoformat
        signal_5d = live_engine.compute_signal(market)
    p_5d_later = signal_5d["model_prob"]

    diff2 = abs(p_at_open - p_5d_later)
    pass2 = diff2 < 0.001

    # --- Assertion 3: match excluded from its own ELO ---
    expected_games_a = db_game_count_before(team_a, cutoff_date)
    expected_games_b = db_game_count_before(team_b, cutoff_date)
    actual_games_a = signal.get("games_a")
    actual_games_b = signal.get("games_b")
    pass3 = (actual_games_a == expected_games_a) and (actual_games_b == expected_games_b)

    print("  reference_p (backtest point-in-time):   %.4f" % reference_p)
    print("  live_p      (engine actual call):        %.4f" % live_p)
    print("  Assertion 1 (live == reference):         %s  (diff %.4f)" %
          ("PASS" if pass1 else "FAIL", diff1))
    print("  Assertion 2 (open-anchored, not scan):   %s  (open=%.4f 5d=%.4f)" %
          ("PASS" if pass2 else "FAIL", p_at_open, p_5d_later))
    print("  Assertion 3 (excludes own match):        %s  (games_a=%s expected=%s, games_b=%s expected=%s)" %
          ("PASS" if pass3 else "FAIL", actual_games_a, expected_games_a, actual_games_b, expected_games_b))
    print()

    return pass1 and pass2 and pass3


def main() -> None:
    print("LIVE-PATH VERIFICATION (engine method called directly, not simulated)")
    print("Engine method invoked: live_engine.compute_signal()")
    print()

    test_markets = [
        # T2 sub-league cases (original validation)
        ("Bomba Team", "Anubis Gaming", "2026-06-11T15:03:30Z"),
        ("Polar Squad Esports", "Zeu5 Esports", "2026-05-04T23:20:58Z"),
        ("E WIE EINFACH E-SPORTS", "Misa Esports", "2026-06-12T15:00:34Z"),
        # T1 LPL case — market from data/polymarket_t1_resolved.json.
        # These are the actual teams and market_create_ts the T1 scanner
        # will encounter. If compute_signal() fails any assertion here,
        # do not enable T1_SCANNING.
        ("Top Esports", "JD Gaming", "2026-01-03T08:30:03Z"),
    ]

    conn = sqlite3.connect(DB_PATH)
    team_leagues = {r[0]: r[1] or "" for r in conn.execute("SELECT team_name, league FROM teams").fetchall()}
    conn.close()

    results = []
    for team_a, team_b, open_ts in test_markets:
        league_a = team_leagues.get(team_a, "")
        results.append(run_market_check(team_a, team_b, open_ts, league_a))
        print("-" * 70)
        print()

    overall = all(results)
    print("OVERALL: %s" % ("PASS" if overall else "FAIL"))
    if not overall:
        print("The live path does not match the validated backtest. Do not")
        print("proceed to T1 scanning until all assertions pass on all markets.")


if __name__ == "__main__":
    main()
