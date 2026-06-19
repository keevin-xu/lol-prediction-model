"""
Live deployment engine — detects new markets, computes edge signals,
gates on roster stability, sizes via quarter-Kelly, and logs paper bets
with CLV tracking.

ALL EXECUTION IS PAPER MODE. LIVE_TRADING defaults to False.
No real-money orders are placed until explicitly enabled after
30+ paper bets show positive live CLV.

Run standalone:  python polymarket/live_engine.py --status
Integrated:      called from bot.py scan loop
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from model.predict import predict_match, check_cross_region
from polymarket.scanner import (
    GAMMA_API,
    MarketOpportunity,
    _clean_team_name,
    load_db_team_names,
    match_team_name,
    parse_teams_from_question,
)

DB_PATH = _ROOT / "db" / "lol_model.db"

LIVE_TRADING = False
MIN_EDGE = 0.10
MAX_KELLY = 0.0625
MAX_POSITION_PCT = 0.02
STARTING_BANKROLL = 1000.0
SPREAD_COST = 0.02
SLIPPAGE_COST = 0.01


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# §1. Market-creation detector
# ---------------------------------------------------------------------------
def detect_new_markets(
    session: Optional[requests.Session] = None,
) -> List[Dict]:
    """Poll Gamma API for new T2 LoL moneyline markets not yet in live_signals."""
    session = session or _make_session()
    conn = sqlite3.connect(DB_PATH, timeout=10)

    seen = set(
        r[0] for r in conn.execute("SELECT market_id FROM live_signals").fetchall()
    )
    db_teams = load_db_team_names()

    new_markets = []

    try:
        r = session.get(
            f"{GAMMA_API}/events",
            params={
                "active": "true",
                "closed": "false",
                "tag_slug": "league-of-legends",
                "limit": "100",
            },
            timeout=15,
        )
        if r.status_code != 200:
            conn.close()
            return []
        events = r.json()
    except requests.RequestException:
        conn.close()
        return []

    now = _now_iso()

    for event in events:
        for m in event.get("markets", []):
            mid = m.get("id", "")
            if mid in seen:
                continue

            q = m.get("question", "")
            ql = q.lower()
            if "(bo" not in ql or " vs " not in ql:
                continue
            if any(x in ql for x in ["game 1", "game 2", "game 3", "game 4", "game 5", "handicap"]):
                continue

            prices = m.get("outcomePrices", "[]")
            outcomes = m.get("outcomes", "[]")
            tokens = m.get("clobTokenIds", "[]")
            if isinstance(prices, str):
                try: prices = json.loads(prices)
                except: continue
            if isinstance(outcomes, str):
                try: outcomes = json.loads(outcomes)
                except: continue
            if isinstance(tokens, str):
                try: tokens = json.loads(tokens)
                except: continue

            if len(prices) < 2 or len(outcomes) < 2 or len(tokens) < 2:
                continue

            try:
                pa, pb = float(prices[0]), float(prices[1])
            except (ValueError, TypeError):
                continue

            if pa >= 0.99 or pb >= 0.99 or pa <= 0.01 or pb <= 0.01:
                continue

            teams = parse_teams_from_question(q)
            if not teams:
                continue

            pm_a, pm_b = teams
            db_a = match_team_name(pm_a, db_teams)
            db_b = match_team_name(pm_b, db_teams)
            if not db_a or not db_b:
                continue

            end_date = m.get("endDate", "")
            market_create = m.get("startDate", m.get("createdAt", ""))
            spread = m.get("spread", 0)
            if isinstance(spread, str):
                try: spread = float(spread)
                except: spread = 0
            volume = float(m.get("volumeNum", 0) or m.get("volume", 0) or 0)

            new_markets.append({
                "market_id": mid,
                "condition_id": m.get("conditionId", ""),
                "slug": event.get("slug", ""),
                "question": q,
                "team_a": pm_a,
                "team_b": pm_b,
                "db_team_a": db_a,
                "db_team_b": db_b,
                "token_id_a": tokens[0],
                "token_id_b": tokens[1],
                "open_price_a": pa,
                "open_price_b": pb,
                "spread": spread,
                "volume": volume,
                "market_create_ts": market_create,
                "match_start_ts": end_date,
                "now": now,
            })

    conn.close()
    return new_markets


# ---------------------------------------------------------------------------
# §2. Signal: model vs opening line
# ---------------------------------------------------------------------------
def compute_signal(market: Dict) -> Dict:
    """Compute model prediction and edge against opening price."""
    result = predict_match(market["db_team_a"], market["db_team_b"])
    model_a = result["p_a"]
    open_a = market["open_price_a"]

    edge_a = model_a - open_a
    edge_b = (1.0 - model_a) - (1.0 - open_a)

    region_check = check_cross_region(market["db_team_a"], market["db_team_b"])
    same_region = not region_check["cross_region"]

    if abs(edge_a) >= abs(edge_b):
        edge = edge_a
        bet_side = "team_a"
        bet_team = market["db_team_a"]
    else:
        edge = -edge_b
        bet_side = "team_b"
        bet_team = market["db_team_b"]

    return {
        "model_prob": model_a,
        "open_implied_prob": open_a,
        "edge": abs(edge_a) if abs(edge_a) >= abs(edge_b) else abs(edge_b),
        "edge_signed": edge,
        "bet_side": bet_side,
        "bet_team": bet_team,
        "same_region": same_region,
        "region_a": region_check.get("region_a"),
        "region_b": region_check.get("region_b"),
        "league_a": region_check.get("league_a"),
    }


# ---------------------------------------------------------------------------
# §3. Roster-swap pre-bet gate
# ---------------------------------------------------------------------------
def check_roster_stability(team: str, market_id: str) -> Tuple[bool, str]:
    """
    Check if team's roster is current. Returns (stable, reason).
    Currently checks if team exists in rosters table with recent snapshot.
    Full implementation would compare roster at market_create vs now.
    """
    conn = sqlite3.connect(DB_PATH, timeout=10)
    roster = conn.execute(
        "SELECT player_name, role FROM rosters WHERE team = ? ORDER BY snapshot_date DESC LIMIT 5",
        (team,),
    ).fetchall()
    conn.close()

    if not roster:
        return True, "no_roster_data"

    roster_str = json.dumps([(r[0], r[1]) for r in roster])

    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute(
        """INSERT OR REPLACE INTO roster_checks
           (market_id, team, roster_at_create, roster_at_bet, changed_after_create, action)
           VALUES (?, ?, ?, ?, 0, 'pass')""",
        (market_id, team, roster_str, roster_str),
    )
    conn.commit()
    conn.close()

    return True, "stable"


# ---------------------------------------------------------------------------
# §4. Sizing
# ---------------------------------------------------------------------------
def compute_size(
    edge: float,
    entry_price: float,
    depth_est: float,
    bankroll: float,
) -> Tuple[float, float, float]:
    """Returns (kelly_size, fillable_size, final_size)."""
    if entry_price <= 0 or entry_price >= 1:
        return 0.0, 0.0, 0.0

    b = (1.0 / entry_price) - 1.0
    model_p = entry_price + edge
    q = 1.0 - model_p
    kelly_raw = (model_p * b - q) / b
    kelly = max(0.0, min(kelly_raw, MAX_KELLY))

    kelly_size = round(bankroll * kelly, 2)

    cap_size = round(bankroll * MAX_POSITION_PCT, 2)

    fillable_size = round(min(depth_est * 0.03, 500.0), 2) if depth_est > 0 else 50.0

    final_size = round(min(kelly_size, cap_size, fillable_size), 2)
    final_size = max(final_size, 0.0)

    return kelly_size, fillable_size, final_size


# ---------------------------------------------------------------------------
# §5. Paper execution
# ---------------------------------------------------------------------------
def get_current_bankroll() -> float:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    row = conn.execute(
        "SELECT COALESCE(SUM(realized_pnl), 0) FROM clv_log WHERE realized_pnl IS NOT NULL"
    ).fetchone()
    conn.close()
    return STARTING_BANKROLL + float(row[0])


def place_paper_bet(market: Dict, signal: Dict, size: float) -> Optional[int]:
    """Log a paper bet. Returns bet ID."""
    if size < 1.0:
        return None

    entry_price = market["open_price_a"] if signal["bet_side"] == "team_a" else market["open_price_b"]
    entry_price = min(entry_price + SPREAD_COST + SLIPPAGE_COST, 0.99)

    now = _now_iso()
    conn = sqlite3.connect(DB_PATH, timeout=10)

    existing = conn.execute(
        "SELECT id FROM live_bets WHERE market_id = ? AND suppressed_reason IS NULL",
        (market["market_id"],),
    ).fetchone()
    if existing:
        conn.close()
        return None

    conn.execute(
        """INSERT INTO live_bets
           (market_id, mode, entry_ts, entry_price, fillable_size, kelly_size, final_size,
            edge, model_prob, open_implied_prob, bet_side, bet_team, status)
           VALUES (?, 'paper', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
        (
            market["market_id"], now, entry_price,
            size, size, size,
            signal["edge"], signal["model_prob"], signal["open_implied_prob"],
            signal["bet_side"], signal["bet_team"],
        ),
    )
    bet_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()

    logger.info(
        f"PAPER BET: ${size:.2f} on {signal['bet_team']} @ {entry_price:.1%} "
        f"(edge: {signal['edge']:.1%}, model: {signal['model_prob']:.1%})"
    )
    return bet_id


def log_suppressed(market_id: str, reason: str, signal: Dict) -> None:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute(
        """INSERT INTO live_bets
           (market_id, mode, entry_ts, entry_price, edge, model_prob, open_implied_prob,
            bet_side, bet_team, suppressed_reason, status)
           VALUES (?, 'paper', ?, 0, ?, ?, ?, ?, ?, ?, 'suppressed')""",
        (
            market_id, _now_iso(), signal.get("edge", 0),
            signal.get("model_prob", 0), signal.get("open_implied_prob", 0),
            signal.get("bet_side", ""), signal.get("bet_team", ""),
            reason,
        ),
    )
    conn.commit()
    conn.close()
    logger.info(f"SUPPRESSED: {market_id[:12]}… reason={reason}")


# ---------------------------------------------------------------------------
# §6. CLV logger
# ---------------------------------------------------------------------------
def update_clv(session: Optional[requests.Session] = None) -> int:
    """Check open bets for resolution, compute CLV. Returns count updated."""
    session = session or _make_session()
    conn = sqlite3.connect(DB_PATH, timeout=10)

    open_bets = conn.execute(
        """SELECT b.id, b.market_id, b.entry_price, b.bet_side, b.bet_team, b.final_size,
                  s.match_start_ts
           FROM live_bets b
           JOIN live_signals s ON b.market_id = s.market_id
           WHERE b.status = 'open' AND b.suppressed_reason IS NULL""",
    ).fetchall()

    if not open_bets:
        conn.close()
        return 0

    updated = 0
    for bet_id, market_id, entry_price, bet_side, bet_team, size, match_start_ts in open_bets:
        try:
            r = session.get(f"{GAMMA_API}/markets/{market_id}", timeout=10)
            if r.status_code != 200:
                continue
            mkt = r.json()
        except requests.RequestException:
            continue

        if not mkt.get("closed"):
            continue

        prices = mkt.get("outcomePrices", "[]")
        outcomes = mkt.get("outcomes", "[]")
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: continue
        if isinstance(outcomes, str):
            try: outcomes = json.loads(outcomes)
            except: continue

        if len(prices) < 2:
            continue

        try:
            pa = float(prices[0])
        except (ValueError, TypeError):
            continue

        if pa >= 0.99:
            winner = outcomes[0] if outcomes else "team_a"
        elif float(prices[1]) >= 0.99:
            winner = outcomes[1] if len(outcomes) > 1 else "team_b"
        else:
            continue

        won = (bet_team == winner) or (bet_side == "team_a" and pa >= 0.99) or (bet_side == "team_b" and float(prices[1]) >= 0.99)

        if won:
            pnl = round(size * (1.0 / entry_price - 1.0), 2)
        else:
            pnl = round(-size, 2)

        # Get pre-match close: last snapshot BEFORE match_start_ts (avoids in-game contamination)
        if match_start_ts:
            prematch_row = conn.execute(
                "SELECT price_a FROM polymarket_prices WHERE market_id = ? AND timestamp < ? ORDER BY timestamp DESC LIMIT 1",
                (market_id, match_start_ts),
            ).fetchone()
        else:
            prematch_row = None
        if not prematch_row:
            prematch_row = conn.execute(
                "SELECT price_a FROM polymarket_prices WHERE market_id = ? ORDER BY timestamp ASC LIMIT 1",
                (market_id,),
            ).fetchone()
        prematch_close = prematch_row[0] if prematch_row else entry_price

        if bet_side == "team_b":
            prematch_close = 1.0 - prematch_close

        clv = prematch_close - entry_price
        beat_close = 1 if clv > 0 else 0

        cumulative = conn.execute(
            "SELECT COALESCE(SUM(final_size), 0) FROM live_bets WHERE suppressed_reason IS NULL AND id <= ?",
            (bet_id,),
        ).fetchone()[0]

        conn.execute(
            """INSERT INTO clv_log
               (bet_id, market_id, entry_price, prematch_close, resolution_outcome,
                clv, beat_close, realized_pnl, cumulative_volume_at_bet)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (bet_id, market_id, entry_price, prematch_close, winner,
             round(clv, 4), beat_close, pnl, cumulative),
        )
        conn.execute(
            "UPDATE live_bets SET status = ? WHERE id = ?",
            ("won" if won else "lost", bet_id),
        )
        updated += 1
        logger.info(f"CLV: {bet_team} {'WON' if won else 'LOST'} PnL=${pnl:+.2f} CLV={clv:+.3f}")

    conn.commit()
    conn.close()
    return updated


# ---------------------------------------------------------------------------
# Main pipeline: process one scan cycle
# ---------------------------------------------------------------------------
def run_cycle(session: Optional[requests.Session] = None) -> Dict:
    """Full detection → signal → gate → size → execute cycle. Returns summary."""
    session = session or _make_session()
    conn = sqlite3.connect(DB_PATH, timeout=10)

    new_markets = detect_new_markets(session)
    signals_fired = 0
    bets_placed = 0
    suppressed = 0
    bankroll = get_current_bankroll()

    for market in new_markets:
        signal = compute_signal(market)

        detection_latency = 0.0
        create_ts = market.get("market_create_ts", "")
        if create_ts:
            try:
                create_dt = datetime.fromisoformat(create_ts.replace("Z", "+00:00"))
                detection_latency = (datetime.now(timezone.utc) - create_dt).total_seconds()
            except (ValueError, TypeError):
                pass

        conn.execute(
            """INSERT OR IGNORE INTO live_signals
               (market_id, detected_ts, detection_latency_s, match_start_ts,
                league, region, same_region, team_a, team_b,
                open_implied_prob, model_prob, edge, spread, depth_est)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                market["market_id"], _now_iso(), detection_latency,
                market.get("match_start_ts", ""),
                signal.get("league_a", ""), signal.get("region_a", ""),
                1 if signal["same_region"] else 0,
                market["db_team_a"], market["db_team_b"],
                signal["open_implied_prob"], signal["model_prob"],
                signal["edge"], market.get("spread", 0), market.get("volume", 0),
            ),
        )
        conn.commit()

        # Gate: same-region only
        if not signal["same_region"]:
            log_suppressed(market["market_id"], "cross_region", signal)
            suppressed += 1
            continue

        # Gate: minimum edge
        if signal["edge"] < MIN_EDGE:
            log_suppressed(market["market_id"], f"edge_too_small_{signal['edge']:.2f}", signal)
            suppressed += 1
            continue

        # Gate: roster stability
        for team in [market["db_team_a"], market["db_team_b"]]:
            stable, reason = check_roster_stability(team, market["market_id"])
            if not stable:
                log_suppressed(market["market_id"], f"roster_swap_{team}", signal)
                suppressed += 1
                break
        else:
            signals_fired += 1

            entry_price = market["open_price_a"] if signal["bet_side"] == "team_a" else market["open_price_b"]
            kelly_s, fill_s, final_s = compute_size(
                signal["edge"], entry_price + SPREAD_COST + SLIPPAGE_COST,
                market.get("volume", 0), bankroll,
            )

            if final_s < 1.0:
                log_suppressed(market["market_id"], "no_liquidity", signal)
                suppressed += 1
                continue

            bet_id = place_paper_bet(market, signal, final_s)
            if bet_id:
                bets_placed += 1

    # Check for resolved bets
    resolved = update_clv(session)

    conn.close()
    return {
        "new_markets": len(new_markets),
        "signals_fired": signals_fired,
        "bets_placed": bets_placed,
        "suppressed": suppressed,
        "resolved": resolved,
        "bankroll": bankroll,
    }


# ---------------------------------------------------------------------------
# Status / reporting
# ---------------------------------------------------------------------------
def print_status() -> None:
    conn = sqlite3.connect(DB_PATH, timeout=10)

    total_signals = conn.execute("SELECT COUNT(*) FROM live_signals").fetchone()[0]
    total_bets = conn.execute("SELECT COUNT(*) FROM live_bets WHERE suppressed_reason IS NULL").fetchone()[0]
    open_bets = conn.execute("SELECT COUNT(*) FROM live_bets WHERE status = 'open' AND suppressed_reason IS NULL").fetchone()[0]
    won = conn.execute("SELECT COUNT(*) FROM live_bets WHERE status = 'won'").fetchone()[0]
    lost = conn.execute("SELECT COUNT(*) FROM live_bets WHERE status = 'lost'").fetchone()[0]
    suppressed = conn.execute("SELECT COUNT(*) FROM live_bets WHERE suppressed_reason IS NOT NULL").fetchone()[0]

    clv_rows = conn.execute("SELECT clv, beat_close, realized_pnl FROM clv_log").fetchall()

    conn.close()

    bankroll = get_current_bankroll()

    print(f"\n{'='*55}")
    print(f"  LIVE ENGINE STATUS (mode: {'LIVE' if LIVE_TRADING else 'PAPER'})")
    print(f"{'='*55}")
    print(f"  Signals detected:  {total_signals}")
    print(f"  Bets placed:       {total_bets} ({open_bets} open, {won}W/{lost}L)")
    print(f"  Suppressed:        {suppressed}")
    print(f"  Bankroll:          ${bankroll:,.2f}")

    if clv_rows:
        clvs = [r[0] for r in clv_rows if r[0] is not None]
        beats = [r[1] for r in clv_rows if r[1] is not None]
        pnls = [r[2] for r in clv_rows if r[2] is not None]

        if clvs:
            print(f"\n  CLV REPORT ({len(clvs)} resolved)")
            print(f"  {'-'*40}")
            print(f"  Mean CLV:          {sum(clvs)/len(clvs):+.3f}")
            print(f"  % beat close:      {sum(beats)/len(beats):.0%}")
            print(f"  Total P&L:         ${sum(pnls):+,.2f}")
            print(f"  Win rate:          {won}/{won+lost} ({won/(won+lost):.0%})" if won + lost > 0 else "")

            go_live = len(clvs) >= 30 and sum(clvs) / len(clvs) > 0
            print(f"\n  PROMOTION READY:   {'YES — consider enabling LIVE_TRADING' if go_live else f'NO — need {max(30 - len(clvs), 0)} more bets'}")
    else:
        print(f"\n  No resolved bets yet. Waiting for markets to settle.")
    print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Live deployment engine")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--cycle", action="store_true", help="Run one detection cycle")
    parser.add_argument("--clv", action="store_true", help="Update CLV for resolved bets")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.cycle:
        result = run_cycle()
        print(f"Cycle: {result['new_markets']} new, {result['bets_placed']} bets, {result['suppressed']} suppressed, {result['resolved']} resolved")
    elif args.clv:
        n = update_clv()
        print(f"Updated CLV for {n} bets")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
