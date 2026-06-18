"""
Forward price collection — captures Polymarket price snapshots on each
scan cycle and tracks market resolution.

Integrates with the bot's scan loop: call record_prices() every 5 minutes
with the current MarketOpportunity list. Resolution checking runs in the
settle loop.

Run standalone:  python polymarket/price_tracker.py
"""

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from polymarket.scanner import GAMMA_API, MarketOpportunity

DB_PATH = _ROOT / "db" / "lol_model.db"


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


# ---------------------------------------------------------------------------
# Record prices (single transaction)
# ---------------------------------------------------------------------------
def record_prices(opportunities: List[MarketOpportunity]) -> int:
    """Register markets and record price snapshots in a single transaction."""
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH, timeout=10)
    recorded = 0

    for opp in opportunities:
        existing = conn.execute(
            "SELECT id FROM polymarket_markets WHERE market_id = ?",
            (opp.market_id,),
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE polymarket_markets SET last_seen = ? WHERE market_id = ?",
                (now, opp.market_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO polymarket_markets
                    (market_id, condition_id, slug, question, team_a, team_b,
                     db_team_a, db_team_b, token_id_a, token_id_b, url,
                     first_seen, last_seen, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    opp.market_id, opp.condition_id, opp.slug, opp.question,
                    opp.team_a, opp.team_b, opp.db_team_a, opp.db_team_b,
                    opp.token_id_a, opp.token_id_b, opp.url, now, now,
                ),
            )
            logger.info(f"New market registered: {opp.db_team_a} vs {opp.db_team_b} ({opp.market_id[:8]}…)")

        conn.execute(
            """
            INSERT INTO polymarket_prices
                (market_id, timestamp, price_a, price_b, spread, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, 'gamma')
            """,
            (opp.market_id, now, opp.market_prob_a, opp.market_prob_b,
             opp.spread, opp.volume),
        )
        recorded += 1

    conn.commit()
    conn.close()
    return recorded


# ---------------------------------------------------------------------------
# Resolution tracking
# ---------------------------------------------------------------------------
def check_market_resolutions(
    session: Optional[requests.Session] = None,
) -> List[str]:
    session = session or _make_session()
    conn = sqlite3.connect(DB_PATH, timeout=10)
    active = conn.execute(
        "SELECT market_id, db_team_a, db_team_b FROM polymarket_markets WHERE status = 'active'"
    ).fetchall()

    if not active:
        conn.close()
        return []

    resolved_ids = []
    for market_id, db_team_a, db_team_b in active:
        try:
            r = session.get(f"{GAMMA_API}/markets/{market_id}", timeout=10)
            if r.status_code != 200:
                continue
            market = r.json()
        except requests.RequestException:
            continue

        if not market.get("closed") or not market.get("resolvedBy"):
            continue

        prices = market.get("outcomePrices", [])
        if len(prices) < 2:
            continue

        try:
            p_a = float(prices[0])
            p_b = float(prices[1])
        except (ValueError, TypeError):
            continue

        if p_a == 1.0:
            winner = db_team_a
        elif p_b == 1.0:
            winner = db_team_b
        else:
            continue

        now = datetime.now(timezone.utc).isoformat()
        last_snap = conn.execute(
            "SELECT price_a, price_b FROM polymarket_prices WHERE market_id = ? ORDER BY timestamp DESC LIMIT 1",
            (market_id,),
        ).fetchone()

        conn.execute(
            """
            UPDATE polymarket_markets SET
                status = 'resolved', resolution_winner = ?, resolution_time = ?,
                closing_price_a = ?, closing_price_b = ?
            WHERE market_id = ?
            """,
            (winner, now, last_snap[0] if last_snap else None,
             last_snap[1] if last_snap else None, market_id),
        )
        resolved_ids.append(market_id)
        logger.info(f"Market resolved: {db_team_a} vs {db_team_b} → {winner}")

    conn.commit()
    conn.close()
    return resolved_ids


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def get_active_markets() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM polymarket_markets WHERE status = 'active'"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_closing_prices() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT market_id, db_team_a, db_team_b,
               closing_price_a, closing_price_b, resolution_winner
        FROM polymarket_markets
        WHERE status = 'resolved' AND closing_price_a IS NOT NULL
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
def main() -> None:
    active = get_active_markets()
    resolved = get_closing_prices()
    print(f"\nPolymarket Market Tracker")
    print(f"  Active markets:   {len(active)}")
    print(f"  Resolved markets: {len(resolved)}")

    if active:
        print(f"\nActive:")
        for m in active:
            print(f"  {m['db_team_a']} vs {m['db_team_b']} — {m['market_id'][:12]}…")

    if resolved:
        print(f"\nResolved:")
        for m in resolved:
            print(
                f"  {m['db_team_a']} vs {m['db_team_b']} → {m['resolution_winner']} "
                f"(closing: {m['closing_price_a']:.1%} / {m['closing_price_b']:.1%})"
            )
    print()


if __name__ == "__main__":
    main()
