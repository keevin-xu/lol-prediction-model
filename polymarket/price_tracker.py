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
CLOB_API = "https://clob.polymarket.com"


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


# ---------------------------------------------------------------------------
# Market registration
# ---------------------------------------------------------------------------
def upsert_market(opp: MarketOpportunity) -> bool:
    """Register or update a market. Returns True if this is a new market."""
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute(
        "SELECT id FROM polymarket_markets WHERE market_id = ?",
        (opp.market_id,),
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE polymarket_markets SET last_seen = ? WHERE market_id = ?",
            (now, opp.market_id),
        )
        conn.commit()
        conn.close()
        return False

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
    conn.commit()
    conn.close()
    logger.info(f"New market registered: {opp.db_team_a} vs {opp.db_team_b} ({opp.market_id[:8]}…)")
    return True


# ---------------------------------------------------------------------------
# Price snapshots
# ---------------------------------------------------------------------------
def record_prices(opportunities: List[MarketOpportunity]) -> int:
    """Record price snapshots for all opportunities. Returns count."""
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    recorded = 0

    for opp in opportunities:
        is_new = upsert_market(opp)
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

        if is_new:
            conn.commit()
            conn.close()
            backfill_market_history(opp.market_id, opp.token_id_a, opp.token_id_b)
            conn = sqlite3.connect(DB_PATH)

    conn.commit()
    conn.close()
    return recorded


# ---------------------------------------------------------------------------
# CLOB price history backfill
# ---------------------------------------------------------------------------
def fetch_clob_price_history(
    token_id: str,
    interval: str = "5m",
    fidelity: int = 500,
    session: Optional[requests.Session] = None,
) -> List[Dict]:
    session = session or _make_session()
    try:
        r = session.get(
            f"{CLOB_API}/prices-history",
            params={"market": token_id, "interval": interval, "fidelity": str(fidelity)},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "history" in data:
            return data["history"]
        if isinstance(data, list):
            return data
        return []
    except requests.RequestException as e:
        logger.warning(f"CLOB price history fetch failed for {token_id[:12]}…: {e}")
        return []


def backfill_market_history(
    market_id: str,
    token_id_a: str,
    token_id_b: str,
    interval: str = "5m",
    session: Optional[requests.Session] = None,
) -> int:
    session = session or _make_session()
    conn = sqlite3.connect(DB_PATH)
    inserted = 0

    for token_id in [token_id_a, token_id_b]:
        if not token_id:
            continue
        history = fetch_clob_price_history(token_id, interval=interval, session=session)
        for point in history:
            ts = point.get("t") or point.get("timestamp") or point.get("time")
            price = point.get("p") or point.get("price")
            if ts is None or price is None:
                continue
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO polymarket_price_history
                        (market_id, token_id, timestamp, price, interval)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (market_id, token_id, str(ts), float(price), interval),
                )
                inserted += 1
            except (ValueError, sqlite3.Error):
                continue

    conn.commit()
    conn.close()
    if inserted > 0:
        logger.info(f"Backfilled {inserted} price history points for market {market_id[:8]}…")
    return inserted


# ---------------------------------------------------------------------------
# Resolution tracking
# ---------------------------------------------------------------------------
def check_market_resolutions(
    session: Optional[requests.Session] = None,
) -> List[str]:
    session = session or _make_session()
    conn = sqlite3.connect(DB_PATH)
    active = conn.execute(
        "SELECT market_id, db_team_a, db_team_b FROM polymarket_markets WHERE status = 'active'"
    ).fetchall()
    conn.close()

    if not active:
        return []

    resolved_ids = []
    for market_id, db_team_a, db_team_b in active:
        try:
            r = session.get(f"{GAMMA_API}/markets/{market_id}", timeout=10)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            market = r.json()
        except requests.RequestException as e:
            logger.debug(f"Resolution check failed for {market_id[:8]}…: {e}")
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

        conn = sqlite3.connect(DB_PATH)
        # Get closing price: last snapshot before resolution
        last_snap = conn.execute(
            """
            SELECT price_a, price_b FROM polymarket_prices
            WHERE market_id = ? ORDER BY timestamp DESC LIMIT 1
            """,
            (market_id,),
        ).fetchone()

        closing_a = last_snap[0] if last_snap else None
        closing_b = last_snap[1] if last_snap else None

        conn.execute(
            """
            UPDATE polymarket_markets SET
                status = 'resolved',
                resolution_winner = ?,
                resolution_time = ?,
                closing_price_a = ?,
                closing_price_b = ?
            WHERE market_id = ?
            """,
            (winner, now, closing_a, closing_b, market_id),
        )
        conn.commit()
        conn.close()

        resolved_ids.append(market_id)
        logger.info(f"Market resolved: {db_team_a} vs {db_team_b} → {winner}")

    return resolved_ids


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
def get_market_history(market_id: str) -> Dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    market = conn.execute(
        "SELECT * FROM polymarket_markets WHERE market_id = ?", (market_id,)
    ).fetchone()
    if not market:
        conn.close()
        return {}

    prices = conn.execute(
        "SELECT timestamp, price_a, price_b, spread, volume FROM polymarket_prices "
        "WHERE market_id = ? ORDER BY timestamp ASC",
        (market_id,),
    ).fetchall()

    conn.close()
    return {
        "market": dict(market),
        "prices": [dict(p) for p in prices],
        "snapshots": len(prices),
    }


def get_active_markets() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM polymarket_markets WHERE status = 'active'"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_closing_prices() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
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
