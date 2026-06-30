"""
Fetch all resolved T2 LoL Polymarket markets with full price histories
and persist to disk as a frozen dataset.

Run once:  python backtest/persist_markets.py
Output:    data/polymarket_resolved.json
"""

import json
import sys
import time
from pathlib import Path

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

T2_KEYWORDS = [
    "lck challengers", "tcl", "ljl", "nacl", "north american challengers",
    "lfl", "nlc", "emea masters",
    "pcs", "vcs", "superliga", "prime league", "hitpoint", "road of legends",
]

T1_KEYWORDS = ["lck", "lpl", "lcs"]
T1_EXCLUDE = ["challengers", "academy", "youth"]


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Persist resolved Polymarket markets")
    parser.add_argument("--tier", choices=["t1", "t2"], default="t2",
                        help="Which tier to fetch: t1 or t2 (default)")
    args = parser.parse_args()

    tier = args.tier
    if tier == "t1":
        keywords = T1_KEYWORDS
        exclude = T1_EXCLUDE
        out_path = _ROOT / "data" / "polymarket_t1_resolved.json"
    else:
        keywords = T2_KEYWORDS
        exclude = []
        out_path = _ROOT / "data" / "polymarket_resolved.json"

    session = _make_session()

    # Step 1: Fetch all resolved LoL events
    all_events = []
    offset = 0
    for _ in range(20):
        r = session.get(
            "%s/events" % GAMMA_API,
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

    logger.info("Fetched %d resolved LoL events" % len(all_events))

    # Step 2: Filter to moneyline markets and fetch price histories
    markets = []
    skipped = 0

    for event in all_events:
        title = event.get("title", "").lower()
        if not any(kw in title for kw in keywords):
            continue
        if any(ex in title for ex in exclude):
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

            # Fetch full price history for team A's token
            try:
                r2 = session.get(
                    "%s/prices-history" % CLOB_API,
                    params={"market": tokens[0], "startTs": "1", "fidelity": "10"},
                    timeout=10,
                )
                if r2.status_code != 200:
                    skipped += 1
                    continue
                hist = r2.json().get("history", [])
                if len(hist) < 10:
                    skipped += 1
                    continue
            except requests.RequestException:
                skipped += 1
                continue

            markets.append({
                "event_title": event.get("title", ""),
                "question": q,
                "market_id": m.get("id", ""),
                "condition_id": m.get("conditionId", ""),
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner,
                "token_a": tokens[0],
                "token_b": tokens[1],
                "volume": float(m.get("volumeNum", 0) or m.get("volume", 0) or 0),
                "created_at": m.get("createdAt", ""),
                "closed_at": m.get("closedAt", ""),
                "price_history": hist,
            })

            if len(markets) % 25 == 0:
                logger.info("  %d markets fetched so far..." % len(markets))

            time.sleep(0.15)

    logger.info("Total: %d markets persisted, %d skipped" % (len(markets), skipped))

    # Step 3: Save to disk
    with open(out_path, "w") as f:
        json.dump(markets, f, indent=2)

    logger.info("Saved to %s (%.1f MB)" % (out_path, out_path.stat().st_size / 1e6))

    # Step 4: Quick summary
    print("\n" + "=" * 60)
    print("  PERSISTED MARKET SUMMARY")
    print("=" * 60)
    print("  Total markets: %d" % len(markets))

    from datetime import datetime
    dates = []
    for m in markets:
        hist = m["price_history"]
        if hist:
            dates.append(datetime.fromtimestamp(hist[0]["t"]).strftime("%Y-%m-%d"))
    if dates:
        print("  Date range: %s to %s" % (min(dates), max(dates)))

    # Price point counts
    point_counts = [len(m["price_history"]) for m in markets]
    print("  Price points per market: min=%d, median=%d, max=%d" %
          (min(point_counts), sorted(point_counts)[len(point_counts) // 2], max(point_counts)))

    # Volume distribution
    volumes = [m["volume"] for m in markets]
    volumes_sorted = sorted(volumes)
    print("  Volume: min=$%.0f, median=$%.0f, max=$%.0f" %
          (min(volumes), volumes_sorted[len(volumes) // 2], max(volumes)))

    print("\n  Data saved to: %s" % out_path)
    print("  This is a frozen artifact — all subsequent analysis uses this file.")
    print()


if __name__ == "__main__":
    main()
