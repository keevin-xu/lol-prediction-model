"""
Historical + live bookmaker odds scraper via the-odds-api.com.

Free tier: 500 requests/month. This scraper is designed to minimize
API calls — one call fetches all live LoL odds, and historical
scores are batched per event.

API docs: https://the-odds-api.com/liveapi/guides/v4/

Run:
  python scrapers/odds_scraper.py --live       # fetch current LoL odds (1 API call)
  python scrapers/odds_scraper.py --scores     # fetch recent results + odds (1 call)
  python scrapers/odds_scraper.py --link       # link odds to matches table
  python scrapers/odds_scraper.py --stats      # show coverage stats
  python scrapers/odds_scraper.py --quota      # check remaining API quota

Requires ODDS_API_KEY in .env (free at https://the-odds-api.com)
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

from scrapers.team_matcher import (
    bulk_match_teams,
    export_unmatched,
    load_db_team_names,
    match_team_name,
)

DB_PATH = _ROOT / "db" / "lol_model.db"
RAW_DIR = _ROOT / "data" / "raw" / "odds_api"

BASE_URL = "https://api.the-odds-api.com/v4"

# the-odds-api sport keys — as of June 2026, they do NOT cover esports.
# When they add it, the key will likely be "esports_lol" or similar.
# Run --discover to check for new sport keys.
LOL_SPORT_KEYS = [
    "esports_lol",
]

# Bookmakers to request (Pinnacle is sharpest; limit markets to save quota)
BOOKMAKERS = "pinnacle"
ODDS_FORMAT = "decimal"


def _get_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        logger.error(
            "ODDS_API_KEY not set.\n"
            "Sign up free at https://the-odds-api.com\n"
            "Add to .env: ODDS_API_KEY=your_key_here"
        )
    return key


def _make_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


# ---------------------------------------------------------------------------
# Odds math
# ---------------------------------------------------------------------------
def decimal_to_implied(odds: float) -> float:
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def remove_vig(prob_a: float, prob_b: float) -> Tuple[float, float]:
    total = prob_a + prob_b
    if total <= 0:
        return (0.5, 0.5)
    return (prob_a / total, prob_b / total)


# ---------------------------------------------------------------------------
# API calls (each one costs 1 request from the 500/month budget)
# ---------------------------------------------------------------------------
def fetch_live_odds(
    api_key: str,
    sport_key: str = "esports_lol",
    session: Optional[object] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Fetch current odds for all live/upcoming LoL events.
    Cost: 1 API request.
    Returns (events, quota_info).
    """
    import requests
    session = session or _make_session()

    r = session.get(
        f"{BASE_URL}/sports/{sport_key}/odds",
        params={
            "apiKey": api_key,
            "regions": "us,eu",
            "markets": "h2h",
            "oddsFormat": ODDS_FORMAT,
            "bookmakers": BOOKMAKERS,
        },
        timeout=15,
    )

    quota = {
        "requests_remaining": r.headers.get("x-requests-remaining", "?"),
        "requests_used": r.headers.get("x-requests-used", "?"),
    }

    if r.status_code == 401:
        logger.error("Invalid ODDS_API_KEY — check your .env")
        return [], quota
    if r.status_code == 422:
        logger.warning(f"Sport '{sport_key}' not available or no events")
        return [], quota

    r.raise_for_status()
    events = r.json()
    logger.info(
        f"Fetched {len(events)} events for {sport_key} "
        f"(quota: {quota['requests_remaining']} remaining)"
    )
    return events, quota


def fetch_scores(
    api_key: str,
    sport_key: str = "esports_lol",
    days_from: int = 3,
    session: Optional[object] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Fetch recently completed event scores (includes winner).
    Cost: 1 API request.
    daysFrom=3 gets last 3 days of results.
    """
    import requests
    session = session or _make_session()

    r = session.get(
        f"{BASE_URL}/sports/{sport_key}/scores",
        params={
            "apiKey": api_key,
            "daysFrom": str(days_from),
        },
        timeout=15,
    )

    quota = {
        "requests_remaining": r.headers.get("x-requests-remaining", "?"),
        "requests_used": r.headers.get("x-requests-used", "?"),
    }

    if r.status_code == 401:
        logger.error("Invalid ODDS_API_KEY")
        return [], quota
    if r.status_code == 422:
        return [], quota

    r.raise_for_status()
    scores = r.json()
    logger.info(
        f"Fetched {len(scores)} scores for {sport_key} "
        f"(quota: {quota['requests_remaining']} remaining)"
    )
    return scores, quota


def fetch_event_odds(
    api_key: str,
    sport_key: str,
    event_id: str,
    session: Optional[object] = None,
) -> Tuple[Optional[Dict], Dict]:
    """
    Fetch odds for a specific event by ID.
    Cost: 1 API request. Use sparingly.
    """
    import requests
    session = session or _make_session()

    r = session.get(
        f"{BASE_URL}/sports/{sport_key}/events/{event_id}/odds",
        params={
            "apiKey": api_key,
            "regions": "us,eu",
            "markets": "h2h",
            "oddsFormat": ODDS_FORMAT,
            "bookmakers": BOOKMAKERS,
        },
        timeout=15,
    )

    quota = {
        "requests_remaining": r.headers.get("x-requests-remaining", "?"),
        "requests_used": r.headers.get("x-requests-used", "?"),
    }

    if r.status_code != 200:
        return None, quota

    return r.json(), quota


def check_quota(api_key: str) -> Dict:
    """Check remaining API quota without fetching data. Cost: 0 (uses sports endpoint)."""
    import requests
    r = requests.get(
        f"{BASE_URL}/sports",
        params={"apiKey": api_key},
        timeout=10,
    )
    return {
        "requests_remaining": r.headers.get("x-requests-remaining", "?"),
        "requests_used": r.headers.get("x-requests-used", "?"),
    }


def discover_esports_keys(api_key: str) -> List[Dict]:
    """List all sports, highlighting any esports/gaming categories. Cost: 0."""
    import requests
    r = requests.get(
        f"{BASE_URL}/sports",
        params={"apiKey": api_key, "all": "true"},
        timeout=10,
    )
    r.raise_for_status()
    sports = r.json()
    esports = []
    for s in sports:
        group = s.get("group", "").lower()
        key = s.get("key", "").lower()
        title = s.get("title", "").lower()
        if any(kw in group or kw in key or kw in title
               for kw in ["esport", "gaming", "lol", "league_of_legends",
                           "counter_strike", "dota", "valorant"]):
            esports.append(s)
    return esports


# ---------------------------------------------------------------------------
# Parse API response into storable rows
# ---------------------------------------------------------------------------
def parse_events_to_odds(events: List[Dict], source: str = "the-odds-api") -> List[Dict]:
    """Convert API event objects to flat odds rows for storage."""
    rows = []

    for event in events:
        team_a = event.get("home_team", "")
        team_b = event.get("away_team", "")
        commence = event.get("commence_time", "")
        sport = event.get("sport_key", "")
        event_id = event.get("id", "")
        completed = event.get("completed", False)

        # Get winner from scores if available
        winner = None
        scores_data = event.get("scores")
        if scores_data and completed:
            for sc in scores_data:
                if sc.get("name") == team_a:
                    score_a = int(sc.get("score", 0))
                elif sc.get("name") == team_b:
                    score_b = int(sc.get("score", 0))
            try:
                if score_a > score_b:
                    winner = team_a
                elif score_b > score_a:
                    winner = team_b
            except NameError:
                pass

        # Extract odds from bookmakers
        odds_a = None
        odds_b = None
        bookmaker_name = None

        for bm in event.get("bookmakers", []):
            bookmaker_name = bm.get("key", "")
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = market.get("outcomes", [])
                for oc in outcomes:
                    if oc.get("name") == team_a:
                        odds_a = oc.get("price")
                    elif oc.get("name") == team_b:
                        odds_b = oc.get("price")
            if odds_a and odds_b:
                break

        # Parse date
        match_date = ""
        if commence:
            try:
                match_date = commence[:10]
            except (IndexError, TypeError):
                pass

        rows.append({
            "event_id": event_id,
            "sport_key": sport,
            "date": match_date,
            "team_a": team_a,
            "team_b": team_b,
            "odds_a": odds_a,
            "odds_b": odds_b,
            "winner": winner,
            "completed": completed,
            "bookmaker": bookmaker_name or BOOKMAKERS,
            "source": source,
        })

    return rows


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def store_odds(odds_rows: List[Dict], league: str = "esports_lol") -> int:
    """Match team names, compute probabilities, and store."""
    db_teams = load_db_team_names()
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    stored = 0
    unmatched = []

    for row in odds_rows:
        if not row.get("odds_a") or not row.get("odds_b"):
            continue

        team_a_raw = row["team_a"]
        team_b_raw = row["team_b"]

        team_a_db = match_team_name(team_a_raw, db_teams, source="the-odds-api")
        team_b_db = match_team_name(team_b_raw, db_teams, source="the-odds-api")

        if not team_a_db:
            unmatched.append(team_a_raw)
        if not team_b_db:
            unmatched.append(team_b_raw)

        implied_a = decimal_to_implied(row["odds_a"])
        implied_b = decimal_to_implied(row["odds_b"])
        no_vig_a, no_vig_b = remove_vig(implied_a, implied_b)

        winner_raw = row.get("winner")
        winner_db = None
        if winner_raw == team_a_raw:
            winner_db = team_a_db
        elif winner_raw == team_b_raw:
            winner_db = team_b_db

        source = f"{row.get('source', 'the-odds-api')}:{row.get('bookmaker', 'pinnacle')}"

        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO bookmaker_odds
                    (source, match_date, league, team_a_raw, team_b_raw,
                     team_a_db, team_b_db, odds_a, odds_b,
                     implied_prob_a, implied_prob_b, no_vig_prob_a, no_vig_prob_b,
                     winner_raw, winner_db, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source, row["date"], league, team_a_raw, team_b_raw,
                    team_a_db, team_b_db, row["odds_a"], row["odds_b"],
                    round(implied_a, 4), round(implied_b, 4),
                    round(no_vig_a, 4), round(no_vig_b, 4),
                    winner_raw, winner_db, now,
                ),
            )
            stored += 1
        except sqlite3.Error as e:
            logger.debug(f"Insert error: {e}")

    conn.commit()
    conn.close()

    if unmatched:
        unique_unmatched = sorted(set(unmatched))
        export_unmatched(unique_unmatched, "the-odds-api")
        logger.warning(f"  {len(unique_unmatched)} unique team names unmatched")

    logger.info(f"Stored {stored}/{len(odds_rows)} odds rows")
    return stored


# ---------------------------------------------------------------------------
# Match linking
# ---------------------------------------------------------------------------
def link_odds_to_matches() -> int:
    """Link bookmaker_odds rows to matches table by team names + date."""
    conn = sqlite3.connect(DB_PATH)
    unlinked = conn.execute(
        """
        SELECT id, match_date, team_a_db, team_b_db
        FROM bookmaker_odds
        WHERE match_id IS NULL AND team_a_db IS NOT NULL AND team_b_db IS NOT NULL
        """
    ).fetchall()

    if not unlinked:
        logger.info("No unlinked odds rows to process")
        conn.close()
        return 0

    linked = 0
    for odds_id, match_date, team_a, team_b in unlinked:
        if not match_date:
            continue

        row = conn.execute(
            """
            SELECT id FROM matches
            WHERE ((blue_team = ? AND red_team = ?) OR (blue_team = ? AND red_team = ?))
              AND date BETWEEN date(?, '-1 day') AND date(?, '+1 day')
            ORDER BY ABS(julianday(date) - julianday(?))
            LIMIT 1
            """,
            (team_a, team_b, team_b, team_a, match_date, match_date, match_date),
        ).fetchone()

        if row:
            conn.execute(
                "UPDATE bookmaker_odds SET match_id = ? WHERE id = ?",
                (row[0], odds_id),
            )
            linked += 1

    conn.commit()
    conn.close()
    logger.info(f"Linked {linked}/{len(unlinked)} odds rows to matches")
    return linked


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def print_stats() -> None:
    conn = sqlite3.connect(DB_PATH)

    total = conn.execute("SELECT COUNT(*) FROM bookmaker_odds").fetchone()[0]
    linked = conn.execute("SELECT COUNT(*) FROM bookmaker_odds WHERE match_id IS NOT NULL").fetchone()[0]
    matched = conn.execute(
        "SELECT COUNT(*) FROM bookmaker_odds WHERE team_a_db IS NOT NULL AND team_b_db IS NOT NULL"
    ).fetchone()[0]
    with_winner = conn.execute(
        "SELECT COUNT(*) FROM bookmaker_odds WHERE winner_raw IS NOT NULL"
    ).fetchone()[0]

    sources = conn.execute(
        "SELECT source, COUNT(*) FROM bookmaker_odds GROUP BY source ORDER BY COUNT(*) DESC"
    ).fetchall()

    dates = conn.execute(
        "SELECT MIN(match_date), MAX(match_date) FROM bookmaker_odds WHERE match_date != ''"
    ).fetchone()

    conn.close()

    print(f"\n{'='*50}")
    print(f"  BOOKMAKER ODDS COVERAGE")
    print(f"{'='*50}")
    print(f"  Total odds rows:    {total}")
    if total > 0:
        print(f"  Teams matched:      {matched} ({matched/total:.0%})")
        print(f"  With winner:        {with_winner} ({with_winner/total:.0%})")
        print(f"  Linked to matches:  {linked} ({linked/total:.0%})")
        if dates[0]:
            print(f"  Date range:         {dates[0]} → {dates[1]}")
    else:
        print(f"  (no data yet — run --live or --scores)")

    if sources:
        print(f"\n  By source:")
        for source, count in sources:
            print(f"    {source:30} {count:6}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Bookmaker odds scraper (the-odds-api.com)")
    parser.add_argument("--live", action="store_true", help="Fetch current LoL odds (1 API call)")
    parser.add_argument("--scores", action="store_true", help="Fetch recent results + odds (1 API call)")
    parser.add_argument("--days", type=int, default=3, help="Days of scores to fetch (default: 3)")
    parser.add_argument("--link", action="store_true", help="Link odds to matches table")
    parser.add_argument("--stats", action="store_true", help="Show coverage stats")
    parser.add_argument("--quota", action="store_true", help="Check remaining API quota (0 calls)")
    parser.add_argument("--discover", action="store_true", help="Check if esports have been added (0 calls)")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if args.link:
        link_odds_to_matches()
        return

    api_key = _get_api_key()
    if not api_key:
        return

    if args.quota:
        q = check_quota(api_key)
        print(f"\n  API Quota: {q['requests_remaining']} remaining, {q['requests_used']} used this month\n")
        return

    if args.discover:
        esports = discover_esports_keys(api_key)
        if esports:
            print(f"\n  Esports keys found ({len(esports)}):")
            for s in esports:
                active = "ACTIVE" if s.get("active") else "inactive"
                print(f"    {s['key']:40} {s.get('title',''):30} {active}")
            print(f"\n  Update LOL_SPORT_KEYS in odds_scraper.py with the correct key.")
        else:
            print(f"\n  No esports sport keys found on the-odds-api.com yet.")
            print(f"  They currently cover: traditional sports only (soccer, basketball, etc.)")
            print(f"  Run --discover periodically to check if they add esports coverage.")
        print()
        return

    if args.live:
        for sport in LOL_SPORT_KEYS:
            events, quota = fetch_live_odds(api_key, sport_key=sport)
            if events:
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                raw_path = RAW_DIR / f"live_{sport}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                raw_path.write_text(json.dumps(events, indent=2))
                logger.info(f"Raw data saved → {raw_path}")
                rows = parse_events_to_odds(events)
                store_odds(rows, league=sport)
            else:
                print(f"  No events for '{sport}'. Run --discover to check available sport keys.")
            print(f"  Quota remaining: {quota['requests_remaining']}")
        return

    if args.scores:
        for sport in LOL_SPORT_KEYS:
            scores, quota = fetch_scores(api_key, sport_key=sport, days_from=args.days)
            if scores:
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                raw_path = RAW_DIR / f"scores_{sport}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                raw_path.write_text(json.dumps(scores, indent=2))
                logger.info(f"Raw data saved → {raw_path}")
                rows = parse_events_to_odds(scores)
                store_odds(rows, league=sport)
                link_odds_to_matches()
            else:
                print(f"  No scores for '{sport}'. Run --discover to check available sport keys.")
            print(f"  Quota remaining: {quota['requests_remaining']}")
        return

    parser.print_help()
    print(f"\nBudget tips (500 calls/month free):")
    print(f"  --discover = 0 calls  (check if esports keys exist)")
    print(f"  --quota    = 0 calls  (check remaining budget)")
    print(f"  --live     = 1 call   (all current LoL odds)")
    print(f"  --scores   = 1 call   (recent results + winners)")
    print(f"  --stats    = 0 calls  (show what's in DB)")
    print()
    print(f"NOTE: As of June 2026, the-odds-api does NOT cover esports.")
    print(f"The scraper infrastructure is ready — run --discover to check for updates.")


if __name__ == "__main__":
    main()
