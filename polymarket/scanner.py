"""
Polymarket scanner — discovers active LoL T2 markets and fetches prices.

Queries the Polymarket Gamma API (public, no auth) for events whose titles
or descriptions mention League of Legends or T2 team names, then extracts
match information and current market prices.

Run standalone:  python polymarket/scanner.py
"""

import json
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

DB_PATH = _ROOT / "db" / "lol_model.db"

# ---------------------------------------------------------------------------
# Polymarket API
# ---------------------------------------------------------------------------
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
PAGE_SIZE = 100

# Keywords for discovering LoL markets (case-insensitive)
LOL_KEYWORDS = [
    "league of legends",
    "lol esports",
    "lol:",
    "lol ",
    "nacl ",
    "lck challengers",
    "lck cl",
    "emea masters",
    "lfl ",
    "nlc ",
    "ljl ",
    "pcs ",
    "vcs ",
    "tcl ",
    "msi 20",
    "mid-season invitational",
    "worlds 20",
    "world championship",
    "esports world cup",
    "lpl ",
    "lck ",
    "lec ",
    "lcp ",
    "lcs ",
]

# Regex patterns for extracting two team names from market titles
# Handles: "Will X beat Y?", "X vs Y", "X v Y", "Who will win X vs Y?"
_VS_RE = re.compile(
    r"(?:will\s+)?(.+?)\s+(?:vs\.?|v\.?|beat|defeat)\s+(.+?)[\?\.]?\s*$",
    re.IGNORECASE,
)
_WHO_WINS_RE = re.compile(
    r"who\s+will\s+win\s+(.+?)\s+(?:vs\.?|v\.?)\s+(.+?)[\?\.]?\s*$",
    re.IGNORECASE,
)

# Fuzzy match cutoff for team name matching
TEAM_MATCH_CUTOFF = 0.80

# Manual team name aliases: Polymarket name → DB name
# Add entries as you discover mismatches
TEAM_ALIASES: Dict[str, str] = {
    # "PM Name": "DB Name",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class MarketOpportunity:
    market_id: str
    condition_id: str
    slug: str
    question: str
    team_a: str
    team_b: str
    db_team_a: str
    db_team_b: str
    token_id_a: str
    token_id_b: str
    market_prob_a: float
    market_prob_b: float
    spread: float
    volume: float
    url: str


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------
def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


# ---------------------------------------------------------------------------
# Team name database
# ---------------------------------------------------------------------------
def load_db_team_names() -> List[str]:
    """Load all team names from the teams table."""
    conn = sqlite3.connect(DB_PATH)
    names = [row[0] for row in conn.execute("SELECT team_name FROM teams").fetchall()]
    conn.close()
    return names


def match_team_name(pm_name: str, db_teams: List[str]) -> Optional[str]:
    """
    Match a Polymarket team name to a DB team name.
    Checks aliases first, then fuzzy match.
    """
    cleaned = pm_name.strip()
    if cleaned in TEAM_ALIASES:
        return TEAM_ALIASES[cleaned]

    matches = get_close_matches(cleaned, db_teams, n=1, cutoff=TEAM_MATCH_CUTOFF)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Market discovery
# ---------------------------------------------------------------------------
# Team names too short or generic to search for in market text
_SKIP_TEAM_SEARCH = {
    "g2 esports", "rogue", "reject", "fuego", "misa esports",
    "dragons", "lions", "wolves", "team heretics academy",
    "riders", "one", "on", "hands", "game",
}
MIN_TEAM_NAME_LEN = 5


def _is_lol_market(text: str, db_teams: List[str]) -> bool:
    """Check if text explicitly mentions LoL or known T2 teams."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in LOL_KEYWORDS):
        return True
    for team in db_teams:
        team_lower = team.lower()
        if len(team_lower) < MIN_TEAM_NAME_LEN:
            continue
        if team_lower in _SKIP_TEAM_SEARCH:
            continue
        if re.search(r"\b" + re.escape(team_lower) + r"\b", text_lower):
            return True
    return False


def _clean_team_name(name: str) -> str:
    """Strip common prefixes/suffixes from parsed team names."""
    s = name.strip().rstrip("?.")
    # Remove "LoL: " or "LoL:" prefix
    s = re.sub(r"^(?:LoL|LOL|lol)\s*:\s*", "", s)
    # Remove "(BON)" series indicator and everything after " - "
    s = re.sub(r"\s*\(BO\d+\).*$", "", s)
    s = re.sub(r"\s*-\s*Game\s+\d+.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*-\s+(?:Mid-Season|Esports|LCK|LEC|LPL|LCS|MSI|Worlds).*$", "", s, flags=re.IGNORECASE)
    return s.strip()


def parse_teams_from_question(question: str) -> Optional[Tuple[str, str]]:
    """Extract two team names from a market question string."""
    for pattern in [_WHO_WINS_RE, _VS_RE]:
        m = pattern.search(question)
        if m:
            a = _clean_team_name(m.group(1))
            b = _clean_team_name(m.group(2))
            if a and b:
                return (a, b)
    return None


MAX_PAGES = 20  # safety cap — 2000 events is more than enough

LOL_TAG_SLUGS = ["league-of-legends", "esports"]


def fetch_active_events(session: requests.Session) -> List[dict]:
    """Fetch active LoL events from the Gamma API using targeted tag queries."""
    all_events: List[dict] = []
    seen_ids: set = set()

    # Targeted search by tag_slug — finds ALL LoL markets regardless of pagination
    for tag in LOL_TAG_SLUGS:
        offset = 0
        for _ in range(5):
            try:
                r = session.get(
                    f"{GAMMA_API}/events",
                    params={
                        "active": "true",
                        "closed": "false",
                        "tag_slug": tag,
                        "limit": str(PAGE_SIZE),
                        "offset": str(offset),
                    },
                    timeout=15,
                )
                if r.status_code != 200:
                    break
                batch = r.json()
            except requests.RequestException as e:
                logger.warning(f"Gamma API error for tag {tag}: {e}")
                break
            for event in batch:
                eid = event.get("id")
                if eid and eid not in seen_ids:
                    all_events.append(event)
                    seen_ids.add(eid)
            if len(batch) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

    return all_events


def scan(session: Optional[requests.Session] = None) -> List[MarketOpportunity]:
    """
    Full scan pipeline: fetch events → filter LoL → parse teams → match to DB.
    Returns list of MarketOpportunity for markets we can price.
    """
    session = session or _make_session()
    db_teams = load_db_team_names()

    if not db_teams:
        logger.error("No teams in DB — run pro_elo.py first")
        return []

    logger.info(f"Scanning Polymarket for LoL T2 markets ({len(db_teams)} teams in DB)…")

    events = fetch_active_events(session)
    logger.info(f"  Fetched {len(events)} active events from Gamma API")

    opportunities: List[MarketOpportunity] = []

    for event in events:
        title = event.get("title", "")
        slug = event.get("slug", "")

        # Skip non-LoL esports (CS, Dota, Valorant)
        if any(kw in title.lower() for kw in ["counter-strike", "dota", "valorant", "cs2", "csgo"]):
            continue

        if not _is_lol_market(title, db_teams):
            continue

        logger.info(f"  Found potential LoL event: {title}")

        # Ensure markets are populated — the events endpoint sometimes omits them
        markets = event.get("markets", [])
        if not markets and slug:
            try:
                r = session.get(
                    f"{GAMMA_API}/events",
                    params={"slug": slug, "limit": "1"},
                    timeout=10,
                )
                if r.status_code == 200:
                    full = r.json()
                    if full:
                        markets = full[0].get("markets", [])
            except requests.RequestException:
                pass

        for market in markets:
            question = market.get("question", "")
            teams = parse_teams_from_question(question)
            if not teams:
                continue

            pm_a, pm_b = teams
            db_a = match_team_name(pm_a, db_teams)
            db_b = match_team_name(pm_b, db_teams)

            if not db_a or not db_b:
                unmatched = []
                if not db_a:
                    unmatched.append(pm_a)
                if not db_b:
                    unmatched.append(pm_b)
                logger.warning(f"    Could not match teams: {unmatched}")
                continue

            # Parse prices — API sometimes returns these as JSON strings
            prices = market.get("outcomePrices", [])
            tokens = market.get("clobTokenIds", [])
            outcomes = market.get("outcomes", [])

            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except (json.JSONDecodeError, TypeError):
                    continue
            if isinstance(tokens, str):
                try:
                    tokens = json.loads(tokens)
                except (json.JSONDecodeError, TypeError):
                    continue

            if len(prices) < 2 or len(tokens) < 2:
                continue

            try:
                price_a = float(prices[0])
                price_b = float(prices[1])
            except (ValueError, TypeError):
                continue

            spread = float(market.get("spread", 0))
            volume = float(market.get("volumeNum", 0) or market.get("volume", 0) or 0)

            opp = MarketOpportunity(
                market_id=market.get("id", ""),
                condition_id=market.get("conditionId", ""),
                slug=slug,
                question=question,
                team_a=pm_a,
                team_b=pm_b,
                db_team_a=db_a,
                db_team_b=db_b,
                token_id_a=tokens[0],
                token_id_b=tokens[1],
                market_prob_a=price_a,
                market_prob_b=price_b,
                spread=spread,
                volume=volume,
                url=f"https://polymarket.com/event/{slug}",
            )
            opportunities.append(opp)
            logger.info(
                f"    Matched: {db_a} ({price_a:.0%}) vs {db_b} ({price_b:.0%})"
            )

    logger.info(f"Scan complete: {len(opportunities)} tradeable markets found")
    return opportunities


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def main() -> None:
    opportunities = scan()

    if not opportunities:
        logger.info("No LoL T2 markets currently active on Polymarket.")
        logger.info(
            "The scanner will keep checking — T2 markets appear sporadically, "
            "typically around match days."
        )
        return

    print(f"\n{'='*60}")
    print(f"  ACTIVE LOL T2 MARKETS ({len(opportunities)})")
    print(f"{'='*60}")
    for opp in opportunities:
        print(f"\n  {opp.question}")
        print(f"  {opp.db_team_a:20} {opp.market_prob_a:6.1%}")
        print(f"  {opp.db_team_b:20} {opp.market_prob_b:6.1%}")
        print(f"  Spread: ${opp.spread:.3f}  |  Volume: ${opp.volume:,.0f}")
        print(f"  {opp.url}")
    print()


if __name__ == "__main__":
    main()
