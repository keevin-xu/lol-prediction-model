"""
Leaguepedia Cargo API roster scraper.

Fetches current team rosters for T2 LoL leagues, fuzzy-matches player
names against TTP data, saves a raw snapshot, and upserts to the
rosters SQLite table.

Run:  python scrapers/roster_scraper.py
"""

import json
import os
import re
import sqlite3
import time
import unicodedata
from datetime import date
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
DB_PATH = _ROOT / "db" / "lol_model.db"
RAW_DIR = _ROOT / "data" / "raw" / "rosters"
PROCESSED_DIR = _ROOT / "data" / "processed"
UNMATCHED_PATH = PROCESSED_DIR / "unmatched_players.json"

# Load credentials / config from .env (no-op if the file is absent)
load_dotenv(_ROOT / ".env")

# ---------------------------------------------------------------------------
# Leaguepedia Cargo API
# ---------------------------------------------------------------------------
LEAGUEPEDIA_API = "https://lol.fandom.com/api.php"
# Contact email in the UA is overridable via LEAGUEPEDIA_USER_AGENT in .env so
# each dev advertises their own contact.
_DEFAULT_UA = "lol-prediction-model/1.0 (kevrocksxd@gmail.com)"
LEAGUEPEDIA_HEADERS = {
    "User-Agent": os.environ.get("LEAGUEPEDIA_USER_AGENT", "").strip() or _DEFAULT_UA,
    "Accept": "application/json",
}
# Seconds between API calls; Leaguepedia rate-limits aggressive scrapers
API_DELAY = 2.0
CURRENT_YEAR = "2026"

# ---------------------------------------------------------------------------
# League name mapping: Leaguepedia full name → OE abbreviation
#
# Confirmed via Cargo API probe on 2026-06-17.
# Uncertain entries are commented out — add when Leaguepedia names verified.
# ---------------------------------------------------------------------------
LEAGUE_MAP: Dict[str, str] = {
    # Confirmed current T2 roster sources (Leaguepedia League name → OE label).
    # Verified against the full 2026 Secondary tournament list on 2026-06-17.
    "North American Challengers League": "NACL",
    "LCK Challengers League": "LCKC",
    "EMEA Masters": "EM",
    "Northern League of Legends Championship": "NLC",
    "La Ligue Française": "LFL",
    "LVP SuperLiga": "LVP SL",                  # see note ‡ below
    "Turkish Championship League": "TCL",
    "Pacific Championship Series": "PCS",
    "Vietnam Championship Series": "VCS",
    "LoL Japan League": "LJL",
    "Liga Regional Norte": "LRN",
    "Liga Regional Sur": "LRS",
    "The Oceanic Resurrection": "LCO",          # OCE successor to the LCO; sole current OCE source
}

# OE labels deliberately NOT mapped to a roster source.
# These appear in the matches table (so teams still earn pro-ELO from them), but
# have no current *Secondary*-level Leaguepedia tournament to pull live rosters
# from — confirmed absent from the full 2026 Secondary tournament list:
#   LEC    → T1 EMEA league (Primary level); its T2 feeders are EM / LFL / NLC / LVP SL.
#   LTA N  → T1 (2025 Americas rebrand, Primary); T2 feeder is LRN (mapped).
#   LTA S  → T1 (2025 Americas rebrand, Primary); T2 feeder is LRS (mapped).
#   LLA    → former top LATAM league, folded into LTA South (now T1); no Secondary split.
#   ESLOL  → Italian league; no 2026 tournament on Leaguepedia (defunct/renamed).
# Teams in these leagues get pro-ELO but no soloq-blended roster; the alpha blend
# (alpha = games/(games+10)) handles that via pro-ELO fallback.
#
# ‡ LVP SL: the map key is correct and DOES match — this is NOT a scraper bug.
#   As of 2026-06-17 Leaguepedia has no 2026 main split for LVP SuperLiga: its
#   latest real competitive split is "SL 2025 Summer Split" (2025-07), and the only
#   2026-labelled event is "SL 2026 Promotion" — a qualifier (IsQualifier=1) that
#   discover_t2_tournaments correctly skips. So there is simply no current roster to
#   pull; LVP will populate automatically once Leaguepedia posts a 2026 split.
#   Do NOT relax the qualifier/"promotion" filters to force it — that would leak
#   promotion/relegation rosters into every other league.

# Tournament name substrings that mark non-main-split events (case-insensitive)
_EXCLUDE_SUBSTRINGS = [
    "qualifier",
    "promotion",
    "relegation",
    "tiebreaker",
    "pre-season",
    "trial",
    "open qualifier",
    "prequal",
    "road to",
]

# Player role strings that indicate coaching staff (not players)
_STAFF_ROLES = {"coach", "manager", "analyst", "head coach", "assistant coach"}

# Length-aware fuzzy-match thresholds.
# False positives cluster entirely on SHORT handles: a one-character slip like
# 'Cid'→'Clid', 'Han'→'Khan', 'Eria'→'Keria' scores ~0.86–0.89 yet maps a T2
# player onto a famous pro — actively corrupting the blend (which leans hardest
# on soloq for low-pro-game T2 teams). The shorter a name, the fewer
# distinguishing characters, so we demand more similarity as it shrinks.
# Empirically (2026-06-17 snapshot): every wrong match was ≤5 chars, every right
# fuzzy match was ≥6 chars. Thresholds keyed on the SHORTER normalized name:
FUZZY_CUTOFF = 0.85  # absolute floor (also the threshold for long names ≥7 chars)
_FUZZY_LEN_TIERS = (
    (4, 1.00),   # ≤4 chars: must match exactly after normalization
    (6, 0.90),   # 5–6 chars: tighter
    # else (≥7 chars): FUZZY_CUTOFF
)


def _required_ratio(a: str, b: str) -> float:
    """Minimum similarity to accept a match, scaled by the shorter name length."""
    n = min(len(a), len(b))
    for max_len, ratio in _FUZZY_LEN_TIERS:
        if n <= max_len:
            return ratio
    return FUZZY_CUTOFF

# Regex to strip wiki disambiguation: "Spawn (Trevor Kerr-Taylor)" → "Spawn"
_PAREN_RE = re.compile(r"\s*\(.*?\)\s*$")


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------
def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        # Cargo reads are sent via POST (see _cargo_query); without this,
        # urllib3 would not retry POST on 5xx since it treats POST as non-idempotent.
        allowed_methods=frozenset(["GET", "POST"]),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


# ---------------------------------------------------------------------------
# Authentication (optional — raises the rate-limit ceiling)
# ---------------------------------------------------------------------------
def login(session: requests.Session) -> bool:
    """
    Authenticate the session with a Leaguepedia bot password if credentials are
    present in the environment. Returns True on success, False if no credentials
    are configured (the caller then proceeds anonymously, exactly as before).

    Standard MediaWiki action=login flow:
      1. POST action=query&meta=tokens&type=login  → login token
      2. POST action=login with lgname / lgpassword / lgtoken
    Session cookies carry the authenticated state into later Cargo requests.
    """
    username = os.environ.get("LEAGUEPEDIA_USERNAME", "").strip()
    password = os.environ.get("LEAGUEPEDIA_BOT_PASSWORD", "").strip()
    if not username or not password:
        logger.warning(
            "No LEAGUEPEDIA_USERNAME / LEAGUEPEDIA_BOT_PASSWORD in env — "
            "running anonymously (lower rate-limit ceiling)."
        )
        return False

    # 1. Fetch a login token
    r = session.post(
        LEAGUEPEDIA_API,
        data={"action": "query", "meta": "tokens", "type": "login", "format": "json"},
        headers=LEAGUEPEDIA_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    token = r.json()["query"]["tokens"]["logintoken"]

    # 2. Submit bot-password credentials
    r = session.post(
        LEAGUEPEDIA_API,
        data={
            "action": "login",
            "lgname": username,
            "lgpassword": password,
            "lgtoken": token,
            "format": "json",
        },
        headers=LEAGUEPEDIA_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    result = r.json().get("login", {})
    if result.get("result") == "Success":
        logger.info(
            f"Authenticated to Leaguepedia as {result.get('lgusername', username)} "
            "(higher rate-limit ceiling)"
        )
        return True

    raise RuntimeError(
        f"Leaguepedia login failed: {result.get('result')} — {result.get('reason', '')}"
    )


# ---------------------------------------------------------------------------
# Cargo API wrapper
# ---------------------------------------------------------------------------
def _cargo_query(
    session: requests.Session,
    params: Dict[str, str],
    retries: int = 6,
) -> List[Dict[str, Any]]:
    """
    POST a cargoquery request and return the list of title-dicts.
    Automatically backs off on rate-limit responses.
    """
    base = {"action": "cargoquery", "format": "json"}
    full_params = {**base, **params}

    for attempt in range(retries):
        time.sleep(API_DELAY)
        # POST (not GET): keeps long batched `OverviewPage IN (...)` queries in the
        # request body so they can't trip URL-length limits or proxy/WAF rules.
        # MediaWiki returns identical JSON for GET vs POST cargoquery.
        r = session.post(
            LEAGUEPEDIA_API,
            data=full_params,
            headers=LEAGUEPEDIA_HEADERS,
            timeout=30,
        )
        data = r.json()
        if "error" in data:
            code = data["error"]["code"]
            if code == "ratelimited":
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limited — sleeping {wait}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Leaguepedia API error [{code}]: {data['error'].get('info', '')}"
            )
        return [row.get("title", {}) for row in data.get("cargoquery", [])]

    raise RuntimeError("Leaguepedia API: max retries exceeded (still rate-limited)")


def _cargo_query_all(
    session: requests.Session,
    params: Dict[str, str],
    page_size: int = 500,
) -> List[Dict[str, Any]]:
    """Paginate through a Cargo query and return all rows."""
    all_rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        batch = _cargo_query(session, {**params, "limit": str(page_size), "offset": str(offset)})
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return all_rows


# ---------------------------------------------------------------------------
# Tournament discovery
# ---------------------------------------------------------------------------
def _is_excluded(name: str) -> bool:
    """True if the tournament name suggests a qualifier/promotion event."""
    low = name.lower()
    return any(sub in low for sub in _EXCLUDE_SUBSTRINGS)


def discover_t2_tournaments(session: requests.Session) -> List[Dict[str, Any]]:
    """
    Return a list of main-split T2 tournaments for 2026 across all mapped leagues.
    Each entry: {overview_page, league, oe_league, name, date_start, is_playoffs}.
    """
    leaguepedia_names = list(LEAGUE_MAP.keys())
    # Build SQL IN list — single quotes escaped; Leaguepedia names have no quotes
    in_list = "', '".join(leaguepedia_names)

    rows = _cargo_query_all(session, {
        "tables": "Tournaments",
        "fields": "Name,OverviewPage,League,DateStart,IsQualifier,IsPlayoffs",
        "where": (
            f"TournamentLevel='Secondary'"
            f" AND Year='{CURRENT_YEAR}'"
            f" AND League IN ('{in_list}')"
        ),
        "order_by": "DateStart DESC",
    })

    tournaments: List[Dict[str, Any]] = []
    for row in rows:
        name = row.get("Name", "")
        if row.get("IsQualifier") == "1":
            continue
        if _is_excluded(name):
            continue
        league_full = row.get("League", "")
        tournaments.append({
            "overview_page": row.get("OverviewPage", ""),
            "league": league_full,
            "oe_league": LEAGUE_MAP.get(league_full, ""),
            "name": name,
            "date_start": row.get("DateStart", ""),
            "is_playoffs": row.get("IsPlayoffs") == "1",
        })

    logger.info(f"Discovered {len(tournaments)} main-event T2 tournaments for {CURRENT_YEAR}")
    for t in tournaments:
        logger.debug(f"  [{t['oe_league']:8}] {t['date_start'][:10]} — {t['name']}")
    return tournaments


def select_latest_per_league(tournaments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each OE league abbreviation, keep only the most recent tournament
    (by DateStart) that has started on or before today, preferring non-playoff
    entries to avoid duplicating roster pulls.
    """
    today = date.today().isoformat()
    by_league: Dict[str, List[Dict[str, Any]]] = {}
    for t in tournaments:
        league = t["oe_league"]
        if not league:
            continue
        started = t["date_start"][:10] <= today if t["date_start"] else False
        if not started:
            continue
        by_league.setdefault(league, []).append(t)

    selected: List[Dict[str, Any]] = []
    for league, ts in by_league.items():
        # Non-playoffs first; among ties prefer later DateStart
        ts_sorted = sorted(ts, key=lambda x: (x["is_playoffs"], x["date_start"]), reverse=True)
        # Pick most recent non-playoffs; fall back to playoffs if that's all there is
        best = next((t for t in ts_sorted if not t["is_playoffs"]), ts_sorted[0])
        selected.append(best)
        logger.debug(f"  Selected for {league}: {best['name']}")

    return selected


# ---------------------------------------------------------------------------
# Roster parsing
# ---------------------------------------------------------------------------
def _clean_player_name(raw: str) -> str:
    """Strip wiki disambiguation parentheticals and whitespace."""
    name = _PAREN_RE.sub("", raw).strip()
    return name


def fetch_rosters(
    overview_pages: List[str], session: requests.Session
) -> List[Dict[str, Any]]:
    """
    Fetch all TournamentRosters rows for the given OverviewPages in a single
    batched query (OverviewPage IN (...)), paginating only if the combined
    result spans more than one page.

    Each returned row includes its OverviewPage so the caller can map it back
    to the originating tournament. Replaces the previous one-request-per-
    tournament loop to minimise calls against Leaguepedia's rate limit.
    """
    if not overview_pages:
        return []
    # Escape single quotes (unlikely but safe) and build the IN (...) list
    quoted = ", ".join("'" + p.replace("'", "\\'") + "'" for p in overview_pages)
    return _cargo_query_all(session, {
        "tables": "TournamentRosters",
        "fields": "Team,OverviewPage,RosterLinks,Roles",
        "where": f"OverviewPage IN ({quoted})",
    })


def parse_roster_row(row: Dict[str, Any], tournament: str, oe_league: str) -> List[Dict[str, Any]]:
    """
    Convert a single TournamentRosters row into player-entry dicts.
    Skips coaching staff. Returns list of {team, player_name, role, ...}.
    """
    team = row.get("Team", "")
    players_raw = [p.strip() for p in row.get("RosterLinks", "").split(";;")]
    roles_raw = [r.strip() for r in row.get("Roles", "").split(";;")]

    entries: List[Dict[str, Any]] = []
    for i, raw_name in enumerate(players_raw):
        if not raw_name:
            continue
        role_str = roles_raw[i] if i < len(roles_raw) else ""
        # Roles can be comma-separated when a player plays multiple positions
        primary_role = role_str.split(",")[0].strip()
        if primary_role.lower() in _STAFF_ROLES:
            continue
        player_name = _clean_player_name(raw_name)
        if not player_name:
            continue
        entries.append({
            "team": team,
            "player_name": player_name,
            "role": primary_role,
            "tournament": tournament,
            "oe_league": oe_league,
        })
    return entries


# ---------------------------------------------------------------------------
# TTP fuzzy matching
# ---------------------------------------------------------------------------
def load_ttp_players(conn: sqlite3.Connection) -> List[str]:
    """Return all player names currently in the players table."""
    cursor = conn.execute("SELECT player_name FROM players")
    return [row[0] for row in cursor.fetchall()]


def _norm(s: str) -> str:
    """
    Normalize a name for fuzzy matching: NFKD-decompose, drop combining accent
    marks, and lowercase. Lets 'Odi11'↔'odi11' and 'Adryh'↔'Àdryh' match while
    keeping FUZZY_CUTOFF strict (we never lower the threshold, just remove
    case/diacritic noise before comparing).
    """
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def fuzzy_match(
    name: str,
    candidates: List[str],
    cutoff: float = FUZZY_CUTOFF,
    norm_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Fuzzy-match `name` against `candidates`, ignoring case and accents. Matching
    runs on normalized forms but the ORIGINAL TTP candidate string is returned.
    `norm_map` (normalized → original) may be precomputed by the caller to avoid
    rebuilding it for every name.

    Two-stage: pull the best candidate at the absolute floor (`cutoff`), then
    accept it only if its similarity clears the length-aware bar from
    `_required_ratio` — so short handles must match (near-)exactly while longer
    names keep the looser tolerance.
    """
    if norm_map is None:
        norm_map = {_norm(c): c for c in candidates}
    nname = _norm(name)
    hits = get_close_matches(nname, list(norm_map), n=1, cutoff=cutoff)
    if not hits:
        return None
    nhit = hits[0]
    if SequenceMatcher(None, nname, nhit).ratio() >= _required_ratio(nname, nhit):
        return norm_map[nhit]
    return None


def match_players(
    entries: List[Dict[str, Any]],
    ttp_players: List[str],
) -> tuple:
    """
    Fuzzy-match each entry's player_name against TTP player list.
    Returns (matched_entries, unmatched_entries).
    Each matched entry gains a 'ttp_player' key.
    """
    matched: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []

    # Build the normalized → original map once, not per-entry.
    norm_map = {_norm(c): c for c in ttp_players}
    for entry in entries:
        hit = fuzzy_match(entry["player_name"], ttp_players, norm_map=norm_map)
        if hit:
            entry = {**entry, "ttp_player": hit}
            matched.append(entry)
        else:
            unmatched.append(entry)

    return matched, unmatched


# ---------------------------------------------------------------------------
# SQLite upsert
# ---------------------------------------------------------------------------
def upsert_rosters(
    entries: List[Dict[str, Any]],
    snapshot_date: str,
    conn: sqlite3.Connection,
) -> int:
    """
    Delete today's roster snapshot and re-insert all entries.
    Returns count of inserted rows.
    """
    conn.execute("DELETE FROM rosters WHERE snapshot_date = ?", (snapshot_date,))
    for entry in entries:
        conn.execute(
            """
            INSERT INTO rosters (team, player_name, role, snapshot_date, tournament)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                entry["team"],
                entry["player_name"],
                entry["role"],
                snapshot_date,
                entry["tournament"],
            ),
        )
    return len(entries)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    session = _make_session()
    login(session)  # authenticates if creds are in .env; otherwise anonymous
    snapshot_date = date.today().isoformat()

    # 1. Discover 2026 secondary tournaments for our leagues
    logger.info("Discovering T2 tournaments from Leaguepedia…")
    all_tournaments = discover_t2_tournaments(session)
    target_tournaments = select_latest_per_league(all_tournaments)
    logger.info(f"  Targeting {len(target_tournaments)} tournaments (one per league)")

    # 2. Fetch rosters for all selected tournaments in a single batched query
    by_page = {t["overview_page"]: t for t in target_tournaments}
    pages = [t["overview_page"] for t in target_tournaments]
    logger.info(f"  Fetching rosters for {len(pages)} tournaments in one batched query…")
    roster_rows = fetch_rosters(pages, session)
    logger.info(f"  Retrieved {len(roster_rows)} roster rows")

    all_entries: List[Dict[str, Any]] = []
    seen_pages = set()
    for row in roster_rows:
        page = row.get("OverviewPage", "")
        t = by_page.get(page)
        if t is None:
            logger.warning(f"  Roster row with unmapped OverviewPage {page!r} — skipping")
            continue
        seen_pages.add(page)
        entries = parse_roster_row(row, t["name"], t["oe_league"])
        all_entries.extend(entries)

    # Preserve the old per-tournament "no roster data" visibility
    for page, t in by_page.items():
        if page not in seen_pages:
            logger.warning(f"  No roster data for {t['name']} — skipped")

    logger.info(f"Fetched {len(all_entries)} player-slot entries across all leagues")

    # 3. Save raw snapshot
    raw_path = RAW_DIR / f"{snapshot_date}.json"
    raw_path.write_text(json.dumps(all_entries, indent=2, ensure_ascii=False))
    logger.info(f"Saved raw snapshot → {raw_path}")

    # 4. Fuzzy-match against TTP players
    # timeout=30: wait up to 30s for a transient lock (e.g. a DB viewer) instead
    # of failing the whole run instantly on "database is locked".
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        ttp_players = load_ttp_players(conn)
        logger.info(f"Loaded {len(ttp_players)} TTP players for fuzzy matching (cutoff={FUZZY_CUTOFF})")

        matched, unmatched = match_players(all_entries, ttp_players)
        match_pct = 100 * len(matched) / max(len(all_entries), 1)
        logger.info(f"Fuzzy match: {len(matched)}/{len(all_entries)} matched ({match_pct:.1f}%)")

        # Save unmatched for manual review
        UNMATCHED_PATH.write_text(json.dumps(unmatched, indent=2, ensure_ascii=False))
        if unmatched:
            logger.info(f"Saved {len(unmatched)} unmatched players → {UNMATCHED_PATH}")

        # 5. Upsert all entries (matched + unmatched) to SQLite
        n = upsert_rosters(all_entries, snapshot_date, conn)
        conn.commit()
        logger.info(f"Upserted {n} roster rows for {snapshot_date}")

    finally:
        conn.close()

    # 6. Summary by league
    from collections import Counter
    league_counts = Counter(e["oe_league"] for e in all_entries)
    parts = " | ".join(f"{k}:{v}" for k, v in sorted(league_counts.items()))
    logger.info(f"Players by league: {parts}")

    logger.info("Roster scraper complete.")


if __name__ == "__main__":
    main()
