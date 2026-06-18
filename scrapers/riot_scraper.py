"""
Riot API soloq scraper — backfills soloq ratings for current-roster players who
have no TrackingThePros link, using the Riot account IDs that Leaguepedia
publishes in its `Players.SoloqueueIds` field.

Pipeline:
  1. Load the latest roster snapshot + TTP players from SQLite.
  2. Find roster players with no TTP fuzzy match (reuses roster_scraper logic).
  3. Pull their `SoloqueueIds` from Leaguepedia (reuses roster_scraper auth).
  4. Parse the (messy) SoloqueueIds markup into (region, gameName, tagLine).
  5. For each account: Riot account-v1 by-riot-id → puuid, then
     league-v4 by-puuid → RANKED_SOLO_5x5 tier/division/LP.
  6. Convert rank → numeric rating with ttp_scraper.rank_to_rating (IDENTICAL
     math, so these ratings are comparable to TTP's).
  7. Save raw snapshot to data/raw/riot/YYYY-MM-DD.json.
  8. Upsert players + accounts into SQLite (forward-only daily snapshot).

This widens *live* soloq coverage only — Riot exposes current rank, never
historical, so it adds nothing to backtests (same forward-only nature as TTP).

Rate limits (per Riot routing value — na1, euw1, americas, …):
  20 requests / 1 second   AND   100 requests / 2 minutes
Both are enforced proactively per routing value; 429s are honored reactively.

Requires RIOT_API_KEY in .env. Dev keys expire every 24h; the scheduled daily
run needs a production key.

Run:
  python scrapers/riot_scraper.py                 # backfill all recoverable
  python scrapers/riot_scraper.py --limit 5       # smoke-test a few accounts
  python scrapers/riot_scraper.py --dry-run       # resolve ranks, skip DB write
"""

import argparse
import json
import os
import re
import sqlite3
import time
from collections import defaultdict, deque
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Sibling scrapers (scrapers/ is on sys.path[0] when run as a script).
from roster_scraper import (
    _cargo_query,
    _make_session as _make_leaguepedia_session,
    _norm,
    fuzzy_match,
    login as leaguepedia_login,
)
from ttp_scraper import rank_to_rating

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
DB_PATH = _ROOT / "db" / "lol_model.db"
RAW_ROSTER_DIR = _ROOT / "data" / "raw" / "rosters"
RAW_DIR = _ROOT / "data" / "raw" / "riot"

load_dotenv(_ROOT / ".env")
RIOT_API_KEY = os.environ.get("RIOT_API_KEY", "").strip()
RIOT_HEADERS = {"X-Riot-Token": RIOT_API_KEY}

# Per-routing-value rate limits (count, window_seconds).
RIOT_RATE_RULES: List[Tuple[int, float]] = [(20, 1.0), (100, 120.0)]
RIOT_MAX_RETRIES = 4

# ---------------------------------------------------------------------------
# Region routing maps
# ---------------------------------------------------------------------------
# Leaguepedia SoloqueueIds region label → Riot platform (league-v4 routing).
LABEL_TO_PLATFORM = {
    "NA": "na1", "NA1": "na1",
    "EUW": "euw1", "EUW1": "euw1",
    "EUNE": "eun1", "EUN": "eun1",
    "KR": "kr",
    "BR": "br1", "BR1": "br1",
    "LAS": "la2", "LAN": "la1", "LA": "la1",
    "TR": "tr1", "TR1": "tr1",
    "RU": "ru",
    "JP": "jp1", "JP1": "jp1",
    "OCE": "oc1", "OC": "oc1",
    "PH": "ph2", "SG": "sg2", "TH": "th2", "TW": "tw2", "VN": "vn2",
}

# Fallback when SoloqueueIds omits a region label: infer from the OE league.
LEAGUE_TO_PLATFORM = {
    "NACL": "na1", "LCKC": "kr", "LJL": "jp1", "PCS": "tw2", "VCS": "vn2",
    "LFL": "euw1", "NLC": "euw1", "EM": "euw1", "TCL": "tr1",
    "LRN": "la1", "LRS": "la2", "LCO": "oc1", "LVP SL": "euw1",
}

# Riot platform → account-v1 regional cluster (valid: americas / asia / europe).
PLATFORM_TO_ACCT_REGION = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "oc1": "americas",
    "kr": "asia", "jp1": "asia", "tw2": "asia", "vn2": "asia",
    "ph2": "asia", "sg2": "asia", "th2": "asia",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe",
}

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_REGION_PREFIX_RE = re.compile(r"^\s*([A-Za-z]{2,4})\s*:\s*")


# ---------------------------------------------------------------------------
# Rate limiter (sliding window, multiple rules, per routing key)
# ---------------------------------------------------------------------------
class RateLimiter:
    """
    Enforce several (max_count, window_seconds) rules independently per key.
    Riot rate-limits per routing value, so each platform / regional cluster
    gets its own window. acquire() blocks until a request is allowed, then
    records it.
    """

    def __init__(self, rules: List[Tuple[int, float]]):
        self.rules = rules
        self.max_window = max(w for _, w in rules)
        self._hits: Dict[str, deque] = defaultdict(deque)

    def acquire(self, key: str) -> None:
        dq = self._hits[key]
        while True:
            now = time.monotonic()
            # Drop timestamps older than the widest window.
            while dq and dq[0] <= now - self.max_window:
                dq.popleft()

            wait = 0.0
            for max_count, window in self.rules:
                cnt = 0
                for t in reversed(dq):  # dq is ascending; newest at the end
                    if t > now - window:
                        cnt += 1
                    else:
                        break
                if cnt >= max_count:
                    oldest_in_window = dq[-cnt]
                    wait = max(wait, oldest_in_window + window - now)

            if wait <= 0:
                dq.append(time.monotonic())
                return
            time.sleep(wait + 0.01)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _make_riot_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=0)  # we handle retries/limits explicitly below
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _riot_get(
    session: requests.Session,
    routing: str,
    path: str,
    limiter: RateLimiter,
) -> Optional[Any]:
    """
    GET https://{routing}.api.riotgames.com{path} respecting the per-routing
    rate limit. Returns parsed JSON, or None for 404 / give-up. Raises on an
    auth failure (expired/missing key) so the run stops loudly.
    """
    url = f"https://{routing}.api.riotgames.com{path}"
    for attempt in range(RIOT_MAX_RETRIES):
        limiter.acquire(routing)
        try:
            r = session.get(url, headers=RIOT_HEADERS, timeout=15)
        except requests.RequestException as e:
            logger.warning(f"Riot request error ({routing}{path}): {e}")
            time.sleep(2 ** attempt)
            continue

        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        if r.status_code in (401, 403):
            raise RuntimeError(
                f"Riot API rejected the key ({r.status_code}) — RIOT_API_KEY is "
                "missing, invalid, or expired. Dev keys last 24h; regenerate at "
                "https://developer.riotgames.com/."
            )
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "10"))
            logger.warning(f"429 on {routing} — sleeping {wait}s (Retry-After)")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        logger.warning(f"Riot {r.status_code} on {path}: {r.text[:160]}")
        return None
    logger.warning(f"Gave up after {RIOT_MAX_RETRIES} attempts: {routing}{path}")
    return None


# ---------------------------------------------------------------------------
# SoloqueueIds parsing
# ---------------------------------------------------------------------------
def parse_soloq_ids(raw: str) -> List[Dict[str, Optional[str]]]:
    """
    Parse Leaguepedia's SoloqueueIds markup into Riot-ID accounts.

    Handles forms like:
      "'''KR:''' We4y"                          → bare name (skipped, no tagLine)
      "'''NA:''' AoJune#3354"                   → NA / AoJune / 3354
      "'''LAS''': Apoka#GOAT <br> '''BR''': X#br1"  → two accounts
      "no school pls#APP"                        → no region label / has tagLine

    Only modern Riot IDs (gameName#tagLine) are returned — account-v1
    by-riot-id requires a tagLine, so legacy bare names are dropped.
    """
    out: List[Dict[str, Optional[str]]] = []
    cleaned = raw.replace("'''", "")
    for chunk in _BR_RE.split(cleaned):
        chunk = chunk.strip()
        if not chunk:
            continue
        region = None
        m = _REGION_PREFIX_RE.match(chunk)
        if m:
            region = m.group(1).upper()
            chunk = chunk[m.end():].strip()
        if "#" not in chunk:
            continue  # legacy bare summoner name — unresolvable
        game_name, _, tag_line = chunk.rpartition("#")
        game_name, tag_line = game_name.strip(), tag_line.strip()
        if not game_name or not tag_line:
            continue
        out.append({
            "region_label": region,
            "platform": LABEL_TO_PLATFORM.get(region) if region else None,
            "game_name": game_name,
            "tag_line": tag_line,
        })
    return out


# ---------------------------------------------------------------------------
# Riot lookups
# ---------------------------------------------------------------------------
def resolve_puuid(
    session: requests.Session, platform: str, game_name: str, tag_line: str,
    limiter: RateLimiter,
) -> Optional[str]:
    """account-v1 by-riot-id → puuid (account data is global; routed by cluster)."""
    region = PLATFORM_TO_ACCT_REGION.get(platform, "americas")
    path = (
        "/riot/account/v1/accounts/by-riot-id/"
        f"{quote(game_name, safe='')}/{quote(tag_line, safe='')}"
    )
    data = _riot_get(session, region, path, limiter)
    return data.get("puuid") if data else None


def fetch_soloq_rank(
    session: requests.Session, platform: str, puuid: str, limiter: RateLimiter,
) -> Optional[Dict[str, Any]]:
    """
    league-v4 by-puuid → the RANKED_SOLO_5x5 entry, normalized to
    {tier, division, lp}. Returns {"tier": "Unranked", ...} if not ranked.
    """
    path = f"/lol/league/v4/entries/by-puuid/{puuid}"
    data = _riot_get(session, platform, path, limiter)
    if not data:
        return {"tier": "Unranked", "division": None, "lp": 0}
    for entry in data:
        if entry.get("queueType") == "RANKED_SOLO_5x5":
            return {
                "tier": str(entry.get("tier", "")).title(),  # "DIAMOND" → "Diamond"
                "division": entry.get("rank"),                # I–IV (None for Master+)
                "lp": int(entry.get("leaguePoints", 0)),
            }
    return {"tier": "Unranked", "division": None, "lp": 0}


# ---------------------------------------------------------------------------
# Input assembly
# ---------------------------------------------------------------------------
def _latest_roster_entries() -> List[Dict[str, Any]]:
    files = sorted(RAW_ROSTER_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No roster snapshot in {RAW_ROSTER_DIR} — run roster_scraper.py first."
        )
    logger.info(f"Using roster snapshot {files[-1].name}")
    return json.loads(files[-1].read_text())


def find_unmatched(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Roster players (deduped) with no fuzzy match in the TTP players table."""
    ttp = [r[0] for r in conn.execute("SELECT player_name FROM players")]
    norm_map = {_norm(c): c for c in ttp}

    seen: Dict[str, Dict[str, Any]] = {}
    for e in _latest_roster_entries():
        name = e["player_name"]
        if name in seen:
            continue
        if fuzzy_match(name, ttp, norm_map=norm_map):
            continue
        seen[name] = {
            "player_name": name,
            "role": e.get("role", ""),
            "team": e.get("team", ""),
            "oe_league": e.get("oe_league", ""),
        }
    return list(seen.values())


def fetch_soloq_id_field(
    handles: List[str], session: requests.Session
) -> Dict[str, str]:
    """Pull non-empty SoloqueueIds for the given handles (exact ID join, chunked)."""
    found: Dict[str, str] = {}
    for i in range(0, len(handles), 120):
        batch = handles[i:i + 120]
        quoted = ", ".join("'" + h.replace("'", "\\'") + "'" for h in batch)
        rows = _cargo_query(session, {
            "tables": "Players",
            "fields": "ID, SoloqueueIds",
            "where": f"ID IN ({quoted})",
            "limit": "500",
        })
        for r in rows:
            val = (r.get("SoloqueueIds", "") or "").strip()
            if val:
                found[r["ID"]] = val
    return found


def build_targets(unmatched: List[Dict[str, Any]], soloq_ids: Dict[str, str]) -> List[Dict[str, Any]]:
    """One target per (player, resolvable account)."""
    # case-insensitive lookup of the SoloqueueIds field
    by_lower = {k.lower(): v for k, v in soloq_ids.items()}
    targets: List[Dict[str, Any]] = []
    for p in unmatched:
        raw = by_lower.get(p["player_name"].lower())
        if not raw:
            continue
        for acct in parse_soloq_ids(raw):
            platform = acct["platform"] or LEAGUE_TO_PLATFORM.get(p["oe_league"])
            if not platform:
                logger.debug(
                    f"  No platform for {p['player_name']} "
                    f"({acct['game_name']}#{acct['tag_line']}) — skipping"
                )
                continue
            targets.append({**p, **acct, "platform": platform})
    return targets


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------
def upsert(results: List[Dict[str, Any]], snapshot_date: str) -> Tuple[int, int]:
    """Upsert player identity rows + insert today's account snapshots."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    n_players = n_accounts = 0
    try:
        for r in results:
            conn.execute(
                """
                INSERT INTO players (player_name, role, team, region, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(player_name) DO UPDATE SET
                    role       = excluded.role,
                    team       = excluded.team,
                    region     = COALESCE(excluded.region, players.region),
                    updated_at = excluded.updated_at
                """,
                (r["player_name"], r["role"], r["team"],
                 r.get("region_label"), snapshot_date),
            )
            n_players += 1
            player_id = conn.execute(
                "SELECT id FROM players WHERE player_name = ?", (r["player_name"],)
            ).fetchone()[0]
            conn.execute(
                """
                INSERT INTO accounts
                    (player_id, summoner_name, server, rank_tier, lp,
                     soloq_rating, snapshot_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (player_id, f"{r['game_name']}#{r['tag_line']}", r["platform"],
                 r["tier"], r["lp"], r["rating"], snapshot_date),
            )
            n_accounts += 1
        conn.commit()
    finally:
        conn.close()
    return n_players, n_accounts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(limit: Optional[int] = None, dry_run: bool = False) -> None:
    if not RIOT_API_KEY:
        raise SystemExit(
            "RIOT_API_KEY is not set in .env — cannot call the Riot API. "
            "Get a key at https://developer.riotgames.com/."
        )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    today = str(date.today())

    # 1–2. unmatched roster players
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        unmatched = find_unmatched(conn)
    finally:
        conn.close()
    logger.info(f"{len(unmatched)} roster players without a TTP soloq link")

    # 3. SoloqueueIds from Leaguepedia
    lp_session = _make_leaguepedia_session()
    leaguepedia_login(lp_session)
    soloq_ids = fetch_soloq_id_field([p["player_name"] for p in unmatched], lp_session)
    logger.info(f"{len(soloq_ids)} of them have a SoloqueueIds field")

    # 4. parse → resolvable Riot-ID targets
    targets = build_targets(unmatched, soloq_ids)
    if limit:
        targets = targets[:limit]
    logger.info(f"{len(targets)} resolvable Riot-ID accounts to look up")
    if not targets:
        logger.info("Nothing to do — exiting.")
        return

    # 5. Riot lookups
    riot = _make_riot_session()
    limiter = RateLimiter(RIOT_RATE_RULES)
    results: List[Dict[str, Any]] = []
    n_not_found = n_unranked = 0
    for i, t in enumerate(targets, 1):
        puuid = resolve_puuid(riot, t["platform"], t["game_name"], t["tag_line"], limiter)
        if not puuid:
            n_not_found += 1
            logger.debug(f"  [{i}/{len(targets)}] no account: "
                         f"{t['game_name']}#{t['tag_line']} ({t['platform']})")
            continue
        rank = fetch_soloq_rank(riot, t["platform"], puuid, limiter)
        rating = rank_to_rating(rank["tier"], rank["division"], rank["lp"])
        if rank["tier"] == "Unranked":
            n_unranked += 1
        results.append({**t, **rank, "puuid": puuid, "rating": rating})
        logger.info(
            f"  [{i}/{len(targets)}] {t['player_name']:18} "
            f"{rank['tier']} {rank['division'] or ''} {rank['lp']}LP "
            f"→ {rating:.0f}"
        )

    logger.info(
        f"Resolved {len(results)}/{len(targets)} accounts "
        f"({n_not_found} not found, {n_unranked} unranked)"
    )

    # 6. raw snapshot (before DB write)
    raw_path = RAW_DIR / f"{today}.json"
    raw_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info(f"Saved raw snapshot → {raw_path}")

    # 7. DB upsert
    if dry_run:
        logger.info("--dry-run: skipping DB write.")
        return
    n_players, n_accounts = upsert(results, today)
    logger.info(f"Upserted {n_players} players | inserted {n_accounts} account rows")
    logger.info("Riot scraper complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of accounts looked up (smoke test)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve ranks but do not write to the DB")
    args = parser.parse_args()
    main(limit=args.limit, dry_run=args.dry_run)
