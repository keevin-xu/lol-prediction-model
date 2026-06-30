"""
Shared team name matching — fuzzy matching + persistent alias table.

Used by: scanner.py, odds_scraper.py, price_tracker.py
Replaces in-code TEAM_ALIASES dicts with a DB-backed alias system.

CLI usage:
  python scrapers/team_matcher.py --add-alias "KCorp" --source oddsportal --db-name "Karmine Corp"
  python scrapers/team_matcher.py --list-aliases
  python scrapers/team_matcher.py --list-aliases --source oddsportal
"""

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

DB_PATH = _ROOT / "db" / "lol_model.db"
PROCESSED_DIR = _ROOT / "data" / "processed"

DEFAULT_CUTOFF = 0.80

STRIP_SUFFIXES = [
    " Esports", " eSports", " E-Sports", " e-Sports",
    " Gaming", " Academy", " Challengers", " Youth",
]


def normalize_team_name(name: str) -> str:
    s = name.strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    for suffix in STRIP_SUFFIXES:
        if s.lower().endswith(suffix.lower()):
            s = s[: -len(suffix)]
    s = re.sub(r"\s*\(.*?\)\s*", "", s)
    return s.strip().lower()


def load_aliases(source: Optional[str] = None) -> Dict[str, str]:
    conn = sqlite3.connect(DB_PATH)
    if source:
        rows = conn.execute(
            "SELECT external_name, db_team_name FROM team_name_aliases WHERE source = ?",
            (source,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT external_name, db_team_name FROM team_name_aliases"
        ).fetchall()
    conn.close()
    return {ext: db for ext, db in rows}


def add_alias(external_name: str, source: str, db_name: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO team_name_aliases (external_name, source, db_team_name)
        VALUES (?, ?, ?)
        ON CONFLICT(external_name, source) DO UPDATE SET db_team_name = excluded.db_team_name
        """,
        (external_name, source, db_name),
    )
    conn.commit()
    conn.close()
    logger.info(f"Alias added: '{external_name}' ({source}) → '{db_name}'")


def load_db_team_names() -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    names = [r[0] for r in conn.execute("SELECT team_name FROM teams").fetchall()]
    conn.close()
    return names


_SUFFIX_EXPANSIONS = [" Challengers", " Academy", " Esports", " Gaming", " Youth"]

# Zero-width / invisible characters that occasionally leak in from scraped
# sources (zero-width space, word joiner, BOM, braille blank) and silently
# break exact-match lookups.
_ZERO_WIDTH_CHARS = "​⁠﻿⠀"


def _core_match_ok(cleaned: str, candidate: str, cutoff: float) -> bool:
    """
    Guard against generic-suffix inflation: "Top Esports" vs "WAP Esports"
    scores 0.82 on the raw strings purely because both share "Esports",
    even though the actual team names "Top" vs "WAP" have nothing in
    common (ratio 0.33). Requires the suffix/diacritic-stripped core names
    to independently clear the cutoff before a fuzzy hit is accepted.
    """
    core_ext = normalize_team_name(cleaned)
    core_cand = normalize_team_name(candidate)
    return SequenceMatcher(None, core_ext, core_cand).ratio() >= cutoff


def match_team_name(
    external_name: str,
    db_teams: List[str],
    source: str = "unknown",
    cutoff: float = DEFAULT_CUTOFF,
) -> Optional[str]:
    cleaned = external_name.strip().lstrip(_ZERO_WIDTH_CHARS).strip()

    # Tier 1: exact alias lookup (source-specific, then global)
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT db_team_name FROM team_name_aliases WHERE external_name = ? AND source = ?",
        (cleaned, source),
    ).fetchone()
    if not row:
        row = conn.execute(
            "SELECT db_team_name FROM team_name_aliases WHERE external_name = ?",
            (cleaned,),
        ).fetchone()
    conn.close()
    if row:
        return row[0]

    # Tier 2: exact case-insensitive match against DB names
    lower_map = {t.lower(): t for t in db_teams}
    if cleaned.lower() in lower_map:
        return lower_map[cleaned.lower()]

    # Tier 3: raw-lowercase fuzzy match (no diacritics/suffix stripping —
    # catches near-exact names that normalization would otherwise distort).
    # Gated by _core_match_ok so a shared generic suffix can't carry an
    # unrelated core name across the cutoff.
    hits = get_close_matches(cleaned.lower(), list(lower_map.keys()), n=1, cutoff=cutoff)
    if hits and _core_match_ok(cleaned, lower_map[hits[0]], cutoff):
        return lower_map[hits[0]]

    # Tier 4: normalized fuzzy match (diacritics stripped, known suffixes stripped)
    norm_ext = normalize_team_name(cleaned)
    norm_map = {normalize_team_name(t): t for t in db_teams}
    hits = get_close_matches(norm_ext, list(norm_map.keys()), n=1, cutoff=cutoff)
    if hits:
        return norm_map[hits[0]]

    # Tier 5: suffix-expansion fuzzy match — try appending a known suffix to
    # the external name in case the DB's full name carries one the external
    # source dropped (e.g. external "T1" vs DB "T1 Academy"). Gated the same
    # way: the unexpanded core names must independently clear the cutoff.
    for suffix in _SUFFIX_EXPANSIONS:
        expanded = normalize_team_name(cleaned + suffix)
        hits = get_close_matches(expanded, list(norm_map.keys()), n=1, cutoff=cutoff)
        if hits and _core_match_ok(cleaned, norm_map[hits[0]], cutoff):
            return norm_map[hits[0]]

    return None


def bulk_match_teams(
    external_names: List[str],
    db_teams: List[str],
    source: str = "unknown",
    cutoff: float = DEFAULT_CUTOFF,
) -> Dict[str, Optional[str]]:
    results = {}
    unmatched = []
    for name in external_names:
        matched = match_team_name(name, db_teams, source=source, cutoff=cutoff)
        results[name] = matched
        if not matched:
            unmatched.append(name)
    if unmatched:
        logger.warning(f"{len(unmatched)}/{len(external_names)} team names unmatched from {source}")
    return results


def export_unmatched(
    unmatched: List[str],
    source: str,
    out_path: Optional[Path] = None,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = out_path or (PROCESSED_DIR / f"unmatched_teams_{source}.json")
    existing = []
    if path.exists():
        existing = json.loads(path.read_text())
    merged = sorted(set(existing + unmatched))
    path.write_text(json.dumps(merged, indent=2))
    logger.info(f"Exported {len(merged)} unmatched team names → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Team name alias manager")
    parser.add_argument("--add-alias", type=str, help="External team name to add")
    parser.add_argument("--source", type=str, default="unknown", help="Source (oddsportal, pinnacle, polymarket)")
    parser.add_argument("--db-name", type=str, help="DB team name to map to")
    parser.add_argument("--list-aliases", action="store_true", help="List all aliases")
    parser.add_argument("--test", type=str, help="Test matching an external name")
    args = parser.parse_args()

    if args.add_alias:
        if not args.db_name:
            parser.error("--db-name required with --add-alias")
        add_alias(args.add_alias, args.source, args.db_name)

    elif args.list_aliases:
        aliases = load_aliases(args.source if args.source != "unknown" else None)
        if not aliases:
            print("No aliases found.")
            return
        print(f"\n{'External Name':35} {'DB Name':35}")
        print("-" * 72)
        for ext, db in sorted(aliases.items()):
            print(f"{ext:35} {db:35}")
        print(f"\n{len(aliases)} aliases total")

    elif args.test:
        db_teams = load_db_team_names()
        result = match_team_name(args.test, db_teams, source=args.source)
        if result:
            print(f"Matched: '{args.test}' → '{result}'")
        else:
            print(f"No match found for '{args.test}'")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
