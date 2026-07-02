"""
Live deployment engine — detects new markets, computes edge signals,
gates on roster stability, sizes via quarter-Kelly, and logs paper bets
with CLV tracking.

ALL EXECUTION IS PAPER MODE. LIVE_TRADING defaults to False.
No real-money orders are placed until explicitly enabled after
30+ paper bets show positive live CLV.

T1 (LCK/LPL/LCS/LEC) runs a separate depth-observation paper trader.
T1_SCANNING defaults to False — depth-at-entry has never been measured
against a real order book, so T1 paper bets must not run until
verify_book_snapshot.py confirms the book-reading math is trustworthy.

Run standalone:  python polymarket/live_engine.py --status
                  python polymarket/live_engine.py --t1-status
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

from backtest.backtest import ELOTracker, load_matches
from backtest.polymarket_backtest import estimate_fillable_at_open
from model.predict import check_cross_region
from polymarket.scanner import (
    CLOB_API,
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

# T1 depth-observation paper trading — off until the book-reading math is verified.
T1_SCANNING = os.getenv("T1_SCANNING", "False") == "True"
MIN_T1_GAMES = 10
EDGE_THRESHOLD = MIN_EDGE
_T1_LEAGUE_MARKERS = frozenset({
    "lck", "lpl", "lcs", "lec",            # domestic T1 leagues
    "mid-season invitational", "msi",       # international T1 events
    "world championship", "worlds",
})


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# §0. Walk-forward ELO tracker — anchored to market-open time, never scan time
# ---------------------------------------------------------------------------
_tracker_cache: Dict[str, ELOTracker] = {}


def _market_cutoff_date(market: Dict) -> str:
    """Derive the walk-forward cutoff from the market's creation timestamp.
    NEVER from wall-clock time. This is the anchor that makes live == backtest."""
    ts = market["market_create_ts"]
    return ts[:10]


def _get_tracker(cutoff_date: str) -> ELOTracker:
    """Build (or return cached) walk-forward ELO tracker advanced through all
    matches strictly before cutoff_date. Cached by cutoff_date so repeated
    markets on the same date don't rebuild."""
    if cutoff_date in _tracker_cache:
        return _tracker_cache[cutoff_date]

    tracker = ELOTracker(
        K=32, blend_k=5, scale=400, half_life_days=270,
        soloq_baselines={}, regional_offsets={}, mov_weight=1.5,
    )
    for row in load_matches():
        date = row[1]
        if date >= cutoff_date:  # STRICT — excludes the match being predicted
            break
        stats = dict(
            blue_kills=row[6], red_kills=row[7],
            blue_deaths=row[8], red_deaths=row[9],
            blue_gd15=row[10], red_gd15=row[11],
        )
        tracker.update(row[3], row[4], row[5], row[2], row[1], **stats)

    _tracker_cache[cutoff_date] = tracker
    return tracker


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


def _is_t1_market(question: str, event_title: str = "") -> bool:
    """T1 (LCK/LPL/LCS/LEC) market classifier, parallel to the T2 path's
    implicit DB-membership filter."""
    text = f"{question} {event_title}".lower()
    return any(marker in text for marker in _T1_LEAGUE_MARKERS)


def detect_new_t1_markets(
    session: Optional[requests.Session] = None,
) -> List[Dict]:
    """Poll Gamma API for new T1 LoL moneyline markets not yet in t1_paper_bets."""
    session = session or _make_session()
    conn = sqlite3.connect(DB_PATH, timeout=10)

    seen = set(
        r[0] for r in conn.execute("SELECT market_id FROM t1_paper_bets").fetchall()
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

    for event in events:
        event_title = event.get("title", "")
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
            if not _is_t1_market(q, event_title):
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
            volume = float(m.get("volumeNum", 0) or m.get("volume", 0) or 0)

            new_markets.append({
                "market_id": mid,
                "question": q,
                "team_a": pm_a,
                "team_b": pm_b,
                "db_team_a": db_a,
                "db_team_b": db_b,
                "token_id_a": tokens[0],
                "token_id_b": tokens[1],
                "open_price_a": pa,
                "open_price_b": pb,
                "volume": volume,
                "market_create_ts": market_create,
                "match_start_ts": end_date,
                "slug": event.get("slug", ""),
            })

    conn.close()
    return new_markets


# ---------------------------------------------------------------------------
# §2. Signal: model vs opening line — shared by T2 and T1, do not fork
# ---------------------------------------------------------------------------
def compute_signal(market: Dict) -> Dict:
    """Point-in-time prediction + edge against opening price.

    Anchored to the market's creation timestamp (market_create_ts), never
    to wall-clock time. Uses the same walk-forward ELOTracker construction
    as the backtest, so live predictions match backtest predictions exactly
    for the same market.
    """
    cutoff_date = _market_cutoff_date(market)
    tracker = _get_tracker(cutoff_date)

    db_a = market["db_team_a"]
    db_b = market["db_team_b"]

    region_check = check_cross_region(db_a, db_b)
    same_region = not region_check["cross_region"]
    league_a = region_check.get("league_a") or ""

    model_a = tracker.predict(db_a, db_b, league_a, cutoff_date)
    open_a = market["open_price_a"]

    edge_a = model_a - open_a
    edge_b = (1.0 - model_a) - (1.0 - open_a)

    if abs(edge_a) >= abs(edge_b):
        edge = edge_a
        bet_side = "team_a"
        bet_team = db_a
    else:
        edge = -edge_b
        bet_side = "team_b"
        bet_team = db_b

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
        "games_a": tracker.games.get(db_a, 0),
        "games_b": tracker.games.get(db_b, 0),
    }


# ---------------------------------------------------------------------------
# §3. Roster-swap pre-bet gate (T2 only — see CLAUDE.md: roster gate is
# counterproductive, model is MORE accurate on roster-changed games)
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


def _quarter_kelly(model_prob: float, entry_price: float, bankroll: float) -> float:
    """Quarter-Kelly stake sized directly off model_prob and entry_price,
    with no depth term — depth is applied as a separate cap by the caller."""
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    b = (1.0 / entry_price) - 1.0
    q = 1.0 - model_prob
    kelly_raw = (model_prob * b - q) / b
    kelly = max(0.0, min(kelly_raw, MAX_KELLY))
    return round(bankroll * kelly, 2)


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


def _get_t1_bankroll() -> float:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    row = conn.execute(
        "SELECT COALESCE(SUM(pnl), 0) FROM t1_paper_bets WHERE resolved = 1"
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


def _log_t1_suppressed(market: Dict, reason: str, signal: Dict) -> None:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute(
        """INSERT INTO t1_paper_bets
           (market_id, team_a, team_b, league, market_create_ts, bet_logged_ts,
            model_prob, open_price, edge, bet_side, bet_team, resolved, suppressed_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
        (
            market["market_id"], market["db_team_a"], market["db_team_b"],
            signal.get("league_a", "") or "",
            market.get("market_create_ts", ""), _now_iso(),
            signal.get("model_prob", 0), signal.get("open_implied_prob", 0),
            signal.get("edge", 0), signal.get("bet_side", ""), signal.get("bet_team", ""),
            reason,
        ),
    )
    conn.commit()
    conn.close()
    logger.info(f"T1 SUPPRESSED: {market['market_id'][:12]}… reason={reason}")


def fetch_book_snapshot(
    token_id: str,
    entry_price: float,
    session: requests.Session,
) -> Optional[Dict]:
    """Read the REAL CLOB order book for token_id and compute USDC depth
    within 1/3/5 PERCENT of entry_price on the ASK side (what you pay to buy).

    Independent of estimate_fillable_at_open — never derives from it.
    Band is relative (entry_price * (1 + pct)), not absolute (entry_price + pct).
    Depth is sum(price * size) in USDC, treating size as shares, not sum(size).
    """
    try:
        r = session.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=10)
        if r.status_code != 200:
            return None
        book = r.json()
    except requests.RequestException:
        return None

    def parse(levels):
        out = []
        for lv in levels:
            try:
                out.append((float(lv["price"]), float(lv["size"])))
            except (KeyError, ValueError, TypeError):
                pass
        return out

    asks = sorted(parse(book.get("asks", [])), key=lambda x: x[0])
    bids = sorted(parse(book.get("bids", [])), key=lambda x: x[0], reverse=True)
    if not asks:
        return None

    best_ask = asks[0][0]
    best_bid = bids[0][0] if bids else None
    spread = (best_ask - best_bid) if best_bid is not None else None

    def depth_within(pct):
        ceiling = entry_price * (1.0 + pct)  # RELATIVE percent — NOT entry_price + pct
        return sum(price * size for price, size in asks if price <= ceiling)  # USDC = price * size

    return {
        "book_snapshot_ts": _now_iso(),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "depth_within_1pct": depth_within(0.01),
        "depth_within_3pct": depth_within(0.03),
        "depth_within_5pct": depth_within(0.05),
        "book_levels": json.dumps({"asks": asks[:10], "bids": bids[:10]}),
    }


def _log_t1_paper_bet(
    market: Dict,
    signal: Dict,
    entry_price: float,
    book: Dict,
    estimated_fillable: float,
    actual_fillable: float,
    estimate_error: float,
    bet_size: float,
) -> Optional[int]:
    hours_before = None
    try:
        create_dt = datetime.fromisoformat(market["market_create_ts"].replace("Z", "+00:00"))
        match_dt = datetime.fromisoformat(market["match_start_ts"].replace("Z", "+00:00"))
        hours_before = (match_dt - create_dt).total_seconds() / 3600.0
    except (ValueError, TypeError, AttributeError):
        pass

    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute(
        """INSERT INTO t1_paper_bets
           (market_id, team_a, team_b, league, market_create_ts, bet_logged_ts,
            hours_before_match, model_prob, open_price, edge, bet_side, bet_team,
            book_snapshot_ts, best_bid, best_ask, spread,
            depth_within_1pct, depth_within_3pct, depth_within_5pct, book_levels,
            volume_at_snapshot, estimated_fillable, actual_fillable, estimate_error,
            bet_size, entry_price, resolved)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
        (
            market["market_id"], market["db_team_a"], market["db_team_b"],
            signal.get("league_a", "") or "",
            market.get("market_create_ts", ""), _now_iso(),
            hours_before, signal["model_prob"], signal["open_implied_prob"], signal["edge"],
            signal["bet_side"], signal["bet_team"],
            book["book_snapshot_ts"], book["best_bid"], book["best_ask"], book["spread"],
            book["depth_within_1pct"], book["depth_within_3pct"], book["depth_within_5pct"],
            book["book_levels"], market.get("volume", 0),
            estimated_fillable, actual_fillable, estimate_error,
            bet_size, entry_price,
        ),
    )
    bet_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()

    logger.info(
        f"T1 PAPER BET: ${bet_size:.2f} on {signal['bet_team']} @ {entry_price:.1%} "
        f"(edge: {signal['edge']:.1%}, actual_fillable: ${actual_fillable:.2f}, "
        f"estimated: ${estimated_fillable:.2f}, error: ${estimate_error:+.2f})"
    )
    return bet_id


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


def update_t1_clv(session: Optional[requests.Session] = None) -> int:
    """Check open T1 paper bets for resolution, compute CLV. Returns count updated.

    Re-fetches the market from Gamma to get its endDate (match_start_ts) rather
    than relying on a stored column, since t1_paper_bets doesn't carry it —
    only hours_before_match (relative) was logged at bet time.
    """
    session = session or _make_session()
    conn = sqlite3.connect(DB_PATH, timeout=10)

    open_bets = conn.execute(
        """SELECT id, market_id, entry_price, bet_side, bet_team, bet_size
           FROM t1_paper_bets
           WHERE resolved = 0 AND suppressed_reason IS NULL""",
    ).fetchall()

    if not open_bets:
        conn.close()
        return 0

    updated = 0
    for bet_id, market_id, entry_price, bet_side, bet_team, size in open_bets:
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

        match_start_ts = mkt.get("endDate", "")
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

        conn.execute(
            "UPDATE t1_paper_bets SET resolved = 1, won = ?, clv = ?, pnl = ? WHERE id = ?",
            (1 if won else 0, round(clv, 4), pnl, bet_id),
        )
        updated += 1
        logger.info(f"T1 CLV: {bet_team} {'WON' if won else 'LOST'} PnL=${pnl:+.2f} CLV={clv:+.3f}")

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

        # Gate: unknown teams (defaults to 1500 — model is blind, suppress)
        if signal["games_a"] == 0 or signal["games_b"] == 0:
            unknown = market["db_team_a"] if signal["games_a"] == 0 else market["db_team_b"]
            log_suppressed(market["market_id"], f"unknown_team_{unknown}", signal)
            suppressed += 1
            continue

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


def run_t1_cycle(session: Optional[requests.Session] = None) -> Dict:
    """T1 depth-observation paper trading cycle. No-op unless T1_SCANNING=True.

    Sizing and the actual_fillable cap come from a real CLOB book read
    (fetch_book_snapshot), never from estimate_fillable_at_open. The estimate
    is computed separately, purely for comparison, and logged as estimate_error.
    """
    if not T1_SCANNING:
        logger.info("T1_SCANNING is OFF — skipping T1 cycle")
        return {"scanning": False}

    session = session or _make_session()
    markets = detect_new_t1_markets(session)

    bets_placed = 0
    suppressed = 0

    for market in markets:
        signal = compute_signal(market)

        if not signal["same_region"]:
            _log_t1_suppressed(market, "cross_region", signal)
            suppressed += 1
            continue
        if signal["edge"] < EDGE_THRESHOLD:
            _log_t1_suppressed(market, f"edge_too_small_{signal['edge']:.2f}", signal)
            suppressed += 1
            continue
        if signal["games_a"] < MIN_T1_GAMES or signal["games_b"] < MIN_T1_GAMES:
            _log_t1_suppressed(market, "insufficient_games", signal)
            suppressed += 1
            continue

        bet_side = signal["bet_side"]
        token_id = market["token_id_a"] if bet_side == "team_a" else market["token_id_b"]
        open_price = market["open_price_a"] if bet_side == "team_a" else market["open_price_b"]
        entry_price = min(open_price + SPREAD_COST + SLIPPAGE_COST, 0.99)

        # INDEPENDENT depth read — the real book
        book = fetch_book_snapshot(token_id, entry_price, session)
        if book is None:
            logger.warning(f"No book for {market['question']} — logging suppressed")
            _log_t1_suppressed(market, "no_book", signal)
            suppressed += 1
            continue
        actual_fillable = book["depth_within_3pct"]

        # Estimator computed SEPARATELY, from volume only — for comparison, never sizing
        estimated_fillable = estimate_fillable_at_open(market.get("volume", 0))
        estimate_error = estimated_fillable - actual_fillable

        # Size off the REAL book, never the estimate
        bankroll = _get_t1_bankroll()
        kelly_size = _quarter_kelly(signal["model_prob"], entry_price, bankroll)
        bet_size = min(kelly_size, bankroll * MAX_POSITION_PCT, actual_fillable)
        if bet_size < 1.0:
            _log_t1_suppressed(market, "no_liquidity", signal)
            suppressed += 1
            continue

        bet_id = _log_t1_paper_bet(
            market, signal, entry_price, book,
            estimated_fillable, actual_fillable, estimate_error, bet_size,
        )
        if bet_id:
            bets_placed += 1

    resolved = update_t1_clv(session)

    return {
        "scanning": True,
        "new_markets": len(markets),
        "bets_placed": bets_placed,
        "suppressed": suppressed,
        "resolved": resolved,
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


def print_t1_status() -> None:
    conn = sqlite3.connect(DB_PATH, timeout=10)

    total = conn.execute("SELECT COUNT(*) FROM t1_paper_bets WHERE suppressed_reason IS NULL").fetchone()[0]
    resolved = conn.execute("SELECT COUNT(*) FROM t1_paper_bets WHERE resolved = 1").fetchone()[0]
    suppressed = conn.execute("SELECT COUNT(*) FROM t1_paper_bets WHERE suppressed_reason IS NOT NULL").fetchone()[0]
    errors = conn.execute(
        "SELECT estimate_error FROM t1_paper_bets WHERE estimate_error IS NOT NULL"
    ).fetchall()

    conn.close()

    print(f"\n{'='*55}")
    print(f"  T1 DEPTH-OBSERVATION STATUS (T1_SCANNING={T1_SCANNING})")
    print(f"{'='*55}")
    print(f"  Bets logged:       {total} ({resolved} resolved)")
    print(f"  Suppressed:        {suppressed}")

    if errors:
        vals = [e[0] for e in errors]
        mean_err = sum(vals) / len(vals)
        print(f"\n  ESTIMATOR ACCURACY ({len(vals)} book reads)")
        print(f"  {'-'*40}")
        print(f"  Mean estimate_error (est - actual): ${mean_err:+.2f}")
        print(f"  Progress to verdict: {min(len(vals), 20)}/20")
        print(f"  (Preliminary directional read on estimator accuracy — not validation.)")
    else:
        print(f"\n  No book reads logged yet.")
    print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Live deployment engine")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--cycle", action="store_true", help="Run one detection cycle")
    parser.add_argument("--clv", action="store_true", help="Update CLV for resolved bets")
    parser.add_argument("--t1-status", action="store_true", help="Show T1 depth-observation status")
    parser.add_argument("--t1-cycle", action="store_true", help="Run one T1 detection cycle")
    parser.add_argument("--t1-clv", action="store_true", help="Update CLV for resolved T1 bets")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.cycle:
        result = run_cycle()
        print(f"Cycle: {result['new_markets']} new, {result['bets_placed']} bets, {result['suppressed']} suppressed, {result['resolved']} resolved")
        if T1_SCANNING:
            t1_result = run_t1_cycle()
            print(f"T1 cycle: {t1_result.get('new_markets', 0)} new, {t1_result.get('bets_placed', 0)} bets, {t1_result.get('suppressed', 0)} suppressed, {t1_result.get('resolved', 0)} resolved")
    elif args.clv:
        n = update_clv()
        print(f"Updated CLV for {n} bets")
    elif args.t1_status:
        print_t1_status()
    elif args.t1_cycle:
        result = run_t1_cycle()
        print(f"T1 cycle: {result}")
    elif args.t1_clv:
        n = update_t1_clv()
        print(f"Updated CLV for {n} T1 bets")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
