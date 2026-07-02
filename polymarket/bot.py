"""
Discord bot — T1 observation tool.

T2 is excluded entirely: edge is too thin after costs, and the opening-line
signal for same-region T2 is not worth the noise. Bot focuses exclusively on
T1 (LCK, LPL, LCS, LEC, MSI, Worlds).

Two categories of output, always labeled:
  OBSERVATIONAL (trustworthy now):  price tracking, new-market detection, depth logging
  SIGNAL (may be stale):            model edge vs line — labeled with OE data date

Alerts fire ONLY on:
  (a) A genuinely new T1 moneyline market opening (one embed, no spam)
  (b) A watched T1 market resolving

All other output is pull-based via slash commands.

Setup:
  1. Create a Discord bot at https://discord.com/developers/applications
  2. Add DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID to .env
  3. Invite the bot with Send Messages + Embed Links permissions

Run:  python polymarket/bot.py
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import discord
import requests
from discord import app_commands
from discord.ext import tasks
from dotenv import load_dotenv
from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from backtest.polymarket_backtest import estimate_fillable_at_open
from model.blend import get_all_ratings
from model.predict import predict_match
from polymarket.live_engine import (
    MAX_POSITION_PCT,
    SPREAD_COST,
    SLIPPAGE_COST,
    T1_SCANNING,
    _get_t1_bankroll,
    _is_t1_market,
    _make_session,
    _quarter_kelly,
    compute_signal,
    detect_new_t1_markets,
    fetch_book_snapshot,
    update_t1_clv,
)
from polymarket.price_tracker import check_market_resolutions
from polymarket.scanner import (
    GAMMA_API,
    load_db_team_names,
    match_team_name,
    parse_teams_from_question,
)

DB_PATH = _ROOT / "db" / "lol_model.db"
SCAN_INTERVAL_MINUTES = 5
MIN_EDGE = 0.10
OE_STALE_DAYS = 14  # warn when max match date is older than this


# ---------------------------------------------------------------------------
# OE data freshness
# ---------------------------------------------------------------------------
def _get_oe_date() -> str:
    """Return max(date) from matches as a proxy for OE data freshness."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        row = conn.execute("SELECT MAX(date) FROM matches").fetchone()
        conn.close()
        return row[0][:10] if row and row[0] else "unknown"
    except Exception:
        return "unknown"


def _oe_is_stale(oe_date: str) -> bool:
    if oe_date == "unknown":
        return True
    try:
        d = datetime.strptime(oe_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - d).days > OE_STALE_DAYS
    except ValueError:
        return True


def _stale_label(oe_date: str) -> str:
    """One-line warning appended to any signal output when OE data is stale."""
    if _oe_is_stale(oe_date):
        return f"⚠️ SIGNAL STALE — OE data {oe_date}, not tradeable"
    return ""


# ---------------------------------------------------------------------------
# T1 watchlist
# ---------------------------------------------------------------------------
def _ensure_t1_watchlist() -> None:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS t1_watchlist (
            id INTEGER PRIMARY KEY,
            market_id TEXT NOT NULL UNIQUE,
            question TEXT,
            team_a TEXT,
            team_b TEXT,
            db_team_a TEXT,
            db_team_b TEXT,
            token_id_a TEXT,
            token_id_b TEXT,
            open_price_a REAL,
            open_price_b REAL,
            slug TEXT,
            added_ts TEXT
        )
    """)
    conn.commit()
    conn.close()


def _get_watchlist() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM t1_watchlist ORDER BY added_ts DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _add_to_watchlist(market: Dict) -> bool:
    """Returns True if newly added (False if already present)."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute(
            """INSERT OR IGNORE INTO t1_watchlist
               (market_id, question, team_a, team_b, db_team_a, db_team_b,
                token_id_a, token_id_b, open_price_a, open_price_b, slug, added_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                market["market_id"], market.get("question", ""),
                market.get("team_a", ""), market.get("team_b", ""),
                market.get("db_team_a", ""), market.get("db_team_b", ""),
                market.get("token_id_a", ""), market.get("token_id_b", ""),
                market.get("open_price_a", 0.0), market.get("open_price_b", 0.0),
                market.get("slug", ""),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        added = conn.execute("SELECT changes()").fetchone()[0] > 0
        conn.commit()
        conn.close()
        return added
    except Exception as e:
        logger.error(f"Watchlist add failed: {e}")
        return False


def _remove_from_watchlist(market_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("DELETE FROM t1_watchlist WHERE market_id = ?", (market_id,))
    removed = conn.execute("SELECT changes()").fetchone()[0] > 0
    conn.commit()
    conn.close()
    return removed


def _search_live_t1_markets(query: str, session: requests.Session) -> List[Dict]:
    """Return live T1 markets whose question contains query (case-insensitive)."""
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
            return []
        events = r.json()
    except requests.RequestException:
        return []

    db_teams = load_db_team_names()
    query_lower = query.lower()
    results = []

    for event in events:
        event_title = event.get("title", "")
        for m in event.get("markets", []):
            q = m.get("question", "")
            ql = q.lower()
            if query_lower not in ql and query_lower not in event_title.lower():
                continue
            if not _is_t1_market(q, event_title):
                continue
            if "(bo" not in ql or " vs " not in ql:
                continue
            if any(x in ql for x in ["game 1", "game 2", "game 3", "game 4", "game 5", "handicap"]):
                continue

            prices = m.get("outcomePrices", "[]")
            tokens = m.get("clobTokenIds", "[]")
            if isinstance(prices, str):
                try: prices = json.loads(prices)
                except: continue
            if isinstance(tokens, str):
                try: tokens = json.loads(tokens)
                except: continue
            if len(prices) < 2 or len(tokens) < 2:
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

            results.append({
                "market_id": m.get("id", ""),
                "question": q,
                "team_a": pm_a, "team_b": pm_b,
                "db_team_a": db_a or pm_a, "db_team_b": db_b or pm_b,
                "token_id_a": tokens[0], "token_id_b": tokens[1],
                "open_price_a": pa, "open_price_b": pb,
                "volume": float(m.get("volumeNum", 0) or m.get("volume", 0) or 0),
                "market_create_ts": m.get("startDate", m.get("createdAt", "")),
                "match_start_ts": m.get("endDate", ""),
                "slug": event.get("slug", ""),
            })

    return results


# ---------------------------------------------------------------------------
# Price recording for watched markets
# ---------------------------------------------------------------------------
def _record_watchlist_prices(session: requests.Session) -> int:
    """Fetch and record current price for every market in t1_watchlist."""
    watchlist = _get_watchlist()
    if not watchlist:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH, timeout=10)
    recorded = 0

    for mkt in watchlist:
        try:
            r = session.get(f"{GAMMA_API}/markets/{mkt['market_id']}", timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
        except requests.RequestException:
            continue

        prices = data.get("outcomePrices", [])
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: continue
        if len(prices) < 2:
            continue
        try:
            pa, pb = float(prices[0]), float(prices[1])
        except (ValueError, TypeError):
            continue

        volume = float(data.get("volumeNum", 0) or data.get("volume", 0) or 0)
        spread = float(data.get("spread", 0) or 0)

        # Register in polymarket_markets if not already present
        conn.execute(
            """INSERT OR IGNORE INTO polymarket_markets
               (market_id, condition_id, slug, question, team_a, team_b,
                db_team_a, db_team_b, token_id_a, token_id_b, url,
                first_seen, last_seen, status)
               VALUES (?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')""",
            (
                mkt["market_id"], mkt.get("slug", ""),
                mkt.get("question", ""),
                mkt.get("team_a", ""), mkt.get("team_b", ""),
                mkt.get("db_team_a", ""), mkt.get("db_team_b", ""),
                mkt.get("token_id_a", ""), mkt.get("token_id_b", ""),
                f"https://polymarket.com/event/{mkt.get('slug', '')}",
                now, now,
            ),
        )
        conn.execute(
            "UPDATE polymarket_markets SET last_seen = ? WHERE market_id = ?",
            (now, mkt["market_id"]),
        )

        conn.execute(
            """INSERT INTO polymarket_prices
               (market_id, timestamp, price_a, price_b, spread, volume, source)
               VALUES (?, ?, ?, ?, ?, ?, 'gamma')""",
            (mkt["market_id"], now, pa, pb, spread, volume),
        )
        recorded += 1

        # Flag if resolved
        if data.get("closed") and (pa >= 0.99 or pb >= 0.99):
            winner = mkt.get("db_team_a", "") if pa >= 0.99 else mkt.get("db_team_b", "")
            conn.execute(
                "UPDATE polymarket_markets SET status = 'resolved', resolution_winner = ?, resolution_time = ? WHERE market_id = ?",
                (winner, now, mkt["market_id"]),
            )

    conn.commit()
    conn.close()
    return recorded


def _get_watchlist_current_prices() -> List[Dict]:
    """Latest price snapshot for each watched market, with movement vs open."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    rows = conn.execute(
        """SELECT w.market_id, w.db_team_a, w.db_team_b,
                  w.open_price_a, w.open_price_b,
                  p.price_a, p.price_b, p.timestamp,
                  m.status, m.resolution_winner
           FROM t1_watchlist w
           LEFT JOIN polymarket_prices p ON p.market_id = w.market_id
               AND p.id = (SELECT MAX(id) FROM polymarket_prices WHERE market_id = w.market_id)
           LEFT JOIN polymarket_markets m ON m.market_id = w.market_id
           ORDER BY w.added_ts DESC"""
    ).fetchall()
    conn.close()
    return [
        {
            "market_id": r[0], "db_team_a": r[1], "db_team_b": r[2],
            "open_a": r[3], "open_b": r[4],
            "cur_a": r[5], "cur_b": r[6],
            "last_update": r[7], "status": r[8], "resolution_winner": r[9],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Depth observation logging
# ---------------------------------------------------------------------------
def _log_t1_depth_obs(
    market: Dict,
    signal: Dict,
    book: Optional[Dict],
    bet_size: float = 0.0,
) -> None:
    """Write one row to t1_paper_bets (depth observation + optional paper bet)."""
    hours_before: Optional[float] = None
    try:
        create_dt = datetime.fromisoformat(market["market_create_ts"].replace("Z", "+00:00"))
        match_dt = datetime.fromisoformat(market["match_start_ts"].replace("Z", "+00:00"))
        hours_before = (match_dt - create_dt).total_seconds() / 3600.0
    except (ValueError, TypeError, AttributeError, KeyError):
        pass

    entry_price = market["open_price_a"] if signal["bet_side"] == "team_a" else market["open_price_b"]
    entry_with_cost = min(entry_price + SPREAD_COST + SLIPPAGE_COST, 0.99)

    actual_fillable = book["depth_within_3pct"] if book else None
    estimated_fillable = estimate_fillable_at_open(market.get("volume", 0))
    estimate_error = (estimated_fillable - actual_fillable) if actual_fillable is not None else None

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
            market.get("market_create_ts", ""),
            datetime.now(timezone.utc).isoformat(),
            hours_before,
            signal["model_prob"], market["open_price_a"], signal["edge"],
            signal["bet_side"], signal["bet_team"],
            book["book_snapshot_ts"] if book else None,
            book["best_bid"] if book else None,
            book["best_ask"] if book else None,
            book["spread"] if book else None,
            book["depth_within_1pct"] if book else None,
            book["depth_within_3pct"] if book else None,
            book["depth_within_5pct"] if book else None,
            book["book_levels"] if book else None,
            market.get("volume", 0),
            estimated_fillable, actual_fillable, estimate_error,
            bet_size, entry_with_cost,
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Embed builders
# ---------------------------------------------------------------------------
def _build_t1_opening_embed(
    market: Dict,
    signal: Dict,
    book: Optional[Dict],
    oe_date: str,
) -> discord.Embed:
    """Single clean embed for a new T1 market opening. One push, no spam."""
    db_a = market["db_team_a"]
    db_b = market["db_team_b"]
    pa = market["open_price_a"]
    pb = market["open_price_b"]
    slug = market.get("slug", "")
    url = f"https://polymarket.com/event/{slug}" if slug else ""

    embed = discord.Embed(
        title=f"New T1 Market: {db_a} vs {db_b}",
        url=url or discord.Embed.Empty,
        color=discord.Color.blue(),
    )

    # Observational fields — always shown, always trustworthy
    embed.add_field(
        name="Opening Prices",
        value=f"**{db_a}**: {pa:.0%}   **{db_b}**: {pb:.0%}",
        inline=False,
    )

    hours_before: Optional[float] = None
    try:
        create_dt = datetime.fromisoformat(market["market_create_ts"].replace("Z", "+00:00"))
        match_dt = datetime.fromisoformat(market["match_start_ts"].replace("Z", "+00:00"))
        hours_before = (match_dt - create_dt).total_seconds() / 3600.0
        if hours_before >= 24:
            lead_str = f"{hours_before / 24:.1f} days before match"
        else:
            lead_str = f"{hours_before:.1f}h before match"
    except (ValueError, TypeError, AttributeError, KeyError):
        lead_str = "unknown"

    market_ts = market.get("market_create_ts", "")
    ts_short = market_ts[:16].replace("T", " ") + " UTC" if market_ts else "unknown"
    embed.add_field(name="Market Opened", value=ts_short, inline=True)
    embed.add_field(name="Lead Time", value=lead_str, inline=True)

    if book:
        depth_3 = book["depth_within_3pct"]
        depth_5 = book["depth_within_5pct"]
        spread = book.get("spread")
        spread_str = f"{spread:.3f}" if spread else "—"
        embed.add_field(
            name="Depth at Open (OBSERVATIONAL)",
            value=f"3%: **${depth_3:,.0f}**   5%: ${depth_5:,.0f}   Spread: {spread_str}",
            inline=False,
        )
    else:
        embed.add_field(
            name="Depth at Open",
            value="⚠️ Book fetch failed — no depth logged",
            inline=False,
        )

    # Signal — shown for dev purposes but labeled if stale
    stale = _stale_label(oe_date)
    model_line = f"{db_a} **{signal['model_prob']:.0%}** vs {db_b} **{1 - signal['model_prob']:.0%}**"
    edge_line = f"Edge: **{signal['edge']:+.0%}** on {signal['bet_team']}"
    region_note = "" if signal["same_region"] else "  ⚠️ cross-region"

    if stale:
        embed.add_field(
            name="Signal (STALE — development only)",
            value=f"{model_line}\n{edge_line}{region_note}\n_{stale}_",
            inline=False,
        )
    else:
        action = "✅ Would bet" if (signal["edge"] >= MIN_EDGE and signal["same_region"]) else "— Below threshold"
        embed.add_field(
            name="Signal",
            value=f"{model_line}\n{edge_line}{region_note}\n{action}",
            inline=False,
        )

    league = signal.get("league_a") or ""
    mode_str = f"T1_SCANNING={'ON' if T1_SCANNING else 'OFF (observing)'}"
    embed.set_footer(text=f"{league} | {mode_str} | {market['market_id'][:12]}…")
    return embed


def _build_status_embed(bot: "T1ObserverBot") -> discord.Embed:
    oe_date = bot.oe_date
    stale = _oe_is_stale(oe_date)
    oe_val = f"🔴 {oe_date} — STALE (signal blocked until OE refreshes)" if stale else f"🟢 {oe_date}"

    uptime = datetime.now(timezone.utc) - bot.start_time
    h = int(uptime.total_seconds() // 3600)
    m = int((uptime.total_seconds() % 3600) // 60)
    last = bot.last_scan.strftime("%H:%M UTC") if bot.last_scan else "never"

    conn = sqlite3.connect(DB_PATH, timeout=10)
    watch_count = conn.execute("SELECT COUNT(*) FROM t1_watchlist").fetchone()[0]
    depth_count = conn.execute(
        "SELECT COUNT(*) FROM t1_paper_bets WHERE depth_within_3pct IS NOT NULL"
    ).fetchone()[0]
    t1_new_today = conn.execute(
        "SELECT COUNT(*) FROM t1_paper_bets WHERE bet_logged_ts >= date('now')"
    ).fetchone()[0]
    t1_resolved = conn.execute(
        "SELECT COUNT(*) FROM t1_paper_bets WHERE resolved = 1"
    ).fetchone()[0]
    conn.close()

    color = discord.Color.red() if stale else discord.Color.green()
    embed = discord.Embed(title="Bot Status", color=color)
    embed.add_field(name="OE Data", value=oe_val, inline=False)
    embed.add_field(name="Uptime", value=f"{h}h {m}m", inline=True)
    embed.add_field(name="Scans", value=str(bot.scan_count), inline=True)
    embed.add_field(name="Last Scan", value=last, inline=True)
    embed.add_field(name="T1 Watched", value=str(watch_count), inline=True)
    embed.add_field(name="Depth Obs", value=str(depth_count), inline=True)
    embed.add_field(name="New Today", value=str(t1_new_today), inline=True)
    embed.add_field(name="T1 Resolved", value=str(t1_resolved), inline=True)
    embed.add_field(
        name="Mode",
        value=f"T1_SCANNING={'ON' if T1_SCANNING else 'OFF'} | LIVE_TRADING=OFF",
        inline=False,
    )
    return embed


def _build_watchlist_embed(prices: List[Dict]) -> discord.Embed:
    embed = discord.Embed(
        title=f"T1 Watchlist ({len(prices)} markets)",
        color=discord.Color.blue(),
    )
    if not prices:
        embed.description = "No markets watched. Use /t1watch to add one."
        return embed

    lines = []
    for p in prices:
        status = p.get("status") or "active"
        if status == "resolved":
            winner = p.get("resolution_winner", "?")
            lines.append(f"`{p['db_team_a']} vs {p['db_team_b']}` — RESOLVED → **{winner}**")
            continue

        cur_a = p.get("cur_a")
        open_a = p.get("open_a")
        if cur_a is not None and open_a:
            move = cur_a - open_a
            arrow = "▲" if move > 0.005 else ("▼" if move < -0.005 else "—")
            lines.append(
                f"`{p['db_team_a']} vs {p['db_team_b']}` "
                f"{p['db_team_a']} {cur_a:.0%} ({move:+.1%}{arrow})"
            )
        else:
            lines.append(f"`{p['db_team_a']} vs {p['db_team_b']}` — no price yet")

    embed.description = "\n".join(lines)
    embed.set_footer(text="Price vs opening | /t1unwatch <market_id> to remove")
    return embed


def _build_depth_embed() -> discord.Embed:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    obs = conn.execute(
        """SELECT team_a, team_b, depth_within_3pct, estimated_fillable,
                  estimate_error, hours_before_match, bet_logged_ts
           FROM t1_paper_bets
           WHERE depth_within_3pct IS NOT NULL
           ORDER BY id DESC LIMIT 20"""
    ).fetchall()
    total = conn.execute(
        "SELECT COUNT(*) FROM t1_paper_bets WHERE depth_within_3pct IS NOT NULL"
    ).fetchone()[0]
    errors = [r[4] for r in obs if r[4] is not None]
    conn.close()

    embed = discord.Embed(
        title=f"T1 Depth Observations ({total} total)",
        color=discord.Color.teal(),
    )

    if errors:
        mean_err = sum(errors) / len(errors)
        progress = min(len(errors), 20)
        embed.add_field(
            name=f"Estimator Accuracy ({progress}/20 toward preliminary verdict)",
            value=(
                f"Mean error (est − actual): **${mean_err:+,.0f}**\n"
                f"{'Estimator overshoots' if mean_err > 0 else 'Estimator undershoots'} "
                f"the real book by ${abs(mean_err):,.0f} on average"
            ),
            inline=False,
        )
    else:
        embed.add_field(
            name="Estimator Accuracy",
            value="No observations yet",
            inline=False,
        )

    if obs:
        lines = []
        for r in obs[:10]:
            ta, tb, d3, _, err, hbm, ts = r
            hbm_str = f"{hbm:.0f}h" if hbm is not None else "?"
            err_str = f"err ${err:+.0f}" if err is not None else ""
            date_str = ts[:10] if ts else "?"
            lines.append(
                f"`{ta} vs {tb}` — ${d3:,.0f} @ 3% | {hbm_str} lead | {err_str} | {date_str}"
            )
        embed.add_field(name="Recent Observations", value="\n".join(lines), inline=False)

    embed.set_footer(
        text="depth_within_3pct = USDC fillable within 3% of entry price | Independent of volume estimator"
    )
    return embed


def _build_dashboard_embed(oe_date: str) -> discord.Embed:
    stale = _oe_is_stale(oe_date)
    color = discord.Color.orange() if stale else discord.Color.blue()
    embed = discord.Embed(
        title="T1 Dashboard",
        color=color,
    )

    if stale:
        embed.add_field(
            name="⚠️ Data Status",
            value=f"OE data: {oe_date} — **STALE**\nSignal not tradeable until OE refreshes past {oe_date}.",
            inline=False,
        )

    # Watchlist summary
    prices = _get_watchlist_current_prices()
    active = [p for p in prices if (p.get("status") or "active") == "active"]
    resolved = [p for p in prices if (p.get("status") or "active") == "resolved"]
    embed.add_field(
        name=f"Watched Markets ({len(prices)})",
        value=f"{len(active)} active, {len(resolved)} resolved",
        inline=True,
    )

    # Depth summary
    conn = sqlite3.connect(DB_PATH, timeout=10)
    depth_total = conn.execute(
        "SELECT COUNT(*) FROM t1_paper_bets WHERE depth_within_3pct IS NOT NULL"
    ).fetchone()[0]
    recent_obs = conn.execute(
        """SELECT team_a, team_b, depth_within_3pct, bet_logged_ts
           FROM t1_paper_bets WHERE depth_within_3pct IS NOT NULL
           ORDER BY id DESC LIMIT 5"""
    ).fetchall()
    errors = [
        r[0] for r in conn.execute(
            "SELECT estimate_error FROM t1_paper_bets WHERE estimate_error IS NOT NULL"
        ).fetchall()
    ]
    new_today = conn.execute(
        "SELECT COUNT(*) FROM t1_paper_bets WHERE bet_logged_ts >= date('now')"
    ).fetchone()[0]
    conn.close()

    mean_err_str = (
        f"Mean est error: ${sum(errors) / len(errors):+,.0f}"
        if errors else "No estimator data yet"
    )
    embed.add_field(name="Depth Obs", value=str(depth_total), inline=True)
    embed.add_field(name="New Today", value=str(new_today), inline=True)
    embed.add_field(name="Estimator", value=mean_err_str, inline=False)

    if active:
        lines = []
        for p in active[:5]:
            cur_a = p.get("cur_a")
            open_a = p.get("open_a")
            if cur_a is not None and open_a:
                move = cur_a - open_a
                arrow = "▲" if move > 0.005 else ("▼" if move < -0.005 else "—")
                lines.append(
                    f"`{p['db_team_a']} vs {p['db_team_b']}` "
                    f"{p['db_team_a']} {cur_a:.0%} ({move:+.1%}{arrow})"
                )
        if lines:
            embed.add_field(name="Active Prices", value="\n".join(lines), inline=False)

    if recent_obs:
        lines = []
        for ta, tb, d3, ts in recent_obs[:5]:
            date_str = ts[:10] if ts else "?"
            lines.append(f"`{ta} vs {tb}` — ${d3:,.0f} @ 3% | {date_str}")
        embed.add_field(name="Recent Depth Obs", value="\n".join(lines), inline=False)

    embed.set_footer(text=f"OE: {oe_date} | T1_SCANNING={'ON' if T1_SCANNING else 'OFF'}")
    return embed


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------
class T1ObserverBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.tree = app_commands.CommandTree(self)
        self.channel_id = int(os.environ.get("DISCORD_CHANNEL_ID", "0"))
        self.start_time = datetime.now(timezone.utc)
        self.last_scan: Optional[datetime] = None
        self.scan_count = 0
        self.oe_date: str = "unknown"

        self._setup_commands()

    def _setup_commands(self) -> None:

        # ------------------------------------------------------------------ #
        # T1 commands (new)
        # ------------------------------------------------------------------ #
        @self.tree.command(name="status", description="Bot status: OE data date, watchlist, depth observations")
        async def cmd_status(interaction: discord.Interaction) -> None:
            embed = _build_status_embed(self)
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="t1watch", description="Add a live T1 market to the watchlist (search by team name)")
        @app_commands.describe(query="Team name or fragment to search for (e.g. 'T1', 'G2', 'BLG')")
        async def cmd_t1watch(interaction: discord.Interaction, query: str) -> None:
            await interaction.response.defer()
            session = _make_session()
            results = _search_live_t1_markets(query, session)
            if not results:
                await interaction.followup.send(
                    f"No live T1 markets found matching **{query}**. "
                    f"Markets may have already resolved or the search term didn't match."
                )
                return
            if len(results) == 1:
                mkt = results[0]
                added = _add_to_watchlist(mkt)
                state = "added to" if added else "already in"
                embed = discord.Embed(
                    title=f"Watchlist: {mkt['db_team_a']} vs {mkt['db_team_b']}",
                    color=discord.Color.green() if added else discord.Color.yellow(),
                )
                embed.add_field(name="Status", value=state + " watchlist", inline=True)
                embed.add_field(
                    name="Prices",
                    value=f"{mkt['db_team_a']} {mkt['open_price_a']:.0%} / {mkt['db_team_b']} {mkt['open_price_b']:.0%}",
                    inline=True,
                )
                embed.add_field(name="Market ID", value=mkt["market_id"][:16] + "…", inline=False)
                await interaction.followup.send(embed=embed)
            else:
                lines = []
                for r in results[:8]:
                    lines.append(
                        f"`{r['market_id'][:12]}…`  {r['db_team_a']} vs {r['db_team_b']} "
                        f"({r['open_price_a']:.0%}/{r['open_price_b']:.0%})"
                    )
                await interaction.followup.send(
                    f"Found {len(results)} matches for **{query}**. Be more specific:\n"
                    + "\n".join(lines)
                )

        @self.tree.command(name="t1watchlist", description="Show watched T1 markets with current prices and movement")
        async def cmd_t1watchlist(interaction: discord.Interaction) -> None:
            prices = _get_watchlist_current_prices()
            embed = _build_watchlist_embed(prices)
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="t1unwatch", description="Remove a market from the T1 watchlist")
        @app_commands.describe(market_id="Market ID prefix (shown in /t1watchlist)")
        async def cmd_t1unwatch(interaction: discord.Interaction, market_id: str) -> None:
            # Accept a prefix match
            conn = sqlite3.connect(DB_PATH, timeout=10)
            row = conn.execute(
                "SELECT market_id, db_team_a, db_team_b FROM t1_watchlist WHERE market_id LIKE ?",
                (market_id + "%",),
            ).fetchone()
            conn.close()
            if not row:
                await interaction.response.send_message(
                    f"No watched market with ID starting with `{market_id}`. "
                    f"Use /t1watchlist to see market IDs."
                )
                return
            full_id, ta, tb = row
            _remove_from_watchlist(full_id)
            await interaction.response.send_message(
                f"Removed **{ta} vs {tb}** from watchlist."
            )

        @self.tree.command(name="t1dashboard", description="T1 at-a-glance: watched markets, depth observations, OE status")
        async def cmd_t1dashboard(interaction: discord.Interaction) -> None:
            embed = _build_dashboard_embed(self.oe_date)
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="t1depth", description="T1 depth observations and estimator accuracy")
        async def cmd_t1depth(interaction: discord.Interaction) -> None:
            embed = _build_depth_embed()
            await interaction.response.send_message(embed=embed)

        # ------------------------------------------------------------------ #
        # Existing commands (kept, refreshed wording)
        # ------------------------------------------------------------------ #
        @self.tree.command(name="predict", description="Model prediction for a matchup")
        @app_commands.describe(team_a="First team", team_b="Second team", best_of="Series format")
        @app_commands.choices(best_of=[
            app_commands.Choice(name="Bo1", value=1),
            app_commands.Choice(name="Bo3", value=3),
            app_commands.Choice(name="Bo5", value=5),
            app_commands.Choice(name="Bo7", value=7),
        ])
        async def cmd_predict(
            interaction: discord.Interaction,
            team_a: str,
            team_b: str,
            best_of: int = 1,
        ) -> None:
            result = predict_match(team_a, team_b, best_of=best_of)
            stale = _stale_label(self.oe_date)
            color = discord.Color.orange() if result.get("cross_region") else discord.Color.blue()
            embed = discord.Embed(
                title=f"{result['team_a']} vs {result['team_b']}",
                color=color,
            )
            embed.add_field(
                name="Ratings",
                value=f"{result['rating_a']:.1f} vs {result['rating_b']:.1f}",
                inline=True,
            )
            embed.add_field(
                name="Single Game",
                value=f"**{result['team_a']}**: {result['p_a']:.1%}   **{result['team_b']}**: {result['p_b']:.1%}",
                inline=True,
            )
            if result.get("series_p_a") is not None:
                embed.add_field(
                    name=f"Bo{best_of} Series",
                    value=f"**{result['team_a']}**: {result['series_p_a']:.1%}   **{result['team_b']}**: {result['series_p_b']:.1%}",
                    inline=False,
                )
            if result.get("warnings"):
                embed.add_field(name="Warnings", value="\n".join(result["warnings"]), inline=False)
            footer = "Calibrated"
            if stale:
                footer += f" | {stale}"
            embed.set_footer(text=footer)
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="leaderboard", description="Top 20 teams by blended rating")
        async def cmd_leaderboard(interaction: discord.Interaction) -> None:
            ratings = get_all_ratings()
            sorted_teams = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:20]
            lines = [
                f"`{i+1:2}. {name:28} {rating:7.1f}`"
                for i, (name, rating) in enumerate(sorted_teams)
            ]
            embed = discord.Embed(
                title="Top 20 Teams by Blended Rating",
                description="\n".join(lines),
                color=discord.Color.gold(),
            )
            embed.set_footer(text=f"OE data: {self.oe_date}")
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="scan", description="Force an immediate T1 market scan")
        async def cmd_scan(interaction: discord.Interaction) -> None:
            await interaction.response.defer()
            session = _make_session()
            t1_new = detect_new_t1_markets(session)
            lines = [f"T1 new markets found: **{len(t1_new)}**"]
            if t1_new:
                for m in t1_new[:8]:
                    lines.append(
                        f"  • {m['db_team_a']} vs {m['db_team_b']} "
                        f"({m['open_price_a']:.0%}/{m['open_price_b']:.0%})"
                    )
            else:
                lines.append("No new T1 markets since last scan.")
            await interaction.followup.send("\n".join(lines))

    # -----------------------------------------------------------------------
    # Background loops
    # -----------------------------------------------------------------------
    @tasks.loop(minutes=SCAN_INTERVAL_MINUTES)
    async def scan_loop(self) -> None:
        try:
            session = _make_session()

            # T1 new market detection — T1_SCANNING controls sizing only
            try:
                new_t1 = detect_new_t1_markets(session)
                channel = self.get_channel(self.channel_id)

                for market in new_t1:
                    # Compute signal (for dev display, may be stale)
                    try:
                        signal = compute_signal(market)
                    except Exception as e:
                        logger.warning(f"Signal compute failed for {market['market_id'][:12]}: {e}")
                        continue

                    # Book snapshot (observational — independent of any cost model)
                    bet_side = signal["bet_side"]
                    open_price = (
                        market["open_price_a"] if bet_side == "team_a" else market["open_price_b"]
                    )
                    entry_with_cost = min(open_price + SPREAD_COST + SLIPPAGE_COST, 0.99)
                    token_id = (
                        market["token_id_a"] if bet_side == "team_a" else market["token_id_b"]
                    )
                    book = fetch_book_snapshot(token_id, entry_with_cost, session)

                    # Sizing — only if T1_SCANNING=True and gates pass
                    bet_size = 0.0
                    if (
                        T1_SCANNING
                        and signal["same_region"]
                        and signal["edge"] >= MIN_EDGE
                        and signal["games_a"] >= 10
                        and signal["games_b"] >= 10
                        and book is not None
                    ):
                        bankroll = _get_t1_bankroll()
                        kelly = _quarter_kelly(signal["model_prob"], entry_with_cost, bankroll)
                        actual_fillable = book["depth_within_3pct"]
                        bet_size = min(kelly, bankroll * MAX_POSITION_PCT, actual_fillable)
                        bet_size = max(bet_size, 0.0)

                    # Log depth observation
                    try:
                        _log_t1_depth_obs(market, signal, book, bet_size)
                    except Exception as e:
                        logger.error(f"Depth obs log failed: {e}")

                    # Auto-add to watchlist
                    _add_to_watchlist(market)

                    # Post ONE clean notification
                    if channel:
                        embed = _build_t1_opening_embed(market, signal, book, self.oe_date)
                        await channel.send(embed=embed)

                    logger.info(
                        f"T1 new: {market['db_team_a']} vs {market['db_team_b']} "
                        f"depth=${book['depth_within_3pct']:,.0f}" if book else
                        f"T1 new: {market['db_team_a']} vs {market['db_team_b']} (no book)"
                    )

            except Exception as e:
                logger.error(f"T1 detection error: {e}")

            # Price snapshots for watchlisted T1 markets
            try:
                n = _record_watchlist_prices(session)
                if n > 0:
                    logger.debug(f"Recorded {n} T1 watchlist price snapshots")
            except Exception as e:
                logger.error(f"Watchlist price error: {e}")

            self.scan_count += 1
            self.last_scan = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Scan loop error: {e}")

    @tasks.loop(hours=1)
    async def settle_loop(self) -> None:
        """Hourly: update T1 CLV and post resolution alerts."""
        try:
            session = _make_session()

            # Keep watchlist market statuses current
            try:
                check_market_resolutions(session)
            except Exception:
                pass

            # T1 CLV update and resolution notifications
            try:
                resolved_count = update_t1_clv(session)
                if resolved_count > 0:
                    logger.info(f"T1: {resolved_count} bets resolved, CLV updated")
                    channel = self.get_channel(self.channel_id)
                    if channel:
                        conn = sqlite3.connect(DB_PATH, timeout=10)
                        recent = conn.execute(
                            """SELECT team_a, team_b, won, pnl, clv
                               FROM t1_paper_bets
                               WHERE resolved = 1
                               ORDER BY id DESC LIMIT ?""",
                            (resolved_count,),
                        ).fetchall()
                        conn.close()
                        for ta, tb, won, pnl, clv in recent:
                            icon = "✅" if won else "❌"
                            embed = discord.Embed(
                                title=f"{icon} T1 Resolved: {ta} vs {tb}",
                                color=discord.Color.green() if won else discord.Color.red(),
                            )
                            embed.add_field(name="P&L", value=f"${pnl:+.2f}" if pnl else "—", inline=True)
                            embed.add_field(name="CLV", value=f"{clv:+.3f}" if clv else "—", inline=True)
                            embed.set_footer(text="T1 paper observation")
                            await channel.send(embed=embed)
            except Exception as e:
                logger.error(f"T1 CLV update error: {e}")

        except Exception as e:
            logger.error(f"Settle loop error: {e}")

    @scan_loop.before_loop
    async def before_scan_loop(self) -> None:
        await self.wait_until_ready()

    @settle_loop.before_loop
    async def before_settle_loop(self) -> None:
        await self.wait_until_ready()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------
    async def on_ready(self) -> None:
        logger.info(f"Bot connected as {self.user} (ID: {self.user.id})")

        _ensure_t1_watchlist()
        self.oe_date = _get_oe_date()
        stale = _oe_is_stale(self.oe_date)

        # Sync slash commands to all guilds
        try:
            for guild in self.guilds:
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                logger.info(f"Synced {len(synced)} commands to {guild.name}")
        except Exception as e:
            logger.error(f"Command sync failed: {e}")

        # Start loops
        if not self.scan_loop.is_running():
            self.scan_loop.start()
        if not self.settle_loop.is_running():
            self.settle_loop.start()

        # Startup message
        channel = self.get_channel(self.channel_id)
        if channel:
            oe_line = (
                f"⚠️ OE data: **{self.oe_date}** — STALE. Signal blocked until refresh."
                if stale else
                f"✅ OE data: **{self.oe_date}**"
            )
            await channel.send(
                f"**T1 Observer Bot online.**\n"
                f"{oe_line}\n"
                f"T1_SCANNING={'ON' if T1_SCANNING else 'OFF'} | T2=disabled\n"
                f"Commands: `/status` `/t1dashboard` `/t1watch` `/t1watchlist` "
                f"`/t1depth` `/predict` `/leaderboard` `/scan`"
            )

    async def setup_hook(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    channel_id = os.environ.get("DISCORD_CHANNEL_ID", "").strip()

    if not token:
        logger.error(
            "DISCORD_BOT_TOKEN not set.\n\n"
            "Setup:\n"
            "  1. https://discord.com/developers/applications → New Application → Bot tab → Reset Token\n"
            "  2. Enable 'Message Content Intent' under Privileged Gateway Intents\n"
            "  3. OAuth2 → URL Generator → bot + Send Messages + Embed Links\n"
            "  4. Add to .env:\n"
            "       DISCORD_BOT_TOKEN=your_token\n"
            "       DISCORD_CHANNEL_ID=your_channel_id\n"
        )
        return

    if not channel_id:
        logger.error(
            "DISCORD_CHANNEL_ID not set.\n"
            "Right-click your Discord channel → Copy Channel ID\n"
            "(Enable Developer Mode in Discord Settings → Advanced first)\n"
        )
        return

    bot = T1ObserverBot()
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
