"""
Discord presentation layer — decision cards, edge health summary,
settlement updates, and alerts.

Interface only. Does NOT change any betting logic, thresholds, or metrics.
Reads from live_signals, live_bets, clv_log tables.
"""

import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

import discord
from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

DB_PATH = _ROOT / "db" / "lol_model.db"

CLV_ROLLING_WINDOW = 20
BACKTEST_HIT_LOWER_CI = 0.617
PROMOTION_THRESHOLD = 30
BACKTEST_MAX_LOSS_STREAK = 3


# ---------------------------------------------------------------------------
# §1. Per-bet decision cards
# ---------------------------------------------------------------------------
def build_decision_card(
    market_id: str,
    team_a: str,
    team_b: str,
    league: str,
    model_prob: float,
    open_prob: float,
    edge: float,
    gates: Dict[str, bool],
    bet_placed: bool,
    bet_team: Optional[str] = None,
    entry_price: Optional[float] = None,
    stake: Optional[float] = None,
    suppress_reason: Optional[str] = None,
) -> discord.Embed:
    """Build a compact decision card showing the evaluation and gate trace."""

    if bet_placed:
        title = f"✅ BET: {team_a} vs {team_b}"
        color = discord.Color.green()
    else:
        title = f"⏸️ PASS: {team_a} vs {team_b}"
        color = discord.Color.light_grey()

    embed = discord.Embed(title=title, color=color)
    embed.add_field(
        name="Match",
        value=f"**{league}** | {team_a} vs {team_b}",
        inline=False,
    )

    model_pct = f"{model_prob:.0%}"
    market_pct = f"{open_prob:.0%}"
    edge_pct = f"{edge:.1%}"
    embed.add_field(name="Model", value=model_pct, inline=True)
    embed.add_field(name="Market Open", value=market_pct, inline=True)
    embed.add_field(name="Edge", value=f"**{edge_pct}**", inline=True)

    gate_lines = []
    for gate_name, passed in gates.items():
        icon = "✅" if passed else "❌"
        gate_lines.append(f"{icon} {gate_name}")
    embed.add_field(name="Gates", value="\n".join(gate_lines), inline=False)

    if bet_placed and bet_team and entry_price and stake:
        embed.add_field(
            name="Execution",
            value=f"**{bet_team}** @ {entry_price:.0%} | ${stake:.2f}",
            inline=False,
        )
    elif suppress_reason:
        embed.add_field(
            name="Suppressed",
            value=f"**{suppress_reason}**",
            inline=False,
        )

    embed.set_footer(text=f"Paper mode | {market_id[:12]}…")
    return embed


# ---------------------------------------------------------------------------
# §2. Edge health summary
# ---------------------------------------------------------------------------
def build_health_summary() -> discord.Embed:
    """Build the edge-health summary embed."""
    conn = sqlite3.connect(DB_PATH, timeout=10)

    total_bets = conn.execute(
        "SELECT COUNT(*) FROM live_bets WHERE suppressed_reason IS NULL"
    ).fetchone()[0]
    open_bets = conn.execute(
        "SELECT COUNT(*) FROM live_bets WHERE status = 'open' AND suppressed_reason IS NULL"
    ).fetchone()[0]
    won = conn.execute("SELECT COUNT(*) FROM live_bets WHERE status = 'won'").fetchone()[0]
    lost = conn.execute("SELECT COUNT(*) FROM live_bets WHERE status = 'lost'").fetchone()[0]
    suppressed = conn.execute(
        "SELECT COUNT(*) FROM live_bets WHERE suppressed_reason IS NOT NULL"
    ).fetchone()[0]
    signals = conn.execute("SELECT COUNT(*) FROM live_signals").fetchone()[0]

    clv_rows = conn.execute(
        "SELECT clv, beat_close, realized_pnl FROM clv_log ORDER BY id DESC"
    ).fetchall()

    conn.close()

    resolved = won + lost
    hit_rate = won / resolved if resolved > 0 else 0
    total_pnl = sum(r[2] for r in clv_rows if r[2] is not None)

    # Rolling CLV
    recent = clv_rows[:CLV_ROLLING_WINDOW]
    rolling_clv = sum(r[0] for r in recent if r[0] is not None) / max(len(recent), 1) if recent else 0
    pct_beat = sum(1 for r in recent if r[1]) / max(len(recent), 1) if recent else 0

    # Trend
    if len(clv_rows) >= CLV_ROLLING_WINDOW * 2:
        older = clv_rows[CLV_ROLLING_WINDOW:CLV_ROLLING_WINDOW * 2]
        older_clv = sum(r[0] for r in older if r[0] is not None) / max(len(older), 1)
        if rolling_clv > older_clv + 0.01:
            trend = "📈 Rising"
        elif rolling_clv < older_clv - 0.01:
            trend = "📉 Declining"
        else:
            trend = "➡️ Flat"
    else:
        trend = "⏳ Insufficient data"

    # CLV health color
    if rolling_clv > 0.05:
        color = discord.Color.green()
        clv_status = "🟢 Healthy"
    elif rolling_clv > 0:
        color = discord.Color.yellow()
        clv_status = "🟡 Marginal"
    else:
        color = discord.Color.red()
        clv_status = "🔴 WARNING — edge may be dying"

    # Promotion gate
    promotion_pct = min(resolved / PROMOTION_THRESHOLD, 1.0)
    promotion_bar = "█" * int(promotion_pct * 10) + "░" * (10 - int(promotion_pct * 10))
    if resolved >= PROMOTION_THRESHOLD and rolling_clv > 0:
        promotion_text = f"✅ READY — {resolved}/{PROMOTION_THRESHOLD} bets, CLV positive"
    else:
        remaining = max(PROMOTION_THRESHOLD - resolved, 0)
        promotion_text = f"{promotion_bar} {resolved}/{PROMOTION_THRESHOLD} ({remaining} to go)"

    embed = discord.Embed(
        title="📊 Edge Health Dashboard",
        color=color,
    )

    embed.add_field(
        name="Rolling CLV (last 20)",
        value=f"**{rolling_clv:+.3f}** {clv_status}\n{trend}",
        inline=False,
    )
    embed.add_field(
        name="Recent % Beating Close",
        value=f"{pct_beat:.0%}",
        inline=True,
    )
    embed.add_field(
        name="Live Hit Rate",
        value=f"{hit_rate:.0%} (backtest floor: {BACKTEST_HIT_LOWER_CI:.0%})",
        inline=True,
    )
    embed.add_field(
        name="Record",
        value=f"{won}W / {lost}L / {open_bets} open",
        inline=True,
    )
    embed.add_field(
        name="Paper P&L",
        value=f"${total_pnl:+,.2f}",
        inline=True,
    )
    embed.add_field(
        name="Markets Scanned",
        value=f"{signals} detected, {suppressed} suppressed",
        inline=True,
    )
    embed.add_field(
        name="Promotion Gate",
        value=promotion_text,
        inline=False,
    )
    embed.set_footer(text="PAPER MODE | Rolling CLV is the leading indicator")
    return embed


# ---------------------------------------------------------------------------
# §3. Settlement updates
# ---------------------------------------------------------------------------
def build_settlement_card(
    bet_team: str,
    won: bool,
    pnl: float,
    entry_price: float,
    prematch_close: float,
    clv: float,
    rolling_clv: float,
    record_w: int,
    record_l: int,
) -> discord.Embed:
    """Compact settlement card."""
    icon = "✅" if won else "❌"
    color = discord.Color.green() if won else discord.Color.red()
    result = "WIN" if won else "LOSS"

    embed = discord.Embed(
        title=f"{icon} {result}: {bet_team}",
        color=color,
    )
    embed.add_field(name="P&L", value=f"${pnl:+.2f}", inline=True)
    embed.add_field(name="Entry", value=f"{entry_price:.0%}", inline=True)
    embed.add_field(name="Pre-match Close", value=f"{prematch_close:.0%}", inline=True)
    embed.add_field(name="CLV", value=f"{clv:+.3f}", inline=True)
    embed.add_field(name="Rolling CLV", value=f"{rolling_clv:+.3f}", inline=True)
    embed.add_field(name="Record", value=f"{record_w}W / {record_l}L", inline=True)
    embed.set_footer(text="PAPER MODE")
    return embed


# ---------------------------------------------------------------------------
# §4. Alerts
# ---------------------------------------------------------------------------
def check_alerts() -> List[discord.Embed]:
    """Check for alert conditions. Returns list of alert embeds."""
    alerts = []
    conn = sqlite3.connect(DB_PATH, timeout=10)

    # Alert 1: Rolling CLV crossing toward zero
    clv_rows = conn.execute(
        "SELECT clv FROM clv_log ORDER BY id DESC LIMIT ?",
        (CLV_ROLLING_WINDOW,),
    ).fetchall()

    if len(clv_rows) >= 10:
        rolling = sum(r[0] for r in clv_rows if r[0] is not None) / len(clv_rows)
        if rolling <= 0:
            embed = discord.Embed(
                title="🚨 CLV ALERT — Edge may be dying",
                description=(
                    f"Rolling CLV ({CLV_ROLLING_WINDOW} bets): **{rolling:+.3f}**\n"
                    f"This has crossed toward zero. The opening-line edge may be "
                    f"decaying as the market matures. Monitor closely."
                ),
                color=discord.Color.red(),
            )
            alerts.append(embed)

    # Alert 2: Losing streak exceeds backtest max
    recent_bets = conn.execute(
        "SELECT status FROM live_bets WHERE suppressed_reason IS NULL AND status != 'open' ORDER BY id DESC LIMIT 10"
    ).fetchall()
    streak = 0
    for r in recent_bets:
        if r[0] == "lost":
            streak += 1
        else:
            break
    if streak > BACKTEST_MAX_LOSS_STREAK:
        embed = discord.Embed(
            title=f"⚠️ STREAK ALERT — {streak} consecutive losses",
            description=(
                f"Backtest max was {BACKTEST_MAX_LOSS_STREAK}. "
                f"Current streak: **{streak}**. Possible regime change."
            ),
            color=discord.Color.orange(),
        )
        alerts.append(embed)

    # Alert 3: Roster gate suppressions (informational)
    roster_sups = conn.execute(
        "SELECT COUNT(*) FROM live_bets WHERE suppressed_reason LIKE 'roster%' AND entry_ts > datetime('now', '-1 hour')"
    ).fetchone()[0]
    if roster_sups > 0:
        embed = discord.Embed(
            title=f"🔄 Roster gate fired ({roster_sups} in last hour)",
            description="Bets suppressed due to detected roster changes.",
            color=discord.Color.blue(),
        )
        alerts.append(embed)

    conn.close()
    return alerts
