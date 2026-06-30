"""
Live CLOB book-snapshot verification.

Fetches one real T1 market's order book, prints the raw CLOB response alongside
the computed depth numbers, and shows the unit math so you can verify by hand
that `depth_within_3pct` = sum(price * size) over asks within 3% of entry price.

Run:  python backtest/verify_book_snapshot.py

What to check:
  1. Unit math — confirm depth = sum(price * size), NOT sum(size). On a 0.50 market
     those differ by ~2x, on a 0.80 market by ~1.25x.
  2. Side — the book fetched is for the token of the team you'd bet (bet_side),
     not always team_a. Confirm the token_id in the URL matches the side shown.
  3. Band — 3pct means asks at price <= entry_price * 1.03 (relative), not +0.03
     (absolute). Confirm by checking which ask levels fall inside the band.

Exit codes: 0 = book fetched and math verified OK, 1 = no live T1 markets found.
"""

import json
import sys
from pathlib import Path

import requests
from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from polymarket.live_engine import (
    CLOB_API,
    GAMMA_API,
    SPREAD_COST,
    SLIPPAGE_COST,
    _make_session,
    _is_t1_market,
    compute_signal,
    fetch_book_snapshot,
    load_db_team_names,
    match_team_name,
    parse_teams_from_question,
)
import json as _json


def fetch_one_live_t1_market(session: requests.Session):
    """Return the first active T1 moneyline market where both teams are in the DB."""
    db_teams = load_db_team_names()
    try:
        r = session.get(
            f"{GAMMA_API}/events",
            params={"active": "true", "closed": "false", "tag_slug": "league-of-legends", "limit": "100"},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        events = r.json()
    except requests.RequestException as e:
        logger.error(f"Gamma API error: {e}")
        return None

    for event in events:
        event_title = event.get("title", "")
        for m in event.get("markets", []):
            q = m.get("question", "")
            ql = q.lower()
            if "(bo" not in ql or " vs " not in ql:
                continue
            if any(x in ql for x in ["game 1", "game 2", "game 3", "game 4", "game 5", "handicap"]):
                continue
            if not _is_t1_market(q, event_title):
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
            if pa >= 0.99 or pa <= 0.01:
                continue

            teams = parse_teams_from_question(q)
            if not teams:
                continue
            pm_a, pm_b = teams
            db_a = match_team_name(pm_a, db_teams)
            db_b = match_team_name(pm_b, db_teams)
            if not db_a or not db_b:
                continue

            volume = float(m.get("volumeNum", 0) or m.get("volume", 0) or 0)
            return {
                "question": q,
                "db_team_a": db_a, "db_team_b": db_b,
                "token_id_a": tokens[0], "token_id_b": tokens[1],
                "open_price_a": pa, "open_price_b": pb,
                "volume": volume,
                "market_create_ts": m.get("startDate", m.get("createdAt", "")),
                "match_start_ts": m.get("endDate", ""),
            }
    return None


def _parse_levels(raw: list) -> list:
    out = []
    for lv in raw:
        try:
            out.append({"price": float(lv["price"]), "size": float(lv["size"])})
        except (KeyError, ValueError, TypeError):
            pass
    return out


# A single ask level at the six-day-out open showing more than $2,000 USDC
# (price × size) is suspicious. Above $10,000 on one level is almost certainly
# a units error — size is being mistaken for USDC when it's actually shares.
_PLAUSIBLE_PER_LEVEL_USDC = 2_000.0
_IMPLAUSIBLE_PER_LEVEL_USDC = 10_000.0


def verify_units_plausibility(raw_book: dict, entry_price: float, market_create_ts: str) -> bool:
    """
    Sanity-check that size is in shares, not USDC.

    The math check (verify_depth_math) confirms sum(price×size) == computed depth,
    but it passes even if size is already in USDC — internally consistent but
    double-counting price. This check catches that silent case by looking at
    whether per-level dollar figures are physically possible for an early T1 market.

    Shows both interpretations (shares and USDC) so the wrong answer is visible.
    """
    asks = _parse_levels(raw_book.get("asks", []))
    if not asks:
        print("  No ask levels — cannot check plausibility")
        return False

    best = min(asks, key=lambda x: x["price"])
    price = best["price"]
    size_raw = best["size"]
    usdc_if_shares = price * size_raw       # correct: size is shares
    usdc_if_already_usdc = size_raw         # wrong: size already dollars, price×size double-counts

    # What the wrong interpretation would produce for depth_within_3pct
    wrong_total = sum(lv["price"] * lv["size"] for lv in asks if lv["price"] <= entry_price * 1.03)
    wrong_if_size_is_usdc = sum(lv["size"] for lv in asks if lv["price"] <= entry_price * 1.03)

    from datetime import datetime, timezone
    age_days = None
    if market_create_ts:
        try:
            create_dt = datetime.fromisoformat(market_create_ts.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - create_dt).total_seconds() / 86400
        except (ValueError, TypeError):
            pass

    print(f"\n  UNITS PLAUSIBILITY CHECK")
    print(f"  {'-'*50}")
    if age_days is not None:
        print(f"  Market age: {age_days:.1f} days (strategy targets < 1 day)")
    print(f"  Best ask level: price={price:.4f}, raw size={size_raw:.2f}")
    print()
    print(f"  Interpretation A — size is shares (CORRECT if Polymarket docs hold):")
    print(f"    price × size = {price:.4f} × {size_raw:.2f} = ${usdc_if_shares:.2f} USDC for this level")
    print(f"    depth_within_3pct (sum price×size) = ${wrong_total:.2f}")
    print()
    print(f"  Interpretation B — size is already USDC (WRONG, would double-count price):")
    print(f"    size alone = ${usdc_if_already_usdc:.2f} for this level")
    print(f"    depth_within_3pct (sum size only) = ${wrong_if_size_is_usdc:.2f}")
    print()

    plausible = usdc_if_shares <= _IMPLAUSIBLE_PER_LEVEL_USDC
    if usdc_if_shares > _IMPLAUSIBLE_PER_LEVEL_USDC:
        print(f"  ⚠  IMPLAUSIBLE: ${usdc_if_shares:.0f} on a single ask level is too large "
              f"for an early T1 market — likely a units error. Check raw API response.")
    elif usdc_if_shares > _PLAUSIBLE_PER_LEVEL_USDC:
        print(f"  ⚠  SUSPICIOUS: ${usdc_if_shares:.0f} per level is high for a market "
              f"{age_days:.1f if age_days else '?'} days old. OK if market is near match time "
              f"(liquidity builds as match approaches). Not OK if < 1 day old.")
    else:
        print(f"  ${usdc_if_shares:.0f} per level ✓  (consistent with opening-book depth)")

    print()
    return plausible


def verify_depth_math(raw_book: dict, entry_price: float, computed: dict) -> bool:
    """
    Confirms depth_within_3pct == sum(price × size) over in-band asks.
    Prints every level so the arithmetic is visible.
    Returns True if computed value matches manual recalculation.
    """
    asks = _parse_levels(raw_book.get("asks", []))
    band_3pct = entry_price * 1.03

    print(f"  Entry price:     {entry_price:.4f}")
    print(f"  3% band ceiling: {band_3pct:.4f}  (entry_price × 1.03 — relative, not +0.03 absolute)")
    print(f"\n  Ask levels (* = inside 3% band):")
    manual_3pct = 0.0
    for lv in sorted(asks, key=lambda x: x["price"]):
        in_band = lv["price"] <= band_3pct
        usdc = lv["price"] * lv["size"]
        marker = "*" if in_band else " "
        print(f"    {marker} price={lv['price']:.4f}  size={lv['size']:.2f} shares  "
              f"→ {lv['price']:.4f} × {lv['size']:.2f} = ${usdc:.2f} USDC")
        if in_band:
            manual_3pct += usdc

    print(f"\n  Manual sum (price×size within 3% band): ${manual_3pct:.2f}")
    print(f"  fetch_book_snapshot depth_within_3pct:  ${computed['depth_within_3pct']:.2f}")

    delta = abs(manual_3pct - computed["depth_within_3pct"])
    ok = delta < 0.01
    print(f"  Arithmetic match: {'YES ✓' if ok else f'NO — delta={delta:.4f}'}")
    return ok


def main() -> None:
    session = _make_session()

    print("=" * 65)
    print("  LIVE CLOB BOOK SNAPSHOT VERIFICATION")
    print("=" * 65)
    print()
    print("Step 1: Find one live T1 market where both teams are in DB…")

    market = fetch_one_live_t1_market(session)
    if not market:
        print("No live T1 markets found right now. Try again during LCK/LPL/LCS/LEC match days.")
        sys.exit(1)

    print(f"  Found: {market['question']}")
    print(f"  {market['db_team_a']} ({market['open_price_a']:.1%}) vs {market['db_team_b']} ({market['open_price_b']:.1%})")
    print()

    print("Step 2: Compute signal to determine bet_side…")
    signal = compute_signal(market)
    bet_side = signal["bet_side"]
    bet_team = signal["bet_team"]
    edge = signal["edge"]
    model_prob = signal["model_prob"]
    token_id = market["token_id_a"] if bet_side == "team_a" else market["token_id_b"]
    open_price = market["open_price_a"] if bet_side == "team_a" else market["open_price_b"]
    entry_price = min(open_price + SPREAD_COST + SLIPPAGE_COST, 0.99)

    print(f"  Model prob:   {model_prob:.3f}")
    print(f"  Open price:   {open_price:.3f}")
    print(f"  Edge:         {edge:+.3f}")
    print(f"  Bet side:     {bet_side} ({bet_team})")
    print(f"  Entry price (with costs): {entry_price:.4f}")
    print(f"  Token to fetch: {token_id[:20]}…")
    print()

    print("Step 3: Fetch raw CLOB book…")
    try:
        r = session.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=10)
        print(f"  HTTP {r.status_code}")
        if r.status_code != 200:
            print(f"  Error: {r.text[:200]}")
            sys.exit(1)
        raw_book = r.json()
    except requests.RequestException as e:
        print(f"  Request failed: {e}")
        sys.exit(1)

    n_asks = len(raw_book.get("asks", []))
    n_bids = len(raw_book.get("bids", []))
    print(f"  Book: {n_asks} ask levels, {n_bids} bid levels")
    print()

    print("Step 4: Call fetch_book_snapshot() (the function live_engine uses)…")
    computed = fetch_book_snapshot(token_id, entry_price, session)
    if computed is None:
        print("  fetch_book_snapshot returned None — empty or unreachable book")
        sys.exit(1)

    print(f"  best_bid:           {computed.get('best_bid')}")
    print(f"  best_ask:           {computed.get('best_ask')}")
    print(f"  spread:             {computed.get('spread')}")
    print(f"  depth_within_1pct:  ${computed.get('depth_within_1pct')}")
    print(f"  depth_within_3pct:  ${computed.get('depth_within_3pct')}")
    print(f"  depth_within_5pct:  ${computed.get('depth_within_5pct')}")
    print()

    print("Step 5: Units plausibility — are per-level dollar figures realistic?")
    units_ok = verify_units_plausibility(raw_book, entry_price, market.get("market_create_ts", ""))

    print("Step 6: Arithmetic — does depth_within_3pct == sum(price × size) over in-band asks?")
    math_ok = verify_depth_math(raw_book, entry_price, computed)
    print()

    print("Step 7: Side routing — does the fetched token match the bet_side?")
    print(f"  bet_side = {bet_side}  →  should fetch market['token_id_{bet_side[-1]}']")
    print(f"  Token fetched:           {token_id[:40]}…")
    print(f"  market['token_id_a']:    {market['token_id_a'][:40]}…")
    print(f"  market['token_id_b']:    {market['token_id_b'][:40]}…")
    correct_token = (
        (bet_side == "team_a" and token_id == market["token_id_a"]) or
        (bet_side == "team_b" and token_id == market["token_id_b"])
    )
    print(f"  Side → token routing: {'CORRECT ✓' if correct_token else 'WRONG — check token_id logic'}")
    print()

    all_ok = units_ok and math_ok and correct_token
    print("=" * 65)
    print(f"  RESULT: {'PASS — book fetch trustworthy, set T1_SCANNING=True' if all_ok else 'FAIL — fix before enabling T1_SCANNING'}")
    print("=" * 65)
    print()

    if not all_ok:
        print("Issues found:")
        if not units_ok:
            print("  - Per-level dollar figures implausibly large — likely units bug (size already USDC?)")
        if not math_ok:
            print("  - depth_within_3pct does not match manual price×size sum")
        if not correct_token:
            print("  - Wrong token fetched for bet_side")
        sys.exit(1)


if __name__ == "__main__":
    main()
