"""
Manual odds comparison tool — enter bookmaker odds for matches,
compare against model predictions, track results.

Usage:
  python backtest/manual_odds_entry.py --add         # add a match with odds
  python backtest/manual_odds_entry.py --pending      # show matches awaiting results
  python backtest/manual_odds_entry.py --result       # record a match result
  python backtest/manual_odds_entry.py --report       # show model vs bookmaker report
  python backtest/manual_odds_entry.py --recent       # show recent matches you can look up odds for
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from model.predict import predict_match
from scrapers.odds_scraper import decimal_to_implied, remove_vig

DB_PATH = _ROOT / "db" / "lol_model.db"


def show_recent_matches(days: int = 7, league: Optional[str] = None) -> None:
    """Show recent completed matches the user can look up odds for."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT m.id, m.date, m.league, m.blue_team, m.red_team, m.winner,
               bo.id IS NOT NULL as has_odds
        FROM matches m
        LEFT JOIN bookmaker_odds bo ON bo.match_id = m.id
        WHERE m.date >= date('now', ? || ' days')
    """
    params = [f"-{days}"]
    if league:
        query += " AND m.league = ?"
        params.append(league)
    query += " ORDER BY m.date DESC LIMIT 30"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        print(f"\n  No matches in the last {days} days.")
        return

    print(f"\n  Recent matches (last {days} days):")
    print(f"  {'ID':>6} {'Date':12} {'League':6} {'Blue Team':25} {'Red Team':25} {'Winner':6} {'Odds?'}")
    print(f"  {'-'*90}")
    for mid, date, league, blue, red, winner, has_odds in rows:
        odds_str = "YES" if has_odds else ""
        print(f"  {mid:6} {date[:10]:12} {league:6} {blue:25} {red:25} {winner:6} {odds_str}")

    print(f"\n  To add odds: python backtest/manual_odds_entry.py --add")
    print(f"  You'll need: match ID, decimal odds for each team (from Pinnacle/bookmaker)")


def add_odds() -> None:
    """Interactive: add bookmaker odds for a match."""
    print("\n  Add bookmaker odds for a match")
    print("  (Use --recent to find match IDs)\n")

    try:
        match_id = int(input("  Match ID: ").strip())
    except (ValueError, EOFError):
        print("  Invalid match ID")
        return

    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT date, league, blue_team, red_team, winner FROM matches WHERE id = ?",
        (match_id,),
    ).fetchone()

    if not row:
        print(f"  Match {match_id} not found")
        conn.close()
        return

    date, league, blue, red, winner = row
    print(f"\n  Match: {blue} vs {red}")
    print(f"  Date: {date[:10]} | League: {league} | Winner: {winner}")

    existing = conn.execute(
        "SELECT id FROM bookmaker_odds WHERE match_id = ?", (match_id,)
    ).fetchone()
    if existing:
        print(f"  Already has odds recorded. Skipping.")
        conn.close()
        return

    try:
        odds_a = float(input(f"\n  Decimal odds for {blue}: ").strip())
        odds_b = float(input(f"  Decimal odds for {red}: ").strip())
        source = input("  Source (e.g., pinnacle, bet365): ").strip() or "pinnacle"
    except (ValueError, EOFError):
        print("  Invalid input")
        conn.close()
        return

    implied_a = decimal_to_implied(odds_a)
    implied_b = decimal_to_implied(odds_b)
    no_vig_a, no_vig_b = remove_vig(implied_a, implied_b)

    winner_db = blue if winner == "blue" else red

    conn.execute(
        """
        INSERT INTO bookmaker_odds
            (source, match_date, league, team_a_raw, team_b_raw,
             team_a_db, team_b_db, odds_a, odds_b,
             implied_prob_a, implied_prob_b, no_vig_prob_a, no_vig_prob_b,
             winner_raw, winner_db, match_id, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (
            source, date[:10], league, blue, red, blue, red,
            odds_a, odds_b,
            round(implied_a, 4), round(implied_b, 4),
            round(no_vig_a, 4), round(no_vig_b, 4),
            winner if winner in ("blue", "red") else None,
            winner_db, match_id,
        ),
    )
    conn.commit()
    conn.close()

    print(f"\n  Saved: {blue} ({odds_a}) vs {red} ({odds_b})")
    print(f"  Implied: {implied_a:.1%} / {implied_b:.1%}")
    print(f"  No-vig:  {no_vig_a:.1%} / {no_vig_b:.1%}")


def show_pending() -> None:
    """Show matches with odds but no result comparison yet."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT bo.id, m.date, m.league, m.blue_team, m.red_team,
               bo.no_vig_prob_a, m.winner
        FROM bookmaker_odds bo
        JOIN matches m ON bo.match_id = m.id
        ORDER BY m.date DESC
    """).fetchall()
    conn.close()

    if not rows:
        print("\n  No odds recorded yet. Use --add to enter some.")
        return

    print(f"\n  Matches with bookmaker odds ({len(rows)}):")
    print(f"  {'Date':12} {'League':6} {'Blue':25} {'Red':25} {'Book Blue':>10} {'Winner':>7}")
    print(f"  {'-'*90}")
    for _, date, league, blue, red, book_prob, winner in rows:
        print(f"  {date[:10]:12} {league:6} {blue:25} {red:25} {book_prob:10.1%} {winner:>7}")


def run_report() -> None:
    """Compare model predictions vs bookmaker odds on all recorded matches."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT m.id, m.date, m.league, m.blue_team, m.red_team, m.winner,
               bo.no_vig_prob_a, bo.no_vig_prob_b, bo.source
        FROM bookmaker_odds bo
        JOIN matches m ON bo.match_id = m.id
        WHERE bo.no_vig_prob_a IS NOT NULL
        ORDER BY m.date ASC
    """).fetchall()
    conn.close()

    if not rows:
        print("\n  No odds data to compare. Use --add to enter bookmaker odds.")
        return

    model_correct = 0
    book_correct = 0
    model_brier_sum = 0.0
    book_brier_sum = 0.0
    n = 0

    details = []
    for mid, date, league, blue, red, winner, book_a, book_b, source in rows:
        result = predict_match(blue, red)
        model_a = result["p_a"]
        actual = 1.0 if winner == "blue" else 0.0

        m_correct = (model_a >= 0.5) == (actual == 1.0)
        b_correct = (book_a >= 0.5) == (actual == 1.0)

        model_brier = (model_a - actual) ** 2
        book_brier = (book_a - actual) ** 2

        model_correct += int(m_correct)
        book_correct += int(b_correct)
        model_brier_sum += model_brier
        book_brier_sum += book_brier
        n += 1

        m_icon = "+" if m_correct else "-"
        b_icon = "+" if b_correct else "-"
        details.append((date, league, blue, red, winner, model_a, book_a, m_icon, b_icon))

    print(f"\n{'='*75}")
    print(f"  MODEL vs BOOKMAKER ({source})")
    print(f"{'='*75}")
    print(f"  Matches: {n}")
    print()
    print(f"  {'':20} {'Model':>10} {'Bookmaker':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Accuracy':20} {model_correct/n:10.1%} {book_correct/n:10.1%}")
    print(f"  {'Brier Score':20} {model_brier_sum/n:10.4f} {book_brier_sum/n:10.4f}")

    better = "MODEL" if model_brier_sum < book_brier_sum else "BOOKMAKER"
    print(f"\n  Winner: {better} (lower Brier = better)")

    print(f"\n  {'Date':12} {'Lg':5} {'Blue':22} {'Red':22} {'Win':5} {'Model':>7} {'Book':>7} {'M':>2} {'B':>2}")
    print(f"  {'-'*90}")
    for date, lg, blue, red, winner, model_a, book_a, mi, bi in details:
        w = "B" if winner == "blue" else "R"
        print(f"  {date[:10]:12} {lg:5} {blue:22} {red:22} {w:5} {model_a:7.1%} {book_a:7.1%} {mi:>2} {bi:>2}")

    # Edge analysis
    edge_wins = 0
    edge_total = 0
    for date, lg, blue, red, winner, model_a, book_a, mi, bi in details:
        edge = abs(model_a - book_a)
        if edge >= 0.03:
            actual = 1.0 if winner == "blue" else 0.0
            bet_blue = model_a > book_a
            won = (bet_blue and actual == 1.0) or (not bet_blue and actual == 0.0)
            edge_total += 1
            if won:
                edge_wins += 1

    if edge_total > 0:
        print(f"\n  Edge bets (model disagrees with book by 3%+):")
        print(f"    {edge_wins}/{edge_total} won ({edge_wins/edge_total:.0%})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual bookmaker odds comparison")
    parser.add_argument("--recent", action="store_true", help="Show recent matches to look up odds for")
    parser.add_argument("--days", type=int, default=7, help="Days of recent matches (default: 7)")
    parser.add_argument("--league", type=str, default=None, help="Filter by league")
    parser.add_argument("--add", action="store_true", help="Add bookmaker odds for a match")
    parser.add_argument("--pending", action="store_true", help="Show matches with odds recorded")
    parser.add_argument("--report", action="store_true", help="Run model vs bookmaker comparison")
    args = parser.parse_args()

    if args.recent:
        show_recent_matches(days=args.days, league=args.league)
    elif args.add:
        add_odds()
    elif args.pending:
        show_pending()
    elif args.report:
        run_report()
    else:
        parser.print_help()
        print("\nWorkflow:")
        print("  1. python backtest/manual_odds_entry.py --recent          # find matches")
        print("  2. Look up Pinnacle closing odds on pinnacle.com or oddsportal.com")
        print("  3. python backtest/manual_odds_entry.py --add             # enter odds")
        print("  4. Repeat for 20+ matches")
        print("  5. python backtest/manual_odds_entry.py --report          # see who wins")


if __name__ == "__main__":
    main()
