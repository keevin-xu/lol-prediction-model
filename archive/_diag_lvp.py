"""Confirm whether LVP SuperLiga has a non-promotion main split. Read-only."""
from roster_scraper import _cargo_query_all, _make_session, login

s = _make_session(); login(s)
rows = _cargo_query_all(s, {
    "tables": "Tournaments",
    "fields": "Name, DateStart, TournamentLevel, IsQualifier, IsPlayoffs",
    "where": "League='LVP SuperLiga'",
    "order_by": "DateStart DESC",
    "limit": "12",
})
print(f"\n=== LVP SuperLiga: {len(rows)} most-recent tournaments (ANY year/level) ===")
for r in rows:
    print(f"  {r.get('DateStart','')[:10]}  lvl={r.get('TournamentLevel',''):10} "
          f"Q={r.get('IsQualifier','0')} PO={r.get('IsPlayoffs','0')}  {r.get('Name','')}")
