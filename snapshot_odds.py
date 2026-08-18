"""
snapshot_odds.py — capture Pinnacle odds near kickoff (Blueprint Section 28)

WHY THIS EXISTS
---------------
Two things depend on it:

1. **True CLV.** Section 22.8 confirmed FootyStats exposes no opening/closing
   distinction — `odds_comparison` is whatever was last recorded. So CLV
   computed from it is approximate. Capturing the price ourselves shortly
   before kickoff gives a genuine closing line, which Section 13.5 identified
   as the primary scoreboard for whether an edge is real.

2. **The La Liga forward test.** Section 24.3 specified a pre-registered rule:
   *back the La Liga home team where Pinnacle's de-vigged implied probability
   is at least 60%*. Section 24 could not settle it retrospectively — Pinnacle
   coverage collapses for older seasons (5/71) and the one testable season was
   the strongest of four. A forward test avoids both problems, and needs odds
   captured live.

Note this records what the rule *would* have done. It places nothing.

DESIGN
------
Each fixture is snapshotted **once**, as close to kickoff as practical:

- Runs hourly, but only considers fixtures kicking off inside a configurable
  window (default 30-100 minutes out).
- Keeps a record of what it has already captured, so no fixture is snapshotted
  twice — a second capture would cost a call and give a worse (earlier) price.
- Budget: 1 call for the fixture list plus one per new fixture. At ~20 relevant
  fixtures a day that is well inside the confirmed 1800/hour limit.

OUTPUT
------
Appends to `odds_snapshots.csv`, and to a Google Sheet tab if configured.
One row per fixture per snapshot, including whether it qualifies under the
pre-registered rule.

USAGE
-----
    python3 snapshot_odds.py                 # normal run
    python3 snapshot_odds.py --dry-run       # show what it would capture
    python3 snapshot_odds.py --window 30 240 # widen the kickoff window
    python3 snapshot_odds.py --league "Spain La Liga"
"""

import argparse
import csv
import os
from datetime import datetime, timezone

from footystats_client import FootyStatsClient, FootyStatsError
import odds_tools

SNAPSHOT_PATH = 'odds_snapshots.csv'

# The pre-registered rule from Section 24.3. Deliberately a module constant
# rather than a command-line option: a threshold that can be tuned per run is
# not pre-registered, and Sections 11-21 showed repeatedly that adjustable
# thresholds are how retrospective fitting sneaks back in.
FORWARD_TEST_LEAGUE = 'Spain La Liga'
FORWARD_TEST_MIN_IMPLIED = 0.60
FORWARD_TEST_SIDE = 'home'

FIELDNAMES = [
    'snapshot_at_utc', 'match_id', 'league', 'kickoff_utc', 'minutes_to_kickoff',
    'home_team', 'away_team',
    'pin_home', 'pin_draw', 'pin_away',
    'pin_overround_pct', 'pin_implied_home', 'pin_implied_draw', 'pin_implied_away',
    'best_home_book', 'best_home_odds',
    'soft_home', 'soft_draw', 'soft_away',
    'qualifies_forward_test', 'books_quoted',
]


def load_seen():
    """match_ids already snapshotted, so nothing is captured twice."""
    if not os.path.exists(SNAPSHOT_PATH):
        return set()
    with open(SNAPSHOT_PATH, newline='') as f:
        return {row['match_id'] for row in csv.DictReader(f) if row.get('match_id')}


def append_rows(rows):
    exists = os.path.exists(SNAPSHOT_PATH)
    with open(SNAPSHOT_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDNAMES})


def minutes_until(unix_ts, now=None):
    now = now or datetime.now(timezone.utc)
    try:
        ko = datetime.fromtimestamp(int(unix_ts), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None
    return (ko - now).total_seconds() / 60.0


def build_row(match, detail, mins):
    """Assembles one snapshot row. Returns None if Pinnacle isn't quoted."""
    oc = detail.get('odds_comparison')
    if not oc:
        return None

    pin = odds_tools.get_pinnacle(oc)
    if not all(pin.get(k) for k in ('home', 'draw', 'away')):
        # Partial Pinnacle quotes can't be de-vigged, so implied probability —
        # which the whole rule depends on — would be undefined. Skip rather
        # than record something half-usable.
        return None

    ov = odds_tools.overround(pin, warn=True)
    fair = odds_tools.devig(pin)
    best = odds_tools.best_available(oc)
    books = odds_tools.list_bookmakers(oc)

    league = detail.get('league') or match.get('league') or ''
    qualifies = (
        FORWARD_TEST_LEAGUE.lower() in str(league).lower()
        and fair is not None
        and fair[FORWARD_TEST_SIDE] >= FORWARD_TEST_MIN_IMPLIED
    )

    ko = detail.get('date_unix') or match.get('date_unix')
    try:
        ko_str = datetime.fromtimestamp(int(ko), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    except (TypeError, ValueError, OSError):
        ko_str = ''

    return {
        'snapshot_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'),
        'match_id': detail.get('id') or match.get('id'),
        'league': league,
        'kickoff_utc': ko_str,
        'minutes_to_kickoff': f'{mins:.0f}' if mins is not None else '',
        'home_team': detail.get('home_name') or match.get('home_name'),
        'away_team': detail.get('away_name') or match.get('away_name'),
        'pin_home': f"{pin['home']:.3f}",
        'pin_draw': f"{pin['draw']:.3f}",
        'pin_away': f"{pin['away']:.3f}",
        'pin_overround_pct': f'{(ov - 1) * 100:.2f}' if ov else '',
        'pin_implied_home': f"{fair['home']:.4f}" if fair else '',
        'pin_implied_draw': f"{fair['draw']:.4f}" if fair else '',
        'pin_implied_away': f"{fair['away']:.4f}" if fair else '',
        'best_home_book': best['home'][0] or '',
        'best_home_odds': f"{best['home'][1]:.3f}" if best['home'][1] else '',
        'soft_home': detail.get('odds_ft_1', ''),
        'soft_draw': detail.get('odds_ft_x', ''),
        'soft_away': detail.get('odds_ft_2', ''),
        'qualifies_forward_test': 'YES' if qualifies else 'no',
        'books_quoted': len(books),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='Report what would be captured without writing or spending /match calls')
    ap.add_argument('--window', nargs=2, type=int, default=[30, 100],
                    metavar=('MIN', 'MAX'),
                    help='Only snapshot fixtures kicking off this many minutes from now '
                         '(default 30 100)')
    ap.add_argument('--league', default=None,
                    help='Restrict to leagues whose name contains this string')
    ap.add_argument('--no-sheets', action='store_true')
    args = ap.parse_args()

    lo, hi = args.window
    now = datetime.now(timezone.utc)
    print(f"Snapshot run at {now.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Kickoff window: {lo}-{hi} minutes from now\n")

    try:
        fs = FootyStatsClient(use_cache=False, verbose=False)
    except FootyStatsError as e:
        print(f"Cannot start: {e}")
        return

    try:
        fixtures = fs.todays_matches()
    except FootyStatsError as e:
        print(f"Could not fetch today's fixtures: {e}")
        return

    print(f"{len(fixtures)} fixture(s) returned for today.")

    seen = load_seen()
    print(f"{len(seen)} fixture(s) already snapshotted previously.\n")

    candidates = []
    for m in fixtures:
        mid = str(m.get('id'))
        if mid in seen:
            continue
        if args.league and args.league.lower() not in str(m.get('league', '')).lower():
            continue
        mins = minutes_until(m.get('date_unix'), now)
        if mins is None or not (lo <= mins <= hi):
            continue
        candidates.append((m, mins))

    candidates.sort(key=lambda x: x[1])

    if not candidates:
        print("Nothing to snapshot in this window.")
        print("(Normal most of the time — fixtures only qualify in the hour before kickoff.)")
        return

    print(f"{len(candidates)} fixture(s) in window:")
    for m, mins in candidates:
        print(f"  {mins:5.0f} min  {m.get('home_name')} vs {m.get('away_name')}"
              f"  [{m.get('league', '?')}]")

    if args.dry_run:
        print(f"\nDRY RUN — would use {len(candidates)} /match call(s). Nothing written.")
        return

    print(f"\nFetching Pinnacle odds ({len(candidates)} calls)...")
    rows, skipped = [], []
    for m, mins in candidates:
        try:
            detail = fs.match_details(m['id'])
        except FootyStatsError as e:
            skipped.append((m, f'API error: {e}'))
            continue
        if not detail:
            skipped.append((m, 'no detail returned'))
            continue
        row = build_row(m, detail, mins)
        if row is None:
            skipped.append((m, 'Pinnacle not quoted (or partial)'))
            continue
        rows.append(row)

    if rows:
        append_rows(rows)
        print(f"\nCaptured {len(rows)} snapshot(s) -> {SNAPSHOT_PATH}")
        print(f"\n{'fixture':44s} {'Pin H/D/A':22s} {'impl H':>7s} {'ovr':>6s}  rule")
        print('-' * 96)
        for r in rows:
            fixture = f"{r['home_team']} vs {r['away_team']}"[:43]
            pin = f"{r['pin_home']}/{r['pin_draw']}/{r['pin_away']}"
            flag = 'QUALIFIES' if r['qualifies_forward_test'] == 'YES' else ''
            print(f"{fixture:44s} {pin:22s} {float(r['pin_implied_home'])*100:6.1f}% "
                  f"{r['pin_overround_pct']:>5s}%  {flag}")

        qual = [r for r in rows if r['qualifies_forward_test'] == 'YES']
        if qual:
            print(f"\n{len(qual)} fixture(s) qualify for the pre-registered "
                  f"{FORWARD_TEST_LEAGUE} rule "
                  f"(home, Pinnacle implied >= {FORWARD_TEST_MIN_IMPLIED:.0%}).")

        if not args.no_sheets:
            try:
                import sheets_sync
                ws = sheets_sync.connect()
                if ws is not None:
                    sheet = ws.spreadsheet
                    try:
                        tab = sheet.worksheet('OddsSnapshots')
                    except Exception:
                        tab = sheet.add_worksheet(title='OddsSnapshots',
                                                  rows=2000, cols=len(FIELDNAMES))
                        tab.append_row(FIELDNAMES)
                    for r in rows:
                        tab.append_row([str(r.get(k, '')) for k in FIELDNAMES])
                    print(f"Also synced {len(rows)} row(s) to the OddsSnapshots tab.")
                else:
                    print("Google Sheets not configured — local CSV only.")
            except Exception as e:
                print(f"Sheets sync failed (local CSV is intact): {e}")

    if skipped:
        print(f"\n{len(skipped)} skipped:")
        for m, why in skipped:
            print(f"  {m.get('home_name')} vs {m.get('away_name')}: {why}")

    print(f"\n{fs.calls_made} API call(s) used. Budget remaining: {fs.requests_remaining}")


if __name__ == '__main__':
    main()
