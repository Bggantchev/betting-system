"""
log_bet.py — Bet log (Blueprint Section 16.3, stage 2)

Logs a bet you've actually placed, pulling the composite score and any
outlier flags automatically from the same scoring logic as match_scorer.py
(no re-typing, no risk of the logged score drifting from what the model
actually said). Appends one row per bet to bet_log.csv, which lives
permanently on your own machine — see the note in the chat about why
this is local-first rather than living in the deployed web app.

USAGE
-----
    python3 log_bet.py

Runs as an interactive prompt — it'll ask for the fixture, then pull the
model's score automatically, then ask you to confirm the bet details
(side, stake, odds). Nothing is logged without your final confirmation.

This file expects the same folder layout as match_scorer.py:
    betting_system/
    ├── match_scorer.py
    ├── log_bet.py          <- this file
    ├── bet_log.csv         <- created automatically on first use
    └── data/
        └── (the 8 league CSVs)
"""

import csv
import os
from datetime import datetime

from match_scorer import analyze_fixture, LEAGUE_CONFIG
import sheets_sync

LOG_PATH = 'bet_log.csv'

FIELDNAMES = [
    'bet_id', 'logged_at', 'league', 'home_team', 'away_team',
    'match_id',                      # FootyStats fixture ID — see find_match_id()
    'side_bet', 'composite_score', 'side_favored_by_model',
    'outlier_flags', 'confidence_note',
    'stake', 'odds_at_bet', 'closing_odds', 'clv_pct',
    'result', 'profit_loss', 'notes',
]


def _next_bet_id():
    """bet_id is just a running count of rows already logged, +1."""
    if not os.path.exists(LOG_PATH):
        return 1
    with open(LOG_PATH, newline='') as f:
        return sum(1 for _ in csv.DictReader(f)) + 1


def find_match_id(league, home_team, away_team, days_ahead=7, verbose=True):
    """
    Look up the FootyStats fixture ID for a match about to be bet on.

    Storing this at logging time makes settlement exact. Without it,
    settle_bets.py has to match on team names, which works but relies on an
    alias table and refuses to settle anything ambiguous — so a fixture with an
    unrecognised name variant sits unsettled until fixed by hand.

    Returns (match_id, description) or (None, reason). Never raises: if the API
    is unavailable or unconfigured, logging must still proceed.
    """
    try:
        from footystats_client import FootyStatsClient, FootyStatsError
        from settle_bets import name_similarity
    except ImportError as e:
        return None, f"lookup unavailable ({e})"

    try:
        fs = FootyStatsClient(verbose=False)
    except Exception as e:
        return None, f"API not configured ({e})"

    # Scan today plus the next few days, since bets are often placed in advance.
    from datetime import datetime, timedelta, timezone
    today = datetime.now(timezone.utc).date()
    candidates = []
    try:
        for offset in range(days_ahead + 1):
            day = (today + timedelta(days=offset)).strftime('%Y-%m-%d')
            for m in fs.todays_matches(date=day):
                h = name_similarity(home_team, m.get('home_name'))
                a = name_similarity(away_team, m.get('away_name'))
                if h > 0 and a > 0:
                    candidates.append((h + a, day, m))
            if candidates:
                break        # stop at the first day with a plausible match
    except Exception as e:
        return None, f"lookup failed ({e})"

    if not candidates:
        return None, f"no fixture found in the next {days_ahead} days"

    candidates.sort(key=lambda x: -x[0])
    score, day, best = candidates[0]

    # Refuse to guess between equally plausible fixtures.
    if len(candidates) > 1 and candidates[1][0] == score:
        return None, f"{len(candidates)} equally plausible fixtures — not guessing"

    desc = (f"{best.get('home_name')} vs {best.get('away_name')} "
            f"({best.get('league', '?')}) on {day}")
    confidence = 'exact names' if score == 2.0 else f'partial match (score {score})'
    return best.get('id'), f"{desc} [{confidence}]"


def log_new_bet(league, home_team, away_team, side_bet, stake, odds_at_bet,
                notes='', data_dir='data', match_id=''):
    """
    Core logging function — also callable directly (not just via the
    interactive prompt below) if you ever want to log bets from another
    script, e.g. a future automated pipeline.

    Writes to the local CSV first (always succeeds if the fixture scores
    OK), then tries to sync the same row to Google Sheets. A Sheets
    failure never blocks the local log — you'll just see a note that it
    didn't sync this time, and can sync later (see sync_pending_rows()).
    """
    scoring = analyze_fixture(league, home_team, away_team, data_dir=data_dir)
    flags_text = '; '.join(f['name'] for f in scoring['outlier_flags']) if scoring['outlier_flags'] else ''

    row = {
        'bet_id': _next_bet_id(),
        'logged_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'league': scoring['league'],
        'home_team': home_team,
        'away_team': away_team,
        'match_id': match_id or '',
        'side_bet': side_bet,
        'composite_score': scoring['composite_score'],
        'side_favored_by_model': scoring['side_favored'],
        'outlier_flags': flags_text,
        'confidence_note': scoring['confidence_note'],
        'stake': stake,
        'odds_at_bet': odds_at_bet,
        'closing_odds': '',      # filled in later, near kickoff (Section 16.3 stage 3/4)
        'clv_pct': '',           # calculated once closing_odds is known
        'result': 'Pending',     # updated after the match (Won / Lost / Push)
        'profit_loss': '',       # calculated once result is known
        'notes': notes,
    }

    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    synced = sheets_sync.append_bet_row(row)
    row['_synced_to_sheets'] = synced  # not written to CSV, just returned for the caller's info

    return row


def load_log():
    """Read the current bet log as a list of dicts. Empty list if none logged yet."""
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, newline='') as f:
        return list(csv.DictReader(f))


def _summarize(rows, source_label):
    if not rows:
        print(f"No bets logged yet ({source_label}).")
        return

    pending = [r for r in rows if r['result'] == 'Pending']
    settled = [r for r in rows if r['result'] != 'Pending']
    total_staked = sum(float(r['stake']) for r in rows if r.get('stake') not in ('', None))

    print("=" * 60)
    print(f"BET LOG SUMMARY — {len(rows)} bet(s) logged  [source: {source_label}]")
    print("=" * 60)
    print(f"Pending (awaiting result): {len(pending)}")
    print(f"Settled:                   {len(settled)}")
    print(f"Total staked (all-time):   {total_staked:.2f}")

    if settled:
        won = [r for r in settled if r['result'] == 'Won']
        lost = [r for r in settled if r['result'] == 'Lost']
        clv_vals = [float(r['clv_pct']) for r in settled if r.get('clv_pct') not in ('', None)]
        pl_vals = [float(r['profit_loss']) for r in settled if r.get('profit_loss') not in ('', None)]
        print(f"Record so far:             {len(won)}W-{len(lost)}L")
        if pl_vals:
            print(f"Total profit/loss:         {sum(pl_vals):+.2f}")
        if clv_vals:
            avg_clv = sum(clv_vals) / len(clv_vals)
            print(f"Average CLV:               {avg_clv:+.2f}%  "
                  f"({'positive — the encouraging sign from Section 13.5/16.3' if avg_clv > 0 else 'not yet positive'})")
        else:
            print("Average CLV:               not yet available (closing odds not filled in)")

    if pending:
        print("\nPending bets:")
        for r in pending:
            print(f"  #{r['bet_id']}  {r['home_team']} vs {r['away_team']} "
                  f"({r['league']})  — {r['side_bet']} @ {r['odds_at_bet']}, staked {r['stake']}")
    print("=" * 60)


def print_summary():
    """
    Shows the log summary. Prefers the live Google Sheet if it's
    reachable — that way, edits made directly in the Sheet (e.g. filling
    in a result from your phone after a match) show up immediately,
    without needing to touch the local CSV at all. Falls back to the
    local CSV if Sheets isn't configured or isn't reachable right now.
    """
    sheet_rows = sheets_sync.read_all_bets()
    if sheet_rows is not None:
        _summarize(sheet_rows, source_label='Google Sheets (live)')
    else:
        print("(Google Sheets not reachable — showing local CSV instead. "
              "See SETUP_GOOGLE_SHEETS.md if you haven't connected it yet.)\n")
        _summarize(load_log(), source_label='local CSV')


def sync_all_local_rows():
    """
    Manual recovery: pushes every row currently in the local CSV up to
    Google Sheets. Useful if some bets were logged while offline, or
    before Sheets was set up, and now need to be caught up.
    Skips rows that fail individually rather than stopping the batch.
    """
    rows = load_log()
    if not rows:
        print("No local rows to sync.")
        return
    synced, failed = 0, 0
    for row in rows:
        ok = sheets_sync.append_bet_row(row)
        if ok:
            synced += 1
        else:
            failed += 1
    print(f"Sync complete: {synced} row(s) pushed, {failed} failed "
          f"(check your Google Sheets setup if any failed).")


def _prompt(label, cast=str, default=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{label}{suffix}: ").strip()
        if raw == '' and default is not None:
            return default
        try:
            return cast(raw)
        except ValueError:
            print(f"  Please enter a valid {cast.__name__}.")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--summary':
        print_summary()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == '--sync':
        sync_all_local_rows()
        sys.exit(0)

    print("\n=== Log a new bet ===")
    print("(This pulls the model's score automatically — you just confirm what you actually bet.)\n")

    print(f"Available league codes: {list(LEAGUE_CONFIG.keys())}")
    league = _prompt("League code")
    home_team = _prompt("Home team (exact name as in the CSV)")
    away_team = _prompt("Away team (exact name as in the CSV)")

    try:
        scoring = analyze_fixture(league, home_team, away_team, data_dir='data')
    except (ValueError, FileNotFoundError) as e:
        print(f"\nCouldn't score this fixture: {e}")
        sys.exit(1)

    print(f"\nModel says: composite {scoring['composite_score']:+.3f} -> favors {scoring['side_favored']}")
    print(f"Confidence note: {scoring['confidence_note']}")
    if scoring['outlier_flags']:
        for flag in scoring['outlier_flags']:
            print(f"OUTLIER FLAG: {flag['name']} — {flag['status']}")

    # Look up the FootyStats fixture ID so settlement can be exact rather than
    # name-matched. Entirely optional — if this fails, logging continues and
    # settlement falls back to name matching.
    print()
    print("Looking up the FootyStats fixture ID...")
    match_id, lookup_note = find_match_id(league, home_team, away_team)
    if match_id:
        print(f"  Found: {lookup_note}")
        print(f"  match_id = {match_id}")
        keep = _prompt("  Is that the right fixture? (y/n)", default='y')
        if keep.lower() != 'y':
            match_id = ''
            print("  Discarded — settlement will fall back to name matching.")
    else:
        print(f"  Not found: {lookup_note}")
        print("  Continuing without it; settlement will use name matching.")
        match_id = ''

    print()
    side_bet = _prompt("Which side did you actually bet? (Home/Draw/Away)")
    stake = _prompt("Stake", cast=float)
    odds_at_bet = _prompt("Odds you got", cast=float)
    notes = _prompt("Notes (e.g. which research phase led to this, or blank)", default='')

    id_note = f", match_id={match_id}" if match_id else " (no match_id)"
    print(f"\nAbout to log: {home_team} vs {away_team} — {side_bet} @ {odds_at_bet}, "
          f"stake {stake}{id_note}")
    confirm = _prompt("Confirm? (y/n)", default='y')
    if confirm.lower() != 'y':
        print("Not logged.")
        sys.exit(0)

    row = log_new_bet(league, home_team, away_team, side_bet, stake, odds_at_bet,
                      notes, match_id=match_id)
    sync_note = "also synced to Google Sheets" if row['_synced_to_sheets'] else \
                "NOT synced to Sheets (saved locally only — run `python3 log_bet.py --sync` later to catch up)"
    print(f"\nLogged as bet #{row['bet_id']} ({sync_note}).")
    print("Run `python3 log_bet.py --summary` any time to review the log.")
