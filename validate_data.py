"""
validate_data.py — data plausibility checks (Blueprint Section 23.4)

WHY THIS EXISTS
---------------
Section 23 found that 33% of Pinnacle closing-odds rows across the project's
data files were corrupt — misaligned columns producing impossible prices, e.g.
Real Madrid at home priced 10.00. Belgium and Primeira Liga were over 90% wrong.

Every one of those files had been through roughly twenty analyses with rigorous
statistical validation — train/test splits, leave-one-out, bootstrap CIs — and
none of it caught the problem, because the statistics were never the issue. The
data was.

A single check would have caught it: does this bookmaker price between 1.00 and
1.06 overround? That check is this file.

USAGE
-----
Command line, to audit files:

    python3 validate_data.py                    # check every CSV in data/
    python3 validate_data.py --file data/SP1_raw.csv
    python3 validate_data.py --strict           # exit code 1 if anything fails

In code, before analysing anything:

    from validate_data import assert_odds_sane, filter_to_clean
    df = filter_to_clean(df, 'PSCH', 'PSCD', 'PSCA')   # drop bad rows
"""

import argparse
import glob
import os
import sys

import pandas as pd


# A real 1X2 book never prices below 1.00 overround (that is a standing
# arbitrage) and rarely above 1.15. Anything outside 1.005-1.06 for a
# sharp book, or 1.005-1.12 generally, means the columns are misaligned.
SHARP_BAND = (1.005, 1.06)      # Pinnacle-style
GENERAL_BAND = (1.005, 1.15)    # any bookmaker

# The book-profile check compares a column's mean margin against what that
# bookmaker should charge. Below this many rows the mean is too noisy to judge,
# so the check is skipped rather than producing false alarms (early-season
# files routinely have only 8-12 matches).
MIN_ROWS_FOR_PROFILE = 30

# Soft bookmakers charge more on smaller leagues: Bet365 runs ~5.5% on the
# Premier League but 7.5-10% on the Belgian Pro League or Primeira Liga. A 9%
# ceiling calibrated on major leagues produced false alarms on smaller ones.
SOFT_BOOK_MAX_MARGIN = 12.0

# Odds column triplets commonly present in this project's files.
# Each entry: (home, draw, away, label, plausibility band, expected margin range)
#
# The expected margin range is the SECOND check, and it catches something the
# band cannot. A column can contain perfectly plausible odds that simply are not
# the book you think they are — e.g. a column labelled Pinnacle containing
# Bet365 values. Pinnacle runs ~2-4% on major-league 1X2; a soft book runs 5-8%.
# If a "Pinnacle" column's clean rows average 5.5%, it is not Pinnacle,
# regardless of how plausible each individual row looks.
KNOWN_TRIPLETS = [
    ('PSCH', 'PSCD', 'PSCA', 'Pinnacle closing', SHARP_BAND, (1.5, 4.5)),
    ('PSH',  'PSD',  'PSA',  'Pinnacle opening', SHARP_BAND, (1.5, 5.0)),
    ('MaxCH', 'MaxCD', 'MaxCA', 'Market max closing', (0.95, 1.10), (0.0, 4.5)),
    ('AvgCH', 'AvgCD', 'AvgCA', 'Market avg closing', GENERAL_BAND, (3.0, SOFT_BOOK_MAX_MARGIN)),
    ('B365H', 'B365D', 'B365A', 'Bet365', GENERAL_BAND, (3.0, SOFT_BOOK_MAX_MARGIN)),
    ('odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win',
     'FootyStats CSV', GENERAL_BAND, (3.0, SOFT_BOOK_MAX_MARGIN)),
    ('odds_ft_1', 'odds_ft_x', 'odds_ft_2', 'FootyStats API', GENERAL_BAND, (3.0, SOFT_BOOK_MAX_MARGIN)),
]


def overround(df, h, d, a):
    """Overround series for a triplet. NaN where any leg is missing/invalid."""
    cols = [h, d, a]
    if not all(c in df.columns for c in cols):
        return None
    vals = df[cols].apply(pd.to_numeric, errors='coerce')
    valid = (vals > 1).all(axis=1)
    ov = (1 / vals[h]) + (1 / vals[d]) + (1 / vals[a])
    return ov.where(valid)


def check_triplet(df, h, d, a, label, band, margin_range=None):
    """Returns a dict summarising one odds triplet's health."""
    ov = overround(df, h, d, a)
    if ov is None:
        return None
    present = ov.notna()
    n = int(present.sum())
    if n == 0:
        return {'label': label, 'cols': (h, d, a), 'n': 0, 'bad': 0,
                'pct_bad': 0.0, 'mean_clean': None, 'verdict': 'no usable rows'}

    lo, hi = band
    in_band = present & (ov >= lo) & (ov <= hi)
    bad = int(present.sum() - in_band.sum())
    pct = bad / n * 100
    mean_clean = float(ov[in_band].mean()) if in_band.any() else None

    if pct == 0:
        verdict = 'CLEAN'
    elif pct < 5:
        verdict = 'minor issues'
    elif pct < 25:
        verdict = 'DEGRADED'
    else:
        verdict = 'UNUSABLE'

    # Second check: is the margin consistent with the book this column claims
    # to be? Plausible-but-wrong-book is invisible to the band check.
    wrong_book = False
    if margin_range and mean_clean is not None:
        if n < MIN_ROWS_FOR_PROFILE:
            # Too few matches for the mean margin to mean anything. Early-season
            # files routinely hold 8-12 matches, where the estimate swings
            # wildly; flagging on that produces false alarms and trains you to
            # ignore the warning, which defeats the point.
            verdict = f'{verdict} (n<{MIN_ROWS_FOR_PROFILE}, profile skipped)'
        else:
            margin_pct = (mean_clean - 1) * 100
            lo_m, hi_m = margin_range
            if not (lo_m <= margin_pct <= hi_m):
                wrong_book = True
                verdict = 'WRONG BOOK?'

    return {'label': label, 'cols': (h, d, a), 'n': n, 'bad': bad,
            'pct_bad': pct, 'mean_clean': mean_clean, 'verdict': verdict,
            'wrong_book': wrong_book, 'expected_margin': margin_range,
            'min': float(ov[present].min()), 'max': float(ov[present].max())}


# Maps our filenames to the league code the file's `Div` column must contain.
EXPECTED_DIV = {
    'E0_raw.csv': 'E0', 'SP1_raw.csv': 'SP1', 'I1_raw.csv': 'I1',
    'D1_raw.csv': 'D1', 'F1_raw.csv': 'F1', 'N1_raw.csv': 'N1',
    'P1_raw.csv': 'P1', 'B1_raw.csv': 'B1',
}


def check_league_identity(df, filename):
    """
    Confirms the file actually contains the league its name claims.

    football-data.co.uk has been observed serving the wrong league at a given
    URL early in a season (E0.csv returning Championship data, SP1.csv
    returning Primeira Liga). Such a file is perfectly valid CSV with plausible
    odds — nothing else in this module would catch it.
    """
    base = os.path.basename(filename)
    expected = EXPECTED_DIV.get(base)
    if not expected or 'Div' not in df.columns:
        return None
    found = df['Div'].dropna().unique().tolist()
    if not found:
        return {'expected': expected, 'found': [], 'ok': False,
                'note': 'Div column present but empty'}
    ok = len(found) == 1 and str(found[0]).strip() == expected
    return {'expected': expected, 'found': found, 'ok': ok,
            'note': '' if ok else f"contains {found}, expected '{expected}'"}


def audit_frame(df, name='dataframe'):
    """Checks every known odds triplet present. Returns list of result dicts."""
    results = []
    for h, d, a, label, band, margin_range in KNOWN_TRIPLETS:
        r = check_triplet(df, h, d, a, label, band, margin_range)
        if r:
            results.append(r)
    return results


def filter_to_clean(df, h='PSCH', d='PSCD', a='PSCA', band=SHARP_BAND, verbose=True):
    """
    Drops rows whose odds are implausible. Use this before any analysis that
    settles bets at these prices.
    """
    ov = overround(df, h, d, a)
    if ov is None:
        if verbose:
            print(f"[validate] columns {h}/{d}/{a} not present — nothing filtered")
        return df
    lo, hi = band
    keep = ov.notna() & (ov >= lo) & (ov <= hi)
    dropped = len(df) - int(keep.sum())
    if verbose and dropped:
        print(f"[validate] dropped {dropped}/{len(df)} rows with implausible "
              f"{h}/{d}/{a} (kept {int(keep.sum())})")
    return df[keep].copy()


def assert_odds_sane(df, h='PSCH', d='PSCD', a='PSCA', band=SHARP_BAND,
                     max_pct_bad=5.0, name='data'):
    """
    Raises ValueError if more than `max_pct_bad` percent of rows are implausible.
    Use at the top of an analysis script so bad data fails loudly and early
    rather than quietly producing a confident wrong answer.
    """
    r = check_triplet(df, h, d, a, 'check', band)
    if r is None:
        raise ValueError(f"{name}: odds columns {h}/{d}/{a} not found")
    if r['n'] == 0:
        raise ValueError(f"{name}: no usable rows in {h}/{d}/{a}")
    if r['pct_bad'] > max_pct_bad:
        raise ValueError(
            f"{name}: {r['bad']}/{r['n']} rows ({r['pct_bad']:.1f}%) have "
            f"implausible {h}/{d}/{a} odds — overround range "
            f"{r['min']:.3f} to {r['max']:.3f}, expected {band[0]}-{band[1]}. "
            f"This is the Section 23 failure mode. Fix the data or use "
            f"filter_to_clean() before analysing."
        )
    return True


def print_report(name, results):
    print(f"\n{name}")
    print("-" * 78)
    if not results:
        print("  no recognised odds columns")
        return
    print(f"  {'market':22s} {'rows':>6s} {'bad':>6s} {'%bad':>7s} "
          f"{'clean ovr':>10s}  verdict")
    for r in results:
        mc = f"{(r['mean_clean']-1)*100:.2f}%" if r['mean_clean'] else "  --  "
        note = ''
        if r.get('wrong_book') and r.get('expected_margin'):
            lo_m, hi_m = r['expected_margin']
            note = f"  (expected {lo_m:.1f}-{hi_m:.1f}%)"
        print(f"  {r['label']:22s} {r['n']:6d} {r['bad']:6d} {r['pct_bad']:6.1f}% "
              f"{mc:>10s}  {r['verdict']}{note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', help='Check one CSV instead of the whole data folder')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--strict', action='store_true',
                    help='Exit 1 if any file is DEGRADED or UNUSABLE')
    args = ap.parse_args()

    files = [args.file] if args.file else sorted(glob.glob(os.path.join(args.data_dir, '*.csv')))
    if not files:
        print(f"No CSVs found in {args.data_dir}/")
        return 0

    print("=" * 78)
    print("DATA PLAUSIBILITY AUDIT")
    print("Checks whether bookmaker odds could be real prices (Section 23.4).")
    print("=" * 78)

    worst = 'CLEAN'
    rank = {'CLEAN': 0, 'no usable rows': 0, 'minor issues': 1,
            'DEGRADED': 2, 'WRONG BOOK?': 3, 'UNUSABLE': 3}

    for f in files:
        try:
            df = pd.read_csv(f, on_bad_lines='skip')
        except Exception as e:
            print(f"\n{f}\n  could not read: {e}")
            continue
        results = audit_frame(df, f)
        print_report(f"{os.path.basename(f)}  ({len(df)} rows)", results)

        ident = check_league_identity(df, f)
        if ident and not ident['ok']:
            print(f"  *** WRONG LEAGUE: {ident['note']}")
            print(f"      This file does not contain the league its name claims.")
            worst = 'UNUSABLE'
        elif ident:
            print(f"  league identity: OK (Div={ident['expected']})")
        for r in results:
            v = r['verdict']
            if 'profile check skipped' in v:
                v = v.split(' (')[0]      # judge on the plausibility result only
            if rank.get(v, 0) > rank.get(worst, 0):
                worst = v

    print("\n" + "=" * 78)
    print(f"WORST VERDICT ACROSS ALL FILES: {worst}")
    if worst in ('DEGRADED', 'UNUSABLE', 'WRONG BOOK?'):
        print("\nAction: do not analyse the affected columns as-is. Either")
        print("  (a) re-download the source data (the GitHub Action does this")
        print("      automatically and avoids transcription entirely), or")
        print("  (b) call filter_to_clean() to drop bad rows before analysis.")
    print("=" * 78)

    if args.strict and worst in ('DEGRADED', 'UNUSABLE', 'WRONG BOOK?'):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
