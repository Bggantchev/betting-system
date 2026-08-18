"""
api_data.py — feed the app from the FootyStats API instead of CSVs
(Blueprint Section 30)

WHY
---
The CSV pipeline has two problems the API does not:

  1. **Coverage.** Eight leagues from football-data.co.uk versus ~50 subscribed
     FootyStats leagues, with xG included.
  2. **Reliability.** Section 27 found football-data.co.uk mismapping 2026-27
     URLs — serving Championship data at the Premier League path and Primeira
     Liga at the La Liga path — plus HTML error pages returned with HTTP 200.

APPROACH
--------
This is an adapter, not a rewrite. It returns a DataFrame using the **same
column names the existing scoring logic already expects** (`HomeTeam`,
`AwayTeam`, `FTHG`, `FTAG`, `FTR`, `Date`), so `score_fixture()` and everything
built on it works unchanged. Changing the data source and the model at the same
time would make any behaviour difference impossible to attribute.

API BUDGET
----------
Streamlit re-runs its script on every interaction, so uncached calls would
exhaust the 1800/hour limit quickly. Everything here is cached:

  - league list: 24 hours (subscriptions rarely change)
  - season matches: 1 hour (results only change after a matchday)

A full 380-match season fits in one request at `max_per_page=1000`, so viewing
a league costs one call per hour at most.

SEASON SELECTION
----------------
The most recent season ID is not always the right one. In mid-August 2026 the
2026-27 season existed but held ~9 matches — too few for the composite, which
needs five per team. This picks the most recent season with enough completed
matches and reports which it chose, rather than silently using an empty one.
"""

import pandas as pd

from footystats_client import FootyStatsClient, FootyStatsError

MIN_COMPLETED_FOR_USE = 30      # below this a season can't support the composite


def get_client(api_key=None, streamlit_secrets=None):
    """
    Builds a client, taking the key from Streamlit secrets when running in the
    deployed app (where there is no local key file).
    """
    if api_key is None and streamlit_secrets is not None:
        try:
            api_key = streamlit_secrets['FOOTYSTATS_KEY']
        except Exception:
            # StreamlitSecretNotFoundError is not a KeyError — catch broadly.
            # Section 26.2 documents what happens when this is too narrow.
            api_key = None
    return FootyStatsClient(api_key=api_key, verbose=False)


def list_leagues(fs):
    """
    Returns [{'name', 'country', 'label', 'seasons': [ids newest-last]}, ...]
    sorted by label, for a dropdown.
    """
    raw = fs.league_list(chosen_only=True)
    out = []
    for lg in raw:
        name = lg.get('name') or ''
        country = lg.get('country') or ''
        seasons = lg.get('season') or lg.get('seasons') or []
        ids = [s.get('id') for s in seasons
               if isinstance(s, dict) and s.get('id')]
        if not ids:
            continue
        label = f"{country} — {name}" if country and country not in name else (name or country)
        out.append({'name': name, 'country': country, 'label': label, 'seasons': ids})
    out.sort(key=lambda x: x['label'])
    return out


def matches_to_frame(matches):
    """
    Converts API match dicts into the CSV-shaped DataFrame the scoring logic
    expects. Only completed matches are included — an unplayed fixture has no
    result to learn from.
    """
    rows = []
    for m in matches:
        if m.get('status') != 'complete':
            continue
        hg = m.get('homeGoalCount', m.get('home_team_goal_count'))
        ag = m.get('awayGoalCount', m.get('away_team_goal_count'))
        if hg is None or ag is None:
            continue
        try:
            hg, ag = int(hg), int(ag)
        except (TypeError, ValueError):
            continue

        ts = m.get('date_unix')
        try:
            date = pd.to_datetime(int(ts), unit='s', utc=True).tz_localize(None)
        except (TypeError, ValueError):
            continue

        rows.append({
            'Date': date,
            'HomeTeam': m.get('home_name'),
            'AwayTeam': m.get('away_name'),
            'FTHG': hg,
            'FTAG': ag,
            'FTR': 'H' if hg > ag else ('A' if ag > hg else 'D'),
            # Extra context the CSVs never had. Deliberately NOT fed into the
            # composite: Sections 11l and 11o tested xG signals and neither
            # survived validation. Displayed as description only.
            'home_xg': m.get('team_a_xg'),
            'away_xg': m.get('team_b_xg'),
            'home_shots_on_target': m.get('team_a_shotsOnTarget'),
            'away_shots_on_target': m.get('team_b_shotsOnTarget'),
            'match_id': m.get('id'),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=['HomeTeam', 'AwayTeam']).sort_values('Date').reset_index(drop=True)
    return df


def pick_season(fs, season_ids, min_completed=MIN_COMPLETED_FOR_USE):
    """
    Walks season IDs newest-first and returns the first with enough completed
    matches, as (season_id, frame, note). Raises FootyStatsError if none work.

    Costs one call per season tried. In practice the newest usually works; early
    in a season it falls back one, exactly as the data-refresh workflow does.
    """
    tried = []
    for sid in reversed(season_ids[-4:]):     # cap at four to bound the cost
        try:
            matches = fs.league_matches(sid)
        except FootyStatsError as e:
            tried.append(f"{sid}: {e}")
            continue
        frame = matches_to_frame(matches)
        if len(frame) >= min_completed:
            note = f"season {sid}, {len(frame)} completed matches"
            if tried:
                note += f" (fell back past {len(tried)} newer season(s) with too little data)"
            return sid, frame, note
        tried.append(f"{sid}: only {len(frame)} completed matches")

    raise FootyStatsError(
        "No season had enough completed matches to score with. Tried: "
        + "; ".join(tried)
    )


def team_list(frame):
    if frame.empty:
        return []
    return sorted(set(frame.HomeTeam) | set(frame.AwayTeam))


def xg_summary(frame, team):
    """
    Rolling xG for/against per game for one team — descriptive context for the
    UI. Returns None when xG is unavailable, which is the case for many leagues.
    """
    if frame.empty or 'home_xg' not in frame.columns:
        return None
    home = frame[frame.HomeTeam == team]
    away = frame[frame.AwayTeam == team]
    xg_for, xg_against, n = 0.0, 0.0, 0
    for _, m in home.iterrows():
        if pd.notna(m.get('home_xg')) and pd.notna(m.get('away_xg')):
            xg_for += float(m['home_xg']); xg_against += float(m['away_xg']); n += 1
    for _, m in away.iterrows():
        if pd.notna(m.get('away_xg')) and pd.notna(m.get('home_xg')):
            xg_for += float(m['away_xg']); xg_against += float(m['home_xg']); n += 1
    if n == 0:
        return None
    return {'matches': n, 'xg_for': xg_for / n, 'xg_against': xg_against / n,
            'xg_diff': (xg_for - xg_against) / n}
