"""
app.py — Web version of the match scorer (Blueprint Section 16.2, Stage 1)

Same scoring logic as match_scorer.py, wrapped as a Streamlit web app so
it's usable from a browser on any device once deployed. See DEPLOYMENT.md
in this same folder for exact steps to put this online.

Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime

import api_data
from footystats_client import FootyStatsError

st.set_page_config(page_title="Betting System — Match Scorer", page_icon="⚽", layout="centered")

LEAGUE_CONFIG = {
    'Premier League':     {'file': 'E0_raw.csv',  'total_games': 38},
    'La Liga':            {'file': 'SP1_raw.csv', 'total_games': 38},
    'Serie A':            {'file': 'I1_raw.csv',  'total_games': 38},
    'Bundesliga':         {'file': 'D1_raw.csv',  'total_games': 34},
    'Ligue 1':            {'file': 'F1_raw.csv',  'total_games': 34},
    'Eredivisie':         {'file': 'N1_raw.csv',  'total_games': 34},
    'Primeira Liga':      {'file': 'P1_raw.csv',  'total_games': 34},
    'Belgian Pro League': {'file': 'B1_raw.csv',  'total_games': 30},
}

OUTLIER_FLAGS = [
    {
        'league': 'La Liga',
        'teams': ['Real Madrid', 'Barcelona', 'Ath Madrid'],
        'condition': 'home',
        'name': 'La Liga Big-3 at home',
        'evidence': ('Section 17, Outlier #1 — 6 independent team-seasons, '
                     '+13pp to +36pp gap vs. de-vigged implied probability, '
                     'home-specific (disappears away from home).'),
        'status': ('OPEN, PROMISING, NOT VALIDATED. Small samples (n=6-11 '
                    'matches/club/season). Needs a third season before trusting '
                    'with real stakes.'),
    },
]

DATA_DIR = 'data'


# ---------------------------------------------------------------------------
# API-backed data, aggressively cached.
#
# Streamlit re-runs this whole script on every interaction, so uncached calls
# would burn the 1800/hour budget within minutes. League subscriptions almost
# never change (24h); results only change after a matchday (1h).
# ---------------------------------------------------------------------------

@st.cache_resource
def _client():
    return api_data.get_client(streamlit_secrets=st.secrets)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_leagues():
    return api_data.list_leagues(_client())


@st.cache_data(ttl=3600, show_spinner="Fetching season data...")
def cached_season(season_ids_tuple):
    sid, frame, note = api_data.pick_season(_client(), list(season_ids_tuple))
    return sid, frame, note



@st.cache_data(ttl=3600)  # re-read files at most once an hour, so a data refresh gets picked up
def load_season_data(csv_path, total_games):
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    need = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    df = df.dropna(subset=need).copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    df['total_games'] = total_games
    return df


def compute_team_stats(df):
    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    stats = {t: {'played': 0, 'pts': 0, 'gf': 0, 'ga': 0, 'last5': []} for t in teams}
    for _, m in df.iterrows():
        h, a = m.HomeTeam, m.AwayTeam
        hs, as_ = stats[h], stats[a]
        hg, ag = m.FTHG, m.FTAG
        hs['played'] += 1; as_['played'] += 1
        hs['gf'] += hg; hs['ga'] += ag
        as_['gf'] += ag; as_['ga'] += hg
        if hg > ag:
            hs['pts'] += 3; hs['last5'].append(3); as_['last5'].append(0)
        elif hg < ag:
            as_['pts'] += 3; as_['last5'].append(0); hs['last5'].append(3)
        else:
            hs['pts'] += 1; as_['pts'] += 1; hs['last5'].append(1); as_['last5'].append(1)
    return stats


def score_fixture(home_team, away_team, stats, total_games):
    hs, as_ = stats[home_team], stats[away_team]
    h_shape = (sum(hs['last5'][-5:]) / len(hs['last5'][-5:])) if hs['last5'] else 1.0
    a_shape = (sum(as_['last5'][-5:]) / len(as_['last5'][-5:])) if as_['last5'] else 1.0
    h_strength = (hs['gf'] - hs['ga']) / hs['played'] if hs['played'] > 0 else 0.0
    a_strength = (as_['gf'] - as_['ga']) / as_['played'] if as_['played'] > 0 else 0.0
    progress = min(hs['played'] / total_games, 1.0)
    h_stakes = (hs['pts'] / max(hs['played'], 1)) * progress
    a_stakes = (as_['pts'] / max(as_['played'], 1)) * progress
    composite = 0.4 * (h_shape - a_shape) + 0.4 * (h_strength - a_strength) + 0.2 * (h_stakes - a_stakes) * 3
    games_played = min(hs['played'], as_['played'])

    if games_played < 5:
        confidence = "Low — fewer than 5 games played (early season, unreliable per Section 11j)"
    elif abs(composite) >= 2.0:
        confidence = "High magnitude — but see Section 15: this tracks shorter odds, not profit"
    elif abs(composite) >= 1.0:
        confidence = "Medium magnitude"
    else:
        confidence = "Low magnitude — near-even match by this signal"

    return {
        'composite_score': composite, 'side_favored': 'Home' if composite > 0 else ('Away' if composite < 0 else 'Even'),
        'games_played_each': games_played, 'confidence_note': confidence,
        'home_form_ppg': hs['pts'] / max(hs['played'], 1), 'away_form_ppg': as_['pts'] / max(as_['played'], 1),
        'home_gd': h_strength, 'away_gd': a_strength,
    }


def check_outlier_flags(league, home_team, away_team):
    flags = []
    for entry in OUTLIER_FLAGS:
        if entry['league'] != league:
            continue
        team_to_check = home_team if entry['condition'] == 'home' else away_team
        if team_to_check in entry['teams']:
            flags.append(entry)
    return flags


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("⚽ Match Scorer")
st.caption("Stage 1 of the pre-bet decision pipeline — Blueprint Section 16.2. "
           "This is NOT a betting recommendation — see the confidence notes on every score.")

# ---------------------------------------------------------------------------
# Data source: FootyStats API when available, bundled CSVs as a fallback.
#
# The API gives ~50 leagues with xG, versus 8 CSVs. It is also more reliable —
# Section 27 found football-data.co.uk serving Championship data at the Premier
# League URL and HTML error pages with HTTP 200.
# ---------------------------------------------------------------------------

use_api = True
try:
    leagues = cached_leagues()
    if not leagues:
        use_api = False
except (FootyStatsError, Exception) as e:
    use_api = False
    api_error = str(e)

if use_api:
    labels = [lg['label'] for lg in leagues]
    default = next((i for i, l in enumerate(labels) if 'La Liga' in l), 0)
    picked = st.selectbox(f"League ({len(labels)} available via API)", labels, index=default)
    lg = next(l for l in leagues if l['label'] == picked)
    league = lg['name']

    try:
        season_id, df, season_note = cached_season(tuple(lg['seasons']))
    except FootyStatsError as e:
        st.error(f"Could not load a usable season for {picked}: {e}")
        st.info("Early in a season there may be too few completed matches to score with. "
                "Try another league, or come back once a few gameweeks have been played.")
        st.stop()

    st.caption(f"**{picked}** — {season_note}. "
               f"Latest match: {df.Date.max().strftime('%Y-%m-%d')}")
    team_list = api_data.team_list(df)

else:
    st.warning("FootyStats API unavailable — using the bundled CSV files instead. "
               "Only 8 leagues, and the data may be stale.")
    league = st.selectbox("League", list(LEAGUE_CONFIG.keys()))
    cfg = LEAGUE_CONFIG[league]
    data_path = os.path.join(DATA_DIR, cfg['file'])
    if not os.path.exists(data_path):
        st.error(f"Data file not found: {data_path}.")
        st.stop()
    df = load_season_data(data_path, cfg['total_games'])
    st.caption(f"Data as of: **{df.Date.max().strftime('%Y-%m-%d')}** ({len(df)} matches)")
    team_list = sorted(set(df.HomeTeam) | set(df.AwayTeam))
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home team", team_list)
with col2:
    away_options = [t for t in team_list if t != home_team]
    away_team = st.selectbox("Away team", away_options)

# --- Scoring, persisted in session state -----------------------------------
#
# Streamlit re-runs the whole script on every interaction, and st.button()
# returns True only on the run immediately following its click. If the results
# (and the bet-logging form) lived inside `if st.button(...)`, then submitting
# the form would trigger a re-run in which the button reads False — so the
# entire block, form handler included, would silently disappear. That is
# exactly the bug this structure avoids: the click stores its result in
# st.session_state, and rendering is driven by that stored state instead.

if st.button("Score this fixture", type="primary"):
    stats = compute_team_stats(df)
    n_teams = len(team_list)
    # Season length from squad size rather than a hardcoded table: the API
    # covers ~50 leagues of varying sizes, so (teams-1)*2 is the general form.
    total_games = max((n_teams - 1) * 2, 10) if n_teams else 38
    st.session_state['scored'] = {
        'league': league,
        'home_team': home_team,
        'away_team': away_team,
        'result': score_fixture(home_team, away_team, stats, total_games),
        'flags': check_outlier_flags(league, home_team, away_team),
    }

scored = st.session_state.get('scored')

# Discard a stale result if the user has since changed the selection, so the
# logging form can never be attached to a fixture other than the one shown.
if scored and (scored['league'] != league
               or scored['home_team'] != home_team
               or scored['away_team'] != away_team):
    scored = None
    st.session_state.pop('scored', None)

if scored:
    result = scored['result']
    flags = scored['flags']

    st.divider()
    st.subheader(f"{home_team} vs {away_team}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Composite score", f"{result['composite_score']:+.3f}")
    c2.metric("Favors", result['side_favored'])
    c3.metric("Games played (min)", result['games_played_each'])

    st.info(result['confidence_note'])

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**{home_team} (home) form:**")
        st.write(f"{result['home_form_ppg']:.2f} pts/game")
        st.write(f"Goal diff/game: {result['home_gd']:+.2f}")
    with c2:
        st.write(f"**{away_team} (away) form:**")
        st.write(f"{result['away_form_ppg']:.2f} pts/game")
        st.write(f"Goal diff/game: {result['away_gd']:+.2f}")

    # xG context, when the API provides it. Deliberately shown BELOW the score
    # and labelled as descriptive: Sections 11l and 11o tested xG-based signals
    # and neither survived out-of-sample validation, so including it in the
    # composite would be adding an unvalidated input to a validated pipeline.
    if use_api:
        xg_h = api_data.xg_summary(df, home_team)
        xg_a = api_data.xg_summary(df, away_team)
        if xg_h and xg_a:
            with st.expander("xG context (not part of the score)"):
                st.caption("Descriptive only. Blueprint Sections 11l and 11o tested "
                           "xG-based signals; neither survived validation, so xG is "
                           "deliberately excluded from the composite.")
                xc1, xc2 = st.columns(2)
                with xc1:
                    st.write(f"**{home_team}** ({xg_h['matches']} matches)")
                    st.write(f"xG for: {xg_h['xg_for']:.2f}/game")
                    st.write(f"xG against: {xg_h['xg_against']:.2f}/game")
                    st.write(f"xG diff: {xg_h['xg_diff']:+.2f}/game")
                with xc2:
                    st.write(f"**{away_team}** ({xg_a['matches']} matches)")
                    st.write(f"xG for: {xg_a['xg_for']:.2f}/game")
                    st.write(f"xG against: {xg_a['xg_against']:.2f}/game")
                    st.write(f"xG diff: {xg_a['xg_diff']:+.2f}/game")

    for flag in flags:
        st.warning(f"**OUTLIER FLAG: {flag['name']}**\n\n"
                   f"Evidence: {flag['evidence']}\n\n"
                   f"Status: {flag['status']}")

    st.caption("Next step: feed this into the pre-match research prompts (Blueprint Section 13), "
               "then the human review gate — this score alone is not a betting signal.")

    st.divider()
    st.subheader("Log a bet on this fixture")
    st.caption("Only fill this in if you've actually placed the bet — this writes a permanent row "
               "to your Google Sheet.")

    with st.form("log_bet_form"):
        side_bet = st.selectbox("Side you bet", ["Home", "Draw", "Away"])
        c1, c2 = st.columns(2)
        stake = c1.number_input("Stake", min_value=0.0, step=1.0)
        odds_at_bet = c2.number_input("Odds you got", min_value=1.0, step=0.01, format="%.2f")
        notes = st.text_input("Notes (optional)")
        submitted = st.form_submit_button("Log this bet", type="primary")

        if submitted:
            flags_text = '; '.join(f['name'] for f in flags) if flags else ''
            row = {
                'bet_id': '',  # Sheets append doesn't need this pre-filled; see note in SETUP_GOOGLE_SHEETS.md
                'logged_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'league': league, 'home_team': home_team, 'away_team': away_team,
                'side_bet': side_bet, 'composite_score': result['composite_score'],
                'side_favored_by_model': result['side_favored'], 'outlier_flags': flags_text,
                'confidence_note': result['confidence_note'], 'stake': stake,
                'odds_at_bet': odds_at_bet, 'closing_odds': '', 'clv_pct': '',
                'result': 'Pending', 'profit_loss': '', 'notes': notes,
            }
            import sheets_sync
            synced = sheets_sync.append_bet_row(row, streamlit_secrets=st.secrets)
            if synced:
                st.success(f"Logged: {home_team} vs {away_team} — {side_bet} @ {odds_at_bet:.2f}, "
                           f"stake {stake:.0f}. Synced to Google Sheets.")
            else:
                st.error("Couldn't reach Google Sheets — nothing was saved. Check Streamlit's "
                         "Secrets are set up correctly (see SETUP_GOOGLE_SHEETS.md) and try again. "
                         "(The web app has no local disk to fall back to, unlike the local script.)")
