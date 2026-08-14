"""
sheets_sync.py — shared Google Sheets connection for the bet log
(Blueprint Section 16.3/16.4)

Used by both log_bet.py (running locally) and app.py (running on
Streamlit Cloud) so there's one piece of connection logic instead of
two copies that could drift apart.

CREDENTIALS — two different sources depending on where this runs:
  - Locally: reads a JSON key file, `service_account.json`, sitting in
    the same folder as the scripts (see SETUP_GOOGLE_SHEETS.md for how
    to get this file). This file must NEVER be committed to GitHub —
    it's excluded via .gitignore.
  - On Streamlit Cloud: reads the same JSON content from Streamlit's
    own secrets manager (st.secrets), which is never stored in the
    public GitHub repo — it lives only in Streamlit Cloud's dashboard.

If neither credential source is available, every function here returns
None / False rather than raising — callers are expected to fall back
to local-only behaviour (see log_bet.py) rather than crash.
"""

import os
import json

SHEET_TAB_NAME = 'Bets'
FIELDNAMES = [
    'bet_id', 'logged_at', 'league', 'home_team', 'away_team',
    'side_bet', 'composite_score', 'side_favored_by_model',
    'outlier_flags', 'confidence_note',
    'stake', 'odds_at_bet', 'closing_odds', 'clv_pct',
    'result', 'profit_loss', 'notes',
]


def _get_credentials_dict(streamlit_secrets=None):
    """
    Returns the service-account credentials as a dict, or None if not
    configured anywhere. Checks Streamlit secrets first (if passed in
    from an app that has them), then falls back to a local JSON file.
    """
    if streamlit_secrets is not None:
        try:
            return dict(streamlit_secrets["gcp_service_account"])
        except (KeyError, TypeError):
            pass

    local_path = 'service_account.json'
    if os.path.exists(local_path):
        with open(local_path) as f:
            return json.load(f)

    return None


def _get_sheet_id(streamlit_secrets=None):
    """The target spreadsheet's ID (from its URL) — see SETUP_GOOGLE_SHEETS.md."""
    if streamlit_secrets is not None:
        try:
            return streamlit_secrets["sheet_id"]
        except (KeyError, TypeError):
            pass
    return "1d6qNWbzLulhJbeNirKnCxy_urbuq7Aaps66tkckTubo"  # or set directly below if simpler for you
    # If you'd rather not use an environment variable locally, you can also
    # just hardcode your sheet ID as a fallback here, e.g.:
    # return "1AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"


def connect(streamlit_secrets=None):
    """
    Returns an open worksheet object ready to read/write, or None if
    Google Sheets isn't configured / reachable. Never raises.
    """
    creds_dict = _get_credentials_dict(streamlit_secrets)
    sheet_id = _get_sheet_id(streamlit_secrets)
    if creds_dict is None or sheet_id is None:
        return None

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id)

        try:
            worksheet = sheet.worksheet(SHEET_TAB_NAME)
        except gspread.WorksheetNotFound:
            worksheet = sheet.add_worksheet(title=SHEET_TAB_NAME, rows=1000, cols=len(FIELDNAMES))
            worksheet.append_row(FIELDNAMES)

        return worksheet
    except Exception as e:
        print(f"[sheets_sync] Could not connect to Google Sheets: {e}")
        return None


def append_bet_row(row_dict, streamlit_secrets=None):
    """
    Appends one logged bet to the Sheet. Returns True on success,
    False if Sheets isn't reachable/configured (caller should treat
    this as "logged locally only, not synced yet" — not an error).
    """
    worksheet = connect(streamlit_secrets)
    if worksheet is None:
        return False
    try:
        row = [row_dict.get(field, '') for field in FIELDNAMES]
        worksheet.append_row(row)
        return True
    except Exception as e:
        print(f"[sheets_sync] Could not append row: {e}")
        return False


def read_all_bets(streamlit_secrets=None):
    """Returns all logged bets from the Sheet as a list of dicts, or None if unreachable."""
    worksheet = connect(streamlit_secrets)
    if worksheet is None:
        return None
    try:
        return worksheet.get_all_records()
    except Exception as e:
        print(f"[sheets_sync] Could not read rows: {e}")
        return None
