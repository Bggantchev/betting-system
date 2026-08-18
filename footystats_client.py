"""
footystats_client.py — FootyStats API client (Blueprint Section 22)

Wraps the FootyStats "Football Data API" (api.football-data-api.com) for use
by the rest of the framework.

WHAT IS VERIFIED vs WHAT IS NOT
--------------------------------
Verified against FootyStats' official documentation:
  - Base URL, and that auth is a `key=` QUERY parameter (not a header)
  - Endpoint names (league-matches, todays-matches, match, league-tables, etc.)
  - The complete field list for /league-matches (see FIELD_MAP below)
  - The `max_time` UNIX-timestamp parameter, which returns stats "as of" a
    point in time — genuinely useful for lookahead-free backtesting
  - Pagination: default 300/page, `max_per_page` up to 1000, `page=N`

NOT verified (their API is unreachable from the environment this was written
in, and their docs site blocks automated access):
  - The exact response envelope. Docs say /league-matches returns "a JSON
    Array", but many endpoints on such APIs wrap results as
    {"success": true, "data": [...]}. This client handles BOTH shapes.
  - Rate limits — undocumented in what was available. Conservative
    self-throttling is applied; tune MIN_SECONDS_BETWEEN_CALLS once known.
  - The /match endpoint's odds-comparison sub-structure.
  - Whether xG is exposed via API (the CSV exports include it; the documented
    /league-matches field list does not mention it).

Run `python3 fs_inspect.py` with your real key to check these and report back —
that script prints actual response shapes so this client can be tightened.

USAGE
-----
    from footystats_client import FootyStatsClient
    fs = FootyStatsClient()            # reads key from footystats_key.txt or env
    fs.test_connection()
    matches = fs.todays_matches()
"""

import json
import os
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone

BASE_URL = "https://api.football-data-api.com"

# CONFIRMED by live inspection (2026-08-14): the API reports
#   request_limit: 1800, refreshed every hour
# 1800/hour = one call every 2.0 seconds sustained. The earlier value of 1.0s
# would have permitted 3600/hour — double the allowance — so this is set to
# 2.0s. The client also reads `request_remaining` from each response's metadata
# and backs off automatically as the budget runs low (see _adapt_throttle).
MIN_SECONDS_BETWEEN_CALLS = 2.0
RATE_LIMIT_PER_HOUR = 1800

CACHE_DIR = ".fs_cache"

# Maps FootyStats API field names -> the internal names used elsewhere in this
# framework (which follow the CSV export convention, since all of Sections
# 8-21 were built on those). The API and the CSV export use DIFFERENT names
# for the same fields, which is an easy source of silent bugs.
FIELD_MAP = {
    'homeGoalCount':          'home_team_goal_count',
    'awayGoalCount':          'away_team_goal_count',
    'odds_ft_1':              'odds_ft_home_team_win',
    'odds_ft_x':              'odds_ft_draw',
    'odds_ft_2':              'odds_ft_away_team_win',
    'team_a_shotsOnTarget':   'home_team_shots_on_target',
    'team_b_shotsOnTarget':   'away_team_shots_on_target',
    'team_a_shots':           'home_team_shots',
    'team_b_shots':           'away_team_shots',
    'team_a_corners':         'home_team_corner_count',
    'team_b_corners':         'away_team_corner_count',
    'team_a_yellow_cards':    'home_team_yellow_cards',
    'team_b_yellow_cards':    'away_team_yellow_cards',
    'team_a_red_cards':       'home_team_red_cards',
    'team_b_red_cards':       'away_team_red_cards',
    'team_a_possession':      'home_team_possession',
    'team_b_possession':      'away_team_possession',
    'pre_match_home_ppg':     'pre_match_home_ppg',
    'pre_match_away_ppg':     'pre_match_away_ppg',
}


class FootyStatsError(Exception):
    """Raised for API errors that the caller should handle explicitly."""


class FootyStatsClient:

    def __init__(self, api_key=None, use_cache=True, verbose=True):
        self.api_key = api_key or self._load_key()
        self.use_cache = use_cache
        self.verbose = verbose
        self._last_call_at = 0.0
        self.calls_made = 0
        self.requests_remaining = None   # populated from response metadata
        self.rate_limit = None
        self._extra_delay = 0.0
        if self.use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)

    # ------------------------------------------------------------------ key

    @staticmethod
    def _load_key():
        """
        Looks for the key in, in order:
          1. FOOTYSTATS_KEY environment variable
          2. a file named footystats_key.txt in the current folder
        The file must be listed in .gitignore — it is a credential.
        """
        env = os.environ.get('FOOTYSTATS_KEY')
        if env:
            return env.strip()
        if os.path.exists('footystats_key.txt'):
            with open('footystats_key.txt') as f:
                return f.read().strip()
        raise FootyStatsError(
            "No API key found. Either set the FOOTYSTATS_KEY environment "
            "variable, or save your key in a file called 'footystats_key.txt' "
            "in this folder. (Get it from footystats.org/api/u/api-settings)"
        )

    # -------------------------------------------------------------- plumbing

    def _throttle(self):
        wait = MIN_SECONDS_BETWEEN_CALLS + self._extra_delay
        elapsed = time.time() - self._last_call_at
        if elapsed < wait:
            time.sleep(wait - elapsed)
        self._last_call_at = time.time()

    def _adapt_throttle(self, payload):
        """
        Reads rate-limit metadata the API returns on every response and slows
        down as the hourly budget depletes, so a long backfill can't silently
        burn through the allowance and start failing mid-run.
        """
        if not isinstance(payload, dict):
            return
        meta = payload.get('metadata') or {}
        try:
            self.rate_limit = int(meta.get('request_limit', self.rate_limit or 0)) or None
            remaining = meta.get('request_remaining')
            if remaining is not None:
                self.requests_remaining = int(remaining)
        except (TypeError, ValueError):
            return

        if self.requests_remaining is None or not self.rate_limit:
            return
        frac = self.requests_remaining / self.rate_limit
        if frac < 0.05:
            self._extra_delay = 8.0
        elif frac < 0.15:
            self._extra_delay = 3.0
        elif frac < 0.30:
            self._extra_delay = 1.0
        else:
            self._extra_delay = 0.0
        if frac < 0.15 and self.verbose:
            print(f"  [budget] {self.requests_remaining}/{self.rate_limit} "
                  f"requests left this hour — slowing down")

    def _cache_path(self, endpoint, params):
        """Cache filename. The API key is deliberately excluded so it never
        ends up written to disk in a filename."""
        safe = endpoint.replace('/', '_')
        keyed = {k: v for k, v in params.items() if k != 'key'}
        digest = urllib.parse.urlencode(sorted(keyed.items())).replace('&', '_').replace('=', '-')
        stem = f"{safe}__{digest}"[:180]
        return os.path.join(CACHE_DIR, stem + ".json")

    def get(self, endpoint, params=None, cache_ok=True, retries=3):
        """
        Low-level GET. Returns the parsed JSON body (whatever shape it is).
        Use the typed helpers below in preference to calling this directly.
        """
        params = dict(params or {})
        params['key'] = self.api_key

        cache_file = self._cache_path(endpoint, params) if self.use_cache else None
        if cache_ok and cache_file and os.path.exists(cache_file):
            with open(cache_file) as f:
                if self.verbose:
                    print(f"  [cache] {endpoint}")
                return json.load(f)

        url = f"{BASE_URL}/{endpoint.lstrip('/')}?{urllib.parse.urlencode(params)}"

        last_error = None
        for attempt in range(1, retries + 1):
            self._throttle()
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'betting-system/1.0'})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = resp.read().decode('utf-8')
                self.calls_made += 1
                data = json.loads(body)
                self._adapt_throttle(data)
                if cache_file:
                    with open(cache_file, 'w') as f:
                        json.dump(data, f)
                if self.verbose:
                    safe_params = {k: v for k, v in params.items() if k != 'key'}
                    print(f"  [api] {endpoint} {safe_params}")
                return data

            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 429:
                    wait = 5 * attempt
                    if self.verbose:
                        print(f"  [rate limited] waiting {wait}s (attempt {attempt}/{retries})")
                    time.sleep(wait)
                    continue
                if e.code in (401, 403):
                    raise FootyStatsError(
                        f"Authentication failed (HTTP {e.code}). Check your API key is "
                        f"correct and your subscription is active."
                    ) from e
                if 500 <= e.code < 600 and attempt < retries:
                    time.sleep(2 * attempt)
                    continue
                raise FootyStatsError(f"HTTP {e.code} calling {endpoint}: {e.reason}") from e

            except urllib.error.URLError as e:
                last_error = e
                if attempt < retries:
                    time.sleep(2 * attempt)
                    continue
                raise FootyStatsError(f"Network error calling {endpoint}: {e.reason}") from e

            except json.JSONDecodeError as e:
                raise FootyStatsError(
                    f"{endpoint} returned something that isn't JSON. First 200 chars: {body[:200]!r}"
                ) from e

        raise FootyStatsError(f"Failed calling {endpoint} after {retries} attempts: {last_error}")

    @staticmethod
    def unwrap(payload):
        """
        Normalises the response envelope. FootyStats' docs describe a bare JSON
        array for some endpoints, but wrapped {"success":..,"data":..} is common
        on this style of API and observed in the wild. Handle both rather than
        assume — an assumption here fails silently and confusingly.
        """
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if 'data' in payload:
                data = payload['data']
                return data if isinstance(data, list) else [data]
            # Some error responses use {"success": false, "message": ...}
            if payload.get('success') is False:
                raise FootyStatsError(
                    f"API reported failure: {payload.get('message') or payload}"
                )
            return [payload]
        raise FootyStatsError(f"Unexpected response type: {type(payload).__name__}")

    @staticmethod
    def normalise_match(m):
        """Renames API fields to the framework's internal names (see FIELD_MAP)."""
        out = dict(m)
        for api_name, internal in FIELD_MAP.items():
            if api_name in m:
                out[internal] = m[api_name]
        return out

    # ------------------------------------------------------------- endpoints

    def test_connection(self):
        """Calls /test-call. Returns True on success, raises FootyStatsError otherwise."""
        payload = self.get('test-call', cache_ok=False)
        return payload

    def league_list(self, chosen_only=True):
        """
        All leagues available to your subscription. Each entry carries the
        season IDs you need for other calls.
        `chosen_only=True` limits to leagues you've selected in your account.
        """
        params = {'chosen_leagues_only': 'true'} if chosen_only else {}
        return self.unwrap(self.get('league-list', params))

    def todays_matches(self, date=None):
        """
        Fixtures for a given day (defaults to today). `date` as 'YYYY-MM-DD'.
        NOTE: only returns matches from leagues selected in your account settings.
        """
        params = {'date': date} if date else {}
        raw = self.unwrap(self.get('todays-matches', params, cache_ok=False))
        return [self.normalise_match(m) for m in raw]

    def league_matches(self, season_id, max_per_page=1000, max_time=None):
        """
        Full match schedule + stats for a season, following pagination.
        `max_time` (UNIX ts) returns stats as they stood at that moment —
        the lookahead-free option for backtesting.
        """
        all_rows, page = [], 1
        while True:
            params = {'season_id': season_id, 'max_per_page': max_per_page, 'page': page}
            if max_time:
                params['max_time'] = max_time
            batch = self.unwrap(self.get('league-matches', params))
            if not batch:
                break
            all_rows.extend(batch)
            if len(batch) < max_per_page:
                break
            page += 1
            if page > 20:   # safety valve against an unexpected pagination loop
                print("  [warn] stopped paginating at page 20 — check results")
                break
        return [self.normalise_match(m) for m in all_rows]

    def match_details(self, match_id):
        """
        Single match: stats, H2H, and odds comparison. This is the endpoint that
        matters for CLV tracking (see settle_bets.py). Its exact sub-structure is
        UNVERIFIED — run fs_inspect.py to confirm before relying on it.
        """
        payload = self.get('match', {'match_id': match_id}, cache_ok=False)
        rows = self.unwrap(payload)
        return self.normalise_match(rows[0]) if rows else None

    def league_table(self, season_id, include_stats=False):
        params = {'season_id': season_id}
        if include_stats:
            params['include'] = 'stats'
        return self.unwrap(self.get('league-tables', params))

    def team_last_x(self, team_id):
        """Team's last 5 / 6 / 10 match aggregates."""
        return self.unwrap(self.get('lastx', {'team_id': team_id}))
