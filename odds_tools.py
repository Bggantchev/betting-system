"""
odds_tools.py — parse FootyStats `odds_comparison` (Blueprint Section 22.8)

WHY THIS MATTERS
----------------
Everything in Sections 8-21 benchmarked against **Pinnacle closing odds** —
the sharp reference the project concluded is efficiently priced (Section 15).
The FootyStats CSV exports and the flat `odds_ft_*` API fields carry a 6-7%
margin, i.e. a soft book. Section 21's whole caveat was that its findings said
something about *one soft book*, not about the market.

The `/match` endpoint's `odds_comparison` field carries per-bookmaker prices
**including Pinnacle**, confirmed by live inspection. That restores the
project's original benchmark for live data.

STRUCTURE (confirmed by inspection, 2026-08-14)
-----------------------------------------------
    odds_comparison = {
      "FT Result":  { "<selection>": { "<bookmaker>": "<decimal odds as string>" } },
      "Win To Nil": { "1": {"bet365": "3.50"}, "2": {"bet365": "7.00"} },
      ...
    }

Selection keys were only partially visible in inspection ("Draw" confirmed for
FT Result; "1"/"2" used by Win To Nil). The parser therefore accepts several
naming conventions rather than assuming one — see SELECTION_ALIASES.
"""

SHARP_BOOKS = ['Pinnacle', 'pinnacle', 'Sbo', 'SBOBET', 'sbo']

# FootyStats' selection keys vary by market; accept the plausible variants.
SELECTION_ALIASES = {
    'home': ['1', 'Home', 'home', 'HOME', 'Home Team', 'team_a'],
    'draw': ['X', 'x', 'Draw', 'draw', 'DRAW', 'Tie'],
    'away': ['2', 'Away', 'away', 'AWAY', 'Away Team', 'team_b'],
}

MARKET_ALIASES = {
    'ft_result': ['FT Result', 'Full Time Result', '1X2', 'Match Result', 'FT_Result'],
}


def _resolve(container, aliases):
    """Find the first alias present as a key in container. Returns (key, value)."""
    if not isinstance(container, dict):
        return None, None
    for a in aliases:
        if a in container:
            return a, container[a]
    return None, None


def get_market(odds_comparison, market='ft_result'):
    """Returns the selection->bookmaker->odds dict for a market, or None."""
    if not isinstance(odds_comparison, dict):
        return None
    _, market_data = _resolve(odds_comparison, MARKET_ALIASES.get(market, [market]))
    return market_data


def get_book_prices(odds_comparison, bookmaker, market='ft_result'):
    """
    All three 1X2 prices from one bookmaker.
    Returns {'home': float|None, 'draw': float|None, 'away': float|None}.
    Bookmaker name matching is case-insensitive.
    """
    market_data = get_market(odds_comparison, market)
    out = {'home': None, 'draw': None, 'away': None}
    if not market_data:
        return out

    for side, aliases in SELECTION_ALIASES.items():
        _, books = _resolve(market_data, aliases)
        if not isinstance(books, dict):
            continue
        for bname, price in books.items():
            if str(bname).lower() == str(bookmaker).lower():
                try:
                    val = float(price)
                    out[side] = val if val > 1 else None
                except (TypeError, ValueError):
                    pass
                break
    return out


def get_pinnacle(odds_comparison):
    """Convenience: Pinnacle's 1X2 prices — the project's benchmark book."""
    return get_book_prices(odds_comparison, 'Pinnacle')


def list_bookmakers(odds_comparison, market='ft_result'):
    """Every bookmaker quoted for a market, sorted."""
    market_data = get_market(odds_comparison, market)
    if not market_data:
        return []
    books = set()
    for _, entry in market_data.items():
        if isinstance(entry, dict):
            books.update(entry.keys())
    return sorted(books)


def best_available(odds_comparison, market='ft_result'):
    """
    Best price across all books for each selection.

    NOTE — Sections 7/8 documented a real trap here: taking the maximum across
    N books systematically overstates edge, because you are always selecting
    the most generous of N noisy draws. This function exists for execution
    ("where should I actually bet?"), NOT for edge estimation. Use Pinnacle for
    anything analytical.
    """
    market_data = get_market(odds_comparison, market)
    out = {'home': (None, None), 'draw': (None, None), 'away': (None, None)}
    if not market_data:
        return out
    for side, aliases in SELECTION_ALIASES.items():
        _, books = _resolve(market_data, aliases)
        if not isinstance(books, dict):
            continue
        best_book, best_price = None, 0.0
        for bname, price in books.items():
            try:
                val = float(price)
            except (TypeError, ValueError):
                continue
            if val > best_price:
                best_book, best_price = bname, val
        if best_book:
            out[side] = (best_book, best_price)
    return out


def overround(prices, warn=True):
    """
    Bookmaker margin from a {'home','draw','away'} price dict.
    Returns None if any leg is missing. 1.02 == 2% margin.

    Sanity guard: a real 1X2 book essentially never prices below 1.0 overround
    (that would be a standing arbitrage) and rarely above ~1.15. Values outside
    that band almost always mean the wrong prices were paired together — e.g. a
    selection-key mismatch pulling 'home' from one market and 'away' from
    another. Warn loudly rather than silently returning a nonsense number.
    """
    if not prices or any(prices.get(k) in (None, 0) for k in ('home', 'draw', 'away')):
        return None
    ov = sum(1.0 / prices[k] for k in ('home', 'draw', 'away'))
    if warn and not (1.0 <= ov <= 1.20):
        print(f"  [odds_tools WARNING] implausible overround {ov:.4f} from "
              f"{prices} — likely a selection-key mismatch, not a real price.")
    return ov


def devig(prices):
    """
    De-vigged (proportional) implied probabilities.

    Caveat carried forward from Section 21.7: proportional de-vigging assumes
    margin is spread evenly with probability. Books typically load margin on
    longshots, which inflates apparent favourite value. Fine for comparing
    like with like; treat single-number 'gaps' with caution.
    """
    ov = overround(prices)
    if not ov:
        return None
    return {k: (1.0 / prices[k]) / ov for k in ('home', 'draw', 'away')}


def compare_to_pinnacle(odds_comparison, side, your_odds):
    """
    Core CLV-style comparison: your price versus Pinnacle's on the same
    selection. Returns a dict, or None if Pinnacle isn't quoted.

    `side` is 'home' | 'draw' | 'away'. `your_odds` is decimal.
    """
    pin = get_pinnacle(odds_comparison)
    if not pin or pin.get(side) is None:
        return None
    pin_price = pin[side]
    fair = devig(pin)
    return {
        'pinnacle_odds': pin_price,
        'your_odds': float(your_odds),
        'edge_vs_pinnacle_pct': (float(your_odds) / pin_price - 1) * 100,
        'pinnacle_devig_prob': fair[side] if fair else None,
        'pinnacle_overround_pct': (overround(pin) - 1) * 100 if overround(pin) else None,
    }
