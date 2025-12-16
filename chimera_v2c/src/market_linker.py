from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from chimera_v2c.lib import espn_schedule
from chimera_v2c.lib import kalshi_utils
from chimera_v2c.lib import team_mapper


@dataclass
class MarketQuote:
    ticker: str
    yes_bid: Optional[int]
    yes_ask: Optional[int]
    home: str
    away: str
    yes_team: Optional[str] = None
    event_ticker: Optional[str] = None
    event_date: Optional[str] = None

    @property
    def mid(self) -> Optional[float]:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return ((self.yes_bid + self.yes_ask) / 2.0) / 100.0

    @property
    def spread(self) -> Optional[float]:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return (self.yes_ask - self.yes_bid) / 100.0


def _find_yes_quotes(market: Dict) -> Tuple[Optional[int], Optional[int]]:
    ob = market.get("order_book") or {}
    bids = ob.get("yes") or market.get("yes") or []
    asks = ob.get("no") or market.get("no") or []

    best_bid = bids[0][0] if bids else market.get("yes_bid")
    # For asks we invert best no bid when only yes/no bids given
    best_no_bid = asks[0][0] if asks else market.get("no_bid")
    best_ask = None
    if best_no_bid is not None:
        best_ask = 100 - best_no_bid
    if market.get("yes_ask") is not None:
        best_ask = market.get("yes_ask")
    return best_bid, best_ask


_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def _event_date_iso_from_core(core: str) -> Optional[str]:
    """
    Parse YYYY-MM-DD from an event core like '25DEC14VANNJ'.
    Returns None if the prefix is not a standard Kalshi YYMMMDD token.
    """
    if not core:
        return None
    core = str(core).strip().upper()
    if len(core) < 7:
        return None
    yy = core[0:2]
    mon = core[2:5]
    dd = core[5:7]
    if not yy.isdigit() or not dd.isdigit():
        return None
    month = _MONTHS.get(mon)
    if month is None:
        return None
    year = 2000 + int(yy)
    try:
        return datetime.date(year, month, int(dd)).isoformat()
    except ValueError:
        return None


def _market_matchup_from_event_core(core: str, league: str) -> Optional[Tuple[str, str, str]]:
    """
    Return (away, home, date_iso) from an event core like '25DEC14VANNJ'.
    Uses team_mapper.match_teams_in_string so Kalshi's 2-letter codes (e.g. NJ, NE, KC) are handled.
    """
    date_iso = _event_date_iso_from_core(core)
    if not date_iso:
        return None
    remainder = core[7:]
    match = team_mapper.match_teams_in_string(remainder, league)
    if match and len(match) == 2:
        away, home = match[0], match[1]
        if away and home and away != home:
            return away, home, date_iso
    return None


def fetch_markets(
    league: str,
    series_ticker: str,
    use_private: bool,
    *,
    status: str = "open",
    target_date: Optional[datetime.date] = None,
) -> List[MarketQuote]:
    markets_raw: List[Dict] = []
    status_norm = (status or "").strip().lower()
    if status_norm == "closed":
        # Kalshi public API uses `settled` for finalized markets; keep `closed` as a legacy alias.
        status_norm = "settled"
    if use_private and kalshi_utils.has_private_creds():
        markets_raw = kalshi_utils.get_markets(series_ticker=series_ticker, status=status_norm, get_all=True)
    else:
        demo = kalshi_utils.list_public_markets(limit=500, status=status_norm, series_ticker=series_ticker)
        markets_raw = demo.get("markets", [])

    results: List[MarketQuote] = []
    for m in markets_raw:
        ticker = m.get("ticker")
        if not ticker or series_ticker.replace("GAME", "") not in ticker:
            continue
        event_ticker = m.get("event_ticker")
        core = None
        if event_ticker:
            parts = str(event_ticker).split("-")
            if parts:
                core = parts[-1]
        if not core:
            parts = str(ticker).split("-")
            core = parts[-2] if len(parts) >= 2 else None
        if not core:
            continue

        parsed = _market_matchup_from_event_core(str(core), league)
        if not parsed:
            continue
        away, home, date_iso = parsed
        if target_date is not None and date_iso != target_date.isoformat():
            continue

        yes_bid, yes_ask = _find_yes_quotes(m)
        yes_team = ticker.split("-")[-1]
        yes_team = team_mapper.normalize_team_code(yes_team, league)
        mq = MarketQuote(
            ticker=ticker,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            home=home,
            away=away,
            yes_team=yes_team,
            event_ticker=str(event_ticker) if event_ticker else None,
            event_date=date_iso,
        )
        results.append(mq)
    return results


def fetch_matchups(league: str, date: datetime.date) -> List[Dict]:
    sb = espn_schedule.get_scoreboard(league, date)
    matchups = espn_schedule.extract_matchups(league, sb)
    out: List[Dict] = []
    for m in matchups:
        home = team_mapper.normalize_team_code(m["home_abbr"], league)
        away = team_mapper.normalize_team_code(m["away_abbr"], league)
        if not home or not away:
            continue
        out.append(
            {
                "home": home,
                "away": away,
                "home_abbr": m["home_abbr"],
                "away_abbr": m["away_abbr"],
                "event_id": m.get("event_id"),
            }
        )
    return out


def match_markets_to_games(matchups: List[Dict], markets: List[MarketQuote]) -> Dict[str, Dict[str, MarketQuote]]:
    """
    Return mapping from matchup key (away@home) to a dict of yes_team -> MarketQuote.
    This keeps both tickers (home-Yes and away-Yes) so edges can be shown for both sides.
    """
    by_matchup: Dict[str, List[MarketQuote]] = {}
    for mq in markets:
        if not mq.away or not mq.home:
            continue
        by_matchup.setdefault(f"{mq.away}@{mq.home}", []).append(mq)

    result: Dict[str, Dict[str, MarketQuote]] = {}
    for m in matchups:
        away = m["away"]
        home = m["home"]
        key = f"{away}@{home}"
        quotes: Dict[str, MarketQuote] = {}
        for mq in by_matchup.get(key, []):
            yes_team = (mq.yes_team or "").strip().upper()
            if not yes_team:
                continue
            quotes[yes_team] = mq
        if quotes:
            result[key] = quotes
    return result
