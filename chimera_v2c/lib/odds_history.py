from __future__ import annotations

import os
import re
from typing import Dict, List, Mapping, Optional, TypedDict

import numpy as np
import requests

from chimera_v2c.lib.team_mapper import normalize_team_code


SPORT_MAP = {"nba": "basketball_nba", "nhl": "icehockey_nhl", "nfl": "americanfootball_nfl"}


class BooksSnapshot(TypedDict, total=False):
    market_proxy: float
    books_home_ml: int
    books_away_ml: int


def normalize_team_name(name: str, league: str) -> Optional[str]:
    """
    Odds API history returns full team names (e.g. "Washington Wizards").
    Normalize to canonical codes by first trying the full string, then tokens.
    """
    if not name:
        return None
    code = normalize_team_code(str(name), league)
    if code:
        return code
    for tok in re.split(r"[^A-Za-z0-9]+", str(name)):
        if not tok:
            continue
        code = normalize_team_code(tok, league)
        if code:
            return code
    return None


def american_to_prob(price: float) -> Optional[float]:
    try:
        price = float(price)
    except Exception:
        return None
    if price == 0:
        return None
    if price > 0:
        return 100.0 / (price + 100.0)
    return -price / (-price + 100.0)


def devig(home_prob: float, away_prob: float) -> Optional[float]:
    if home_prob is None or away_prob is None:
        return None
    total = home_prob + away_prob
    if total <= 0:
        return None
    return float(home_prob / total)


def fetch_books_snapshot(
    *,
    league: str,
    snapshot_iso: str,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, BooksSnapshot]:
    """
    Fetch an Odds API "odds-history" snapshot and return mapping:
      AWAY@HOME -> {market_proxy, books_home_ml, books_away_ml}
    where:
      - market_proxy is the median devigged home probability across books
      - books_home_ml / books_away_ml are median US moneylines across books
    """
    sport = SPORT_MAP.get(league.lower())
    if not sport:
        return {}

    key = api_key or os.getenv("THE_ODDS_API_HISTORY_KEY") or os.getenv("THE_ODDS_API_HISTORY")
    if not key:
        raise RuntimeError("Missing THE_ODDS_API_HISTORY_KEY / THE_ODDS_API_HISTORY.")

    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds-history"
    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "date": snapshot_iso,
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    events = data.get("data") if isinstance(data, Mapping) else data
    if not isinstance(events, list):
        return {}

    proxy: Dict[str, List[float]] = {}
    home_mls: Dict[str, List[int]] = {}
    away_mls: Dict[str, List[int]] = {}

    for event in events:
        home = normalize_team_name(event.get("home_team") or "", league) or ""
        away = normalize_team_name(event.get("away_team") or "", league) or ""
        if not home or not away:
            continue
        key_str = f"{away}@{home}"
        bookmakers = event.get("bookmakers") or []
        for book in bookmakers:
            for market in book.get("markets", []):
                if (market.get("key") or "").lower() != "h2h":
                    continue
                outcomes = market.get("outcomes") or []
                home_prob = away_prob = None
                home_price = away_price = None
                for outcome in outcomes:
                    name = outcome.get("name") or ""
                    price = outcome.get("price")
                    prob = american_to_prob(price)
                    norm = normalize_team_name(name, league)
                    if norm == home:
                        home_prob = prob
                        try:
                            home_price = int(float(price))
                        except Exception:
                            home_price = None
                    elif norm == away:
                        away_prob = prob
                        try:
                            away_price = int(float(price))
                        except Exception:
                            away_price = None

                fair = devig(home_prob, away_prob)
                if fair is not None:
                    proxy.setdefault(key_str, []).append(fair)
                if home_price is not None:
                    home_mls.setdefault(key_str, []).append(home_price)
                if away_price is not None:
                    away_mls.setdefault(key_str, []).append(away_price)

    out: Dict[str, BooksSnapshot] = {}
    for k in set(proxy) | set(home_mls) | set(away_mls):
        snap: BooksSnapshot = {}
        if proxy.get(k):
            snap["market_proxy"] = float(np.median(proxy[k]))
        if home_mls.get(k):
            snap["books_home_ml"] = int(float(np.median(home_mls[k])))
        if away_mls.get(k):
            snap["books_away_ml"] = int(float(np.median(away_mls[k])))
        if snap:
            out[k] = snap

    return out

