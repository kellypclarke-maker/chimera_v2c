from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

import requests

from chimera_v2c.lib import team_mapper


def _american_to_prob(odds: int) -> float:
    if odds == 0:
        return 0.0
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def _devig_two_way(p1: float, p2: float) -> Optional[float]:
    s = p1 + p2
    if s <= 0:
        return None
    return p1 / s


def _normalize_team_name(name: str, league: str) -> Optional[str]:
    """
    The Odds API returns full team names. Try full-string normalization first,
    then fall back to token matching against known aliases.
    """
    if not name:
        return None
    code = team_mapper.normalize_team_code(str(name), league)
    if code:
        return code
    for tok in re.split(r"[^A-Za-z0-9]+", str(name)):
        if not tok:
            continue
        code = team_mapper.normalize_team_code(tok, league)
        if code:
            return code
    return None


def fetch_sharp_home_probs(
    league: str,
    target_date: date,
    api_key: Optional[str] = None,
    require: bool = False,
) -> Dict[str, float]:
    """
    Fetch sharp home win probabilities from The Odds API (NBA/NHL supported).
    Returns mapping of matchup key `AWAY@HOME` -> fair home win prob (devigged).
    """
    league_key = league.lower()
    sport_map = {"nba": "basketball_nba", "nhl": "icehockey_nhl"}
    sport = sport_map.get(league_key)
    if not sport:
        return {}

    key = api_key or os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
    if not key:
        msg = "[error] THE_ODDS_API_KEY / ODDS_API_KEY missing; cannot fetch sharp prior from The Odds API."
        print(msg)
        if require:
            raise SystemExit(msg)
        return {}

    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        events: List[dict] = resp.json()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        body = exc.response.text if exc.response is not None else ""
        text = body.lower()
        # Quota / limit handling: fall back to Kalshi mids but log explicitly.
        if status in (402, 429) or "limit" in text or "quota" in text:
            print("[warn] The Odds API quota/limit reached; Kalshi mids used instead for market signal.")
            return {}
        msg = f"[error] failed to fetch The Odds API odds (status={status}): {exc}"
        print(msg)
        if require:
            raise SystemExit(msg)
        return {}
    except Exception as exc:
        msg = f"[error] failed to fetch The Odds API odds: {exc}"
        print(msg)
        if require:
            raise SystemExit(msg)
        return {}

    out: Dict[str, float] = {}
    for ev in events or []:
        home_team = ev.get("home_team")
        away_team = ev.get("away_team")
        if not home_team or not away_team:
            continue
        home_alias = _normalize_team_name(home_team, league_key)
        away_alias = _normalize_team_name(away_team, league_key)
        if not home_alias or not away_alias:
            continue

        # Filter to target_date if the API provides commence_time
        commence = ev.get("commence_time")
        if commence:
            try:
                ts = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                if ts.date() != target_date:
                    continue
            except Exception:
                pass

        bookmakers = ev.get("bookmakers") or []
        if not bookmakers:
            continue
        markets = bookmakers[0].get("markets") or []
        if not markets:
            continue
        outcomes = markets[0].get("outcomes") or []
        if len(outcomes) < 2:
            continue

        home_price = None
        away_price = None
        for o in outcomes:
            name = (o.get("name") or "").strip()
            odds = o.get("price")
            if odds is None:
                continue
            try:
                odds_int = int(odds)
            except Exception:
                continue
            p = _american_to_prob(odds_int)
            norm_name = _normalize_team_name(name, league_key)
            if norm_name and norm_name.lower() == home_alias.lower():
                home_price = p
            elif norm_name and norm_name.lower() == away_alias.lower():
                away_price = p

        if home_price is None or away_price is None:
            continue

        fair_home = _devig_two_way(home_price, away_price)
        if fair_home is None:
            continue

        # planner key is away@home
        key_str = f"{away_alias}@{home_alias}"
        out[key_str] = fair_home

    return out
