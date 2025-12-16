from __future__ import annotations

import datetime
from typing import Any, Dict, List

import requests

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

LEAGUE_PATHS = {
    "nba": "basketball/nba",
    "nfl": "football/nfl",
    "nhl": "hockey/nhl",
    "mlb": "baseball/mlb",
    "cfb": "football/college-football",
}


def get_scoreboard(league: str, date: datetime.date) -> Dict[str, Any]:
    """
    Fetch ESPN scoreboard JSON for a given league and date.
    league: 'nba', 'nfl', 'nhl', 'mlb', 'cfb'
    """
    league_key = league.lower()
    path = LEAGUE_PATHS.get(league_key)
    if not path:
        raise ValueError(f"Unsupported league for ESPN scoreboard: {league}")

    dates_param = date.strftime("%Y%m%d")
    url = f"{ESPN_BASE}/{path}/scoreboard"
    params = {"dates": dates_param}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def extract_matchups(league: str, scoreboard: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    From a scoreboard JSON, return a list of matchups with normalized home/away info.
    Each entry: {
      'home_abbr': 'GSW',
      'away_abbr': 'DEN',
      'home_name': 'warriors',
      'away_name': 'nuggets',
      'event_id': '...'
    }
    """
    matchups: List[Dict[str, Any]] = []
    for event in scoreboard.get("events", []):
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        def _abbr(team: Dict[str, Any]) -> str:
            team_info = team.get("team") or {}
            return (team_info.get("abbreviation") or team_info.get("shortDisplayName") or "").upper()

        def _name(team: Dict[str, Any]) -> str:
            team_info = team.get("team") or {}
            return (team_info.get("shortDisplayName") or team_info.get("name") or "").lower()

        matchups.append(
            {
                "home_abbr": _abbr(home),
                "away_abbr": _abbr(away),
                "home_name": _name(home),
                "away_name": _name(away),
                "event_id": event.get("id"),
            }
        )
    return matchups
