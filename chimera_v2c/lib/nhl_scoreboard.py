"""ESPN scoreboard fallbacks for leagues (NHL, NBA, etc.)."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import requests

LEAGUE_PATHS = {
    "nhl": "hockey/nhl",
    "nba": "basketball/nba",
    "nfl": "football/nfl",
}
SCOREBOARD_BASE = "https://site.api.espn.com/apis/site/v2/sports"


def _normalize_date(date_hint: Optional[str]) -> str:
    # Use local timezone for "today"/"tomorrow"/"yesterday" semantics so that
    # commands like "today" reflect the operator's local calendar day (PST for
    # this deployment) rather than pure UTC.
    if not date_hint:
        return datetime.now().astimezone().date().isoformat()
    clean = str(date_hint).strip().lower()
    today = datetime.now().astimezone().date()
    if clean in {"today", "tonight"}:
        return today.isoformat()
    if clean == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    if clean == "yesterday":
        return (today - timedelta(days=1)).isoformat()
    try:
        return datetime.fromisoformat(str(date_hint)).date().isoformat()
    except ValueError:
        return datetime.now(timezone.utc).date().isoformat()


def _format_competitors(data: List[Dict[str, Any]]) -> Optional[Dict[str, Dict[str, str]]]:
    home: Optional[Dict[str, str]] = None
    away: Optional[Dict[str, str]] = None
    for entry in data or []:
        team = entry.get("team") or {}
        record = {
            "alias": team.get("abbreviation")
            or team.get("shortDisplayName")
            or team.get("displayName")
            or team.get("name"),
            "name": team.get("displayName") or team.get("name") or team.get("shortDisplayName"),
        }
        if entry.get("homeAway") == "home":
            home = record
        elif entry.get("homeAway") == "away":
            away = record
    if not home or not away:
        return None
    return {"home": home, "away": away}


def _parse_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    competitions = event.get("competitions") or []
    if not competitions:
        return None
    comp = competitions[0]
    teams = _format_competitors(comp.get("competitors") or [])
    if not teams:
        return None
    venue = comp.get("venue") or {}
    address = venue.get("address") or {}
    status = comp.get("status") or {}
    status_type = status.get("type") or {}
    competitors = comp.get("competitors") or []
    home_score = None
    away_score = None
    for entry in competitors:
        score = entry.get("score")
        if entry.get("homeAway") == "home":
            home_score = score
        elif entry.get("homeAway") == "away":
            away_score = score
    return {
        "game_id": event.get("id") or comp.get("id"),
        "start_time": comp.get("date") or event.get("date"),
        "teams": teams,
        "venue": {
            "name": venue.get("fullName") or venue.get("name"),
            "city": address.get("city"),
            "state": address.get("state"),
        },
        "status": {
            "state": status_type.get("state"),
            "description": status_type.get("description"),
            "detail": status_type.get("detail"),
            "short_detail": status_type.get("shortDetail"),
            "period": status.get("period"),
            "clock": status.get("clock"),
            "display_clock": status.get("displayClock"),
        },
        "scores": {
            "home": home_score,
            "away": away_score,
        },
    }


def _fetch_scoreboard(league: str, date_hint: Optional[str]) -> Dict[str, Any]:
    iso_date = _normalize_date(date_hint)
    league_path = LEAGUE_PATHS.get(league.lower())
    if not league_path:
        return {
            "status": "error",
            "message": f"Unsupported scoreboard league '{league}'.",
            "date": iso_date,
            "source": {"type": "espn_scoreboard", "url": None, "parsed": 0},
        }

    date_token = iso_date.replace("-", "")
    url = f"{SCOREBOARD_BASE}/{league_path}/scoreboard?dates={date_token}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {
            "status": "error",
            "message": f"ESPN scoreboard error: {exc}",
            "date": iso_date,
            "source": {"type": "espn_scoreboard", "url": url, "parsed": 0},
        }

    events = payload.get("events", []) or []
    games = []
    for event in events:
        parsed = _parse_event(event)
        if parsed:
            games.append(parsed)

    status = "ok" if games else "empty"
    reason = (
        f"ESPN scoreboard parsed {len(games)} games."
        if games
        else f"No {league.upper()} games parsed from ESPN."
    )
    return {
        "status": status,
        "date": iso_date,
        "games": games,
        "reason": reason,
        "source": {"type": "espn_scoreboard", "url": url, "parsed": len(games)},
    }


def fetch_nhl_scoreboard(date_hint: Optional[str]) -> Dict[str, Any]:
    """Fetch NHL games from ESPN's public scoreboard as a fallback."""

    return _fetch_scoreboard("nhl", date_hint)


def fetch_nba_scoreboard(date_hint: Optional[str]) -> Dict[str, Any]:
    """Fetch NBA games from ESPN's public scoreboard as a fallback."""

    return _fetch_scoreboard("nba", date_hint)


def fetch_nfl_scoreboard(date_hint: Optional[str]) -> Dict[str, Any]:
    """Fetch NFL games from ESPN's public scoreboard as a fallback."""

    return _fetch_scoreboard("nfl", date_hint)
