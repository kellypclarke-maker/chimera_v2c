from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from io import StringIO
from typing import Any, Dict, List, Optional

import requests

from chimera_v2c.lib.team_mapper import normalize_team_code


MONEYPUCK_BASE = "https://moneypuck.com"


@dataclass(frozen=True)
class MoneyPuckGame:
    game_id: int
    away: str
    home: str
    est: str

    @property
    def matchup(self) -> str:
        return f"{self.away}@{self.home}"


@dataclass(frozen=True)
class MoneyPuckPregame:
    game_id: int
    away: str
    home: str
    moneypuck_home_win: Optional[float]
    betting_odds_home_win: Optional[float]
    starting_goalie: Optional[int]


def season_string_for_date(d: date) -> str:
    """
    MoneyPuck season schedule JSON is named like SeasonSchedule-20252026.json.
    NHL seasons span two years; for Jan-Jun, use previous year as season start.
    """
    start_year = d.year if d.month >= 7 else (d.year - 1)
    return f"{start_year}{start_year + 1}"


def schedule_url(season_string: str) -> str:
    return f"{MONEYPUCK_BASE}/moneypuck/OldSeasonScheduleJson/SeasonSchedule-{season_string}.json"


def predictions_url(game_id: int) -> str:
    return f"{MONEYPUCK_BASE}/moneypuck/predictions/{int(game_id)}.csv"


def _safe_get(url: str, *, timeout: int = 30) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Chimera-v2c MoneyPuck fetch)"}
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    return resp.text


def fetch_schedule(season_string: str) -> List[Dict[str, Any]]:
    return json.loads(_safe_get(schedule_url(season_string)))


def games_for_date(schedule: List[Dict[str, Any]], date_iso: str) -> List[MoneyPuckGame]:
    ymd = date_iso.replace("-", "")
    out: List[MoneyPuckGame] = []
    for row in schedule:
        try:
            est = str(row.get("est") or "")
            if not est.startswith(ymd):
                continue
            gid = int(row.get("id"))
            away_raw = str(row.get("a") or "").strip().upper()
            home_raw = str(row.get("h") or "").strip().upper()
        except Exception:
            continue
        away = normalize_team_code(away_raw, "nhl") or away_raw
        home = normalize_team_code(home_raw, "nhl") or home_raw
        out.append(MoneyPuckGame(game_id=gid, away=away, home=home, est=est))
    return out


def parse_pregame_csv(text: str) -> MoneyPuckPregame:
    """
    Parse MoneyPuck per-game pregame predictions CSV.
    Schema observed (2025-12):
      gameID,homeTeamCode,roadTeamCode,
      preGameHomeTeamWinInRegScore,preGameHomeTeamWinInOTScore,preGameHomeTeamWinOverallScore,
      preGameAwayTeamWinInRegScore,preGameAwayTeamWinInOTScore,preGameAwayTeamWinOverallScore,
      preGameMoneyPuckHomeWinPrediction,preGameBettingOddsHomeWinPrediction,startingGoalie
    """
    reader = csv.DictReader(StringIO(text))
    rows = list(reader)
    if not rows:
        raise ValueError("empty MoneyPuck pregame CSV")
    row = rows[0]

    def _flt(key: str) -> Optional[float]:
        val = row.get(key)
        if val is None:
            return None
        s = str(val).strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None

    def _int(key: str) -> Optional[int]:
        val = row.get(key)
        if val is None:
            return None
        s = str(val).strip()
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:
            return None

    gid = int(float(str(row.get("gameID") or "0").strip() or "0"))
    home_raw = str(row.get("homeTeamCode") or "").strip().upper()
    away_raw = str(row.get("roadTeamCode") or "").strip().upper()
    home = normalize_team_code(home_raw, "nhl") or home_raw
    away = normalize_team_code(away_raw, "nhl") or away_raw

    mp = _flt("preGameMoneyPuckHomeWinPrediction")
    odds = _flt("preGameBettingOddsHomeWinPrediction")
    goalie = _int("startingGoalie")

    return MoneyPuckPregame(
        game_id=gid,
        away=away,
        home=home,
        moneypuck_home_win=mp,
        betting_odds_home_win=odds,
        starting_goalie=goalie,
    )


def fetch_pregame(game_id: int) -> MoneyPuckPregame:
    return parse_pregame_csv(_safe_get(predictions_url(game_id)))

