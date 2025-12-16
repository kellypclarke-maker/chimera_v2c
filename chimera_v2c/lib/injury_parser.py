from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class InjuryAdjustmentConfig:
    """
    Simple, deterministic injury adjustment configuration.

    Allows manual Elo-like penalties per team/date without depending on
    an online LLM call. Can later be fed by an LLM-based ETL.

    File format (JSON), e.g.:

    {
      "NBA": {
        "2025-12-03": {
          "NYK": -3.0,
          "BOS": -1.5
        }
      }
    }
    """

    base_path: Path = Path("data") / "injury_adjustments.json"

    def load_all(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        if not self.base_path.exists():
            return {}
        try:
            with self.base_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data  # type: ignore[return-value]
        except Exception:
            return {}
        return {}

    def get_delta(self, league: str, team: str, date: str) -> float:
        data = self.load_all()
        league_key = (league or "").upper()
        team_key = (team or "").upper()
        by_league = data.get(league_key) or {}
        by_date = by_league.get(date) or {}
        try:
            val = float(by_date.get(team_key, 0.0))
        except Exception:
            return 0.0
        return val


_CFG = InjuryAdjustmentConfig()


def apply_injury_adjustments(home_team: str, away_team: str, date: str, league: str = "NBA") -> Tuple[float, float]:
    """
    Return rating deltas (home_delta, away_delta) based on injury config.

    Positive values indicate an adjustment in favor of the team, negative
    values indicate a penalty (e.g., key players missing).
    """
    lg = (league or "NBA").upper()
    dh = _CFG.get_delta(lg, home_team, date)
    da = _CFG.get_delta(lg, away_team, date)
    return dh, da
