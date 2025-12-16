from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class V2CConfig:
    league: str
    series_ticker: str
    weights: Dict[str, float]
    require_confluence: bool
    use_sharp_prior: bool
    edge_min: float
    max_fraction: float
    daily_max_fraction: float
    target_spread_bp: float
    elo: Dict[str, Any]
    four_factors: Dict[str, Any]
    goalie: Dict[str, Any]
    paths: Dict[str, str]
    execution: Dict[str, Any]
    doctrine_cfg: Dict[str, Any]
    calibration: Dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "V2CConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return V2CConfig(
            league=raw["league"],
            series_ticker=raw["series_ticker"],
            weights=raw["weights"],
            require_confluence=bool(raw.get("require_confluence", False)),
            use_sharp_prior=bool(raw.get("use_sharp_prior", False)),
            edge_min=float(raw["edge_min"]),
            max_fraction=float(raw["max_fraction"]),
            daily_max_fraction=float(raw.get("daily_max_fraction", raw.get("max_fraction", 0.01) * 8)),
            target_spread_bp=float(raw.get("target_spread_bp", 2.0)),
            elo=raw["elo"],
            four_factors=raw["four_factors"],
            goalie=raw.get("goalie", {}),
            paths=raw["paths"],
            execution=raw["execution"],
            doctrine_cfg=raw.get("doctrine", {}),
            calibration=raw.get("calibration", {}),
        )
