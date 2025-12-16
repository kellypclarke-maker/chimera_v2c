from __future__ import annotations

"""
ProbabilityEngine for v2c.

Intentionally JSON-driven (ratings/factors/injuries) for simplicity and testability,
rather than pulling external ratings. If you swap to a shared engine later, keep this
note to preserve the intent.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from chimera_v2c.lib.injury_parser import InjuryAdjustmentConfig
from chimera_v2c.lib import team_mapper


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@dataclass
class ProbabilityEngine:
    league: str
    weights: Dict[str, float]
    elo_cfg: Dict[str, float]
    ff_cfg: Dict[str, float]
    ratings_path: Path
    factors_path: Path
    injury_path: Path
    goalie_cfg: Dict[str, float] = None
    goalie_path: Optional[Path] = None
    ff_model_path: Path = Path("chimera_v2c/data/ff_model.json")

    def __post_init__(self) -> None:
        self.goalie_cfg = self.goalie_cfg or {}
        self.ratings = _load_json(self.ratings_path)
        if not self.ratings:
            print(f"[warn] team_ratings.json missing or empty at {self.ratings_path}; "
                  "Elo component will fall back to base_rating for all teams.")

        self.factors = _load_json(self.factors_path)
        if not self.factors:
            print(f"[warn] team_four_factors.json missing or empty at {self.factors_path}; "
                  "Four Factors component will be unavailable.")

        self.ff_model = _load_json(self.ff_model_path) if self.ff_model_path else {}
        if not self.ff_model:
            print(f"[warn] ff_model.json missing at {self.ff_model_path}; using heuristic Four Factors weights.")

        self.inj_cfg = InjuryAdjustmentConfig(base_path=self.injury_path)
        # Warn if injury file is missing or empty; this is informational and non-fatal.
        try:
            data = self.inj_cfg.load_all()  # type: ignore[attr-defined]
            if not data:
                print("[warn] injury_adjustments.json is missing or empty; injuries not applied.")
        except Exception:
            print("[warn] injury_adjustments.json could not be read; injuries not applied.")

        self.goalie_ratings = {}
        if self.goalie_path:
            try:
                self.goalie_ratings = _load_json(self.goalie_path)
                if not self.goalie_ratings:
                    print(f"[warn] goalie ratings missing or empty at {self.goalie_path}; goalie component disabled.")
            except Exception:
                print(f"[warn] failed to load goalie ratings at {self.goalie_path}; goalie component disabled.")

    def ensemble_prob(
        self,
        home: str,
        away: str,
        date: str,
        market_mid: Optional[float],
        sharp_prior: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Return (p_final, components) using config-aligned weights."""
        comps: Dict[str, float] = {}

        p_elo = self._p_elo(home, away, date)
        p_ff = self._p_ff(home, away)
        p_goalie = self._p_goalie(home, away)
        market_signal = sharp_prior if sharp_prior is not None else market_mid

        if p_elo is not None:
            comps["elo"] = p_elo
        if p_ff is not None:
            comps["four_factors"] = p_ff
        if p_goalie is not None:
            comps["goalie"] = p_goalie
        if market_signal is not None:
            comps["market"] = market_signal
        if market_mid is not None:
            comps["kalshi_mid"] = market_mid
        if sharp_prior is not None:
            comps["sharp_prior"] = sharp_prior

        num = 0.0
        denom = 0.0
        for key, prob in comps.items():
            w = 0.0 if key in {"kalshi_mid", "sharp_prior"} else self.weights.get(key, 0.0)
            num += w * prob
            denom += w

        if denom <= 0.0:
            p_final = sum(comps.values()) / len(comps) if comps else 0.5
        else:
            p_final = num / denom
        p_final = max(0.01, min(0.99, p_final))
        comps["final"] = p_final
        return p_final, comps

    # ----------------
    # Components
    # ----------------
    def _inj_delta(self, team: str, date: str) -> float:
        try:
            return float(self.inj_cfg.get_delta(self.league, team, date))
        except Exception:
            return 0.0

    def _rating(self, team: str) -> float:
        key = team.upper()
        if key not in self.ratings:
            print(f"[warn] rating missing for {key}; using base {self.elo_cfg.get('base_rating', 1500.0)}")
        return float(self.ratings.get(key, self.elo_cfg.get("base_rating", 1500.0)))

    def _p_elo(self, home: str, away: str, date: str) -> Optional[float]:
        h = self._rating(home) + self.elo_cfg.get("home_bonus", 3.0) + self._inj_delta(home, date)
        a = self._rating(away) + self._inj_delta(away, date)
        diff = h - a
        k = self.elo_cfg.get("k_scale", 0.18)
        x = -k * diff
        p = 1.0 / (1.0 + math.exp(x))
        return max(0.01, min(0.99, p))

    def _p_ff(self, home: str, away: str) -> Optional[float]:
        # Support league-namespaced factors JSON ({"NHL": {...}, "NBA": {...}})
        factors_map = self.factors
        league_key = (self.league or "").upper()
        if isinstance(factors_map.get(league_key), dict):
            factors_map = factors_map.get(league_key, {})

        h = factors_map.get(home.upper())
        a = factors_map.get(away.upper())
        if not isinstance(h, dict) or not isinstance(a, dict):
            return None

        def _val(d: Dict[str, float], key: str) -> float:
            try:
                return float(d.get(key, 0.0))
            except Exception:
                return 0.0

        # NHL factors: process + special teams + goalie rolled into one component
        if self.league.lower() == "nhl":
            cfg = self.ff_cfg
            xgf_diff = _val(h, "xgf_pct") - _val(a, "xgf_pct")
            hdcf_diff = _val(h, "hdcf_pct") - _val(a, "hdcf_pct")
            pp_diff = _val(h, "pp_index") - _val(a, "pp_index")
            pk_diff = _val(h, "pk_index") - _val(a, "pk_index")
            goalie_diff = _val(h, "goalie_rating") - _val(a, "goalie_rating")
            oi_sh_diff = _val(h, "oi_sh_pct") - _val(a, "oi_sh_pct")

            score = (
                cfg.get("xgf_weight", 0.35) * xgf_diff
                + cfg.get("hdcf_weight", 0.20) * hdcf_diff
                + cfg.get("pp_weight", 0.20) * pp_diff
                + cfg.get("pk_weight", 0.15) * pk_diff
                + cfg.get("goalie_weight", 0.05) * goalie_diff
                + cfg.get("oi_sh_weight", 0.05) * oi_sh_diff
                + cfg.get("intercept", 0.0)
            )
            steep = cfg.get("steepness", 1.2)
            p = 1.0 / (1.0 + math.exp(-steep * score))
            return max(0.01, min(0.99, p))

        if self.ff_model:
            coef = self.ff_model.get("coef") or {}
            intercept = float(self.ff_model.get("intercept", 0.0))
            efg_diff = _val(h, "efg_pct") - _val(a, "efg_pct")
            tov_diff = _val(h, "tov_pct") - _val(a, "tov_pct")
            orb_diff = _val(h, "orb_pct") - _val(a, "orb_pct")
            ftr_diff = _val(h, "ft_rate") - _val(a, "ft_rate")
            # fallback to legacy keys if pct keys are missing
            if efg_diff == 0.0 and "efg" in h and "efg" in a:
                efg_diff = _val(h, "efg") - _val(a, "efg")
            if tov_diff == 0.0 and "tov" in h and "tov" in a:
                tov_diff = _val(h, "tov") - _val(a, "tov")
            if orb_diff == 0.0 and "orb" in h and "orb" in a:
                orb_diff = _val(h, "orb") - _val(a, "orb")
            if ftr_diff == 0.0 and "ftr" in h and "ftr" in a:
                ftr_diff = _val(h, "ftr") - _val(a, "ftr")
            score = (
                intercept
                + float(coef.get("efg_diff", 0.0)) * efg_diff
                + float(coef.get("tov_diff", 0.0)) * tov_diff
                + float(coef.get("orb_diff", 0.0)) * orb_diff
                + float(coef.get("ftr_diff", 0.0)) * ftr_diff
            )
            p = 1.0 / (1.0 + math.exp(-score))
            return max(0.01, min(0.99, p))
        else:
            efg_diff = _val(h, "efg") - _val(a, "efg")
            tov_diff = _val(h, "tov") - _val(a, "tov")
            orb_diff = _val(h, "orb") - _val(a, "orb")
            ftr_diff = _val(h, "ftr") - _val(a, "ftr")

            cfg = self.ff_cfg
            score = (
                cfg.get("efg_weight", 0.4) * efg_diff
                + cfg.get("tov_weight", -0.25) * tov_diff
                + cfg.get("orb_weight", 0.2) * orb_diff
                + cfg.get("ftr_weight", 0.15) * ftr_diff
                + cfg.get("intercept", 0.0)
            )
            steep = cfg.get("steepness", 1.75)
            p = 1.0 / (1.0 + math.exp(-steep * score))
            return max(0.01, min(0.99, p))

    def _goalie_rating(self, team: str) -> Optional[float]:
        if not self.goalie_ratings:
            return None
        try:
            return float(self.goalie_ratings.get(team.upper()))
        except Exception:
            return None

    def _p_goalie(self, home: str, away: str) -> Optional[float]:
        """Simple logistic on goalie rating diff (home - away)."""
        if self.league.lower() == "nhl" and self.ff_cfg.get("goalie_weight", 0.0) > 0:
            # Goalie handled inside NHL factors component
            return None
        gh = self._goalie_rating(home)
        ga = self._goalie_rating(away)
        if gh is None or ga is None:
            return None
        diff = gh - ga + self.goalie_cfg.get("home_bonus", 0.0)
        k = self.goalie_cfg.get("k_scale", 0.12)
        p = 1.0 / (1.0 + math.exp(-k * diff))
        return max(0.01, min(0.99, p))


def normalize_team(code: str, league: str) -> Optional[str]:
    return team_mapper.normalize_team_code(code, league)
