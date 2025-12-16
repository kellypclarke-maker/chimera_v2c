from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from chimera_v2c.src.calibration import PlattScaler
from chimera_v2c.src.offset_calibration import clamp_prob


@dataclass(frozen=True)
class GrokMidHybridParams:
    """
    Hybrid probability:
      p_grok_cal = Platt(grok_raw)
      p_hybrid = clamp(p_mid + alpha * (p_grok_cal - p_mid))
    """

    a: float
    b: float
    alpha: float
    n: int = 0
    league: str = "all"

    def grok_calibrator(self) -> PlattScaler:
        return PlattScaler(a=float(self.a), b=float(self.b))

    def predict(self, *, p_grok_raw: float, p_mid: float) -> float:
        p_mid_f = clamp_prob(float(p_mid))
        p_grok_cal = self.grok_calibrator().predict(clamp_prob(float(p_grok_raw)))
        alpha_f = float(max(0.0, min(1.0, float(self.alpha))))
        return clamp_prob(p_mid_f + alpha_f * (float(p_grok_cal) - p_mid_f))


def load_grok_mid_hybrid_params(path: Path) -> Optional[GrokMidHybridParams]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        return GrokMidHybridParams(
            a=float(payload.get("a", 1.0)),
            b=float(payload.get("b", 0.0)),
            alpha=float(payload.get("alpha", 1.0)),
            n=int(payload.get("n", 0) or 0),
            league=str(payload.get("league", "all") or "all"),
        )
    except Exception:
        return None

