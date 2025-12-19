from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ModelPerf:
    model: str
    n: int
    net_pnl: float
    risked: float
    roi: float
    roi_lb90: float


@dataclass(frozen=True)
class ModelEligibility:
    league: str
    trained_through: str  # YYYY-MM-DD
    min_games: int
    confidence_level: float  # e.g. 0.90
    bootstrap_sims: int
    primary_models: List[str]
    secondary_models: List[str]
    perf_by_model: Dict[str, ModelPerf]

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "league": self.league,
            "trained_through": self.trained_through,
            "min_games": int(self.min_games),
            "confidence_level": float(self.confidence_level),
            "bootstrap_sims": int(self.bootstrap_sims),
            "primary_models": list(self.primary_models),
            "secondary_models": list(self.secondary_models),
            "perf_by_model": {
                k: {
                    "model": v.model,
                    "n": int(v.n),
                    "net_pnl": float(v.net_pnl),
                    "risked": float(v.risked),
                    "roi": float(v.roi),
                    "roi_lb90": float(v.roi_lb90),
                }
                for k, v in sorted(self.perf_by_model.items())
            },
        }

    @classmethod
    def from_json_dict(cls, d: Dict[str, object]) -> "ModelEligibility":
        perf_raw = dict(d.get("perf_by_model") or {})
        perf: Dict[str, ModelPerf] = {}
        for k, vv in perf_raw.items():
            if not isinstance(vv, dict):
                continue
            perf[str(k)] = ModelPerf(
                model=str(vv.get("model") or k),
                n=int(vv.get("n") or 0),
                net_pnl=float(vv.get("net_pnl") or 0.0),
                risked=float(vv.get("risked") or 0.0),
                roi=float(vv.get("roi") or 0.0),
                roi_lb90=float(vv.get("roi_lb90") or 0.0),
            )
        return cls(
            league=str(d.get("league") or ""),
            trained_through=str(d.get("trained_through") or ""),
            min_games=int(d.get("min_games") or 10),
            confidence_level=float(d.get("confidence_level") or 0.9),
            bootstrap_sims=int(d.get("bootstrap_sims") or 2000),
            primary_models=[str(x) for x in (d.get("primary_models") or [])],
            secondary_models=[str(x) for x in (d.get("secondary_models") or [])],
            perf_by_model=perf,
        )

    @classmethod
    def load_json(cls, path: Path) -> "ModelEligibility":
        d = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            raise ValueError("eligibility JSON must be an object")
        return cls.from_json_dict(d)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bootstrap_roi_lower_bound(
    *,
    net: Sequence[float],
    risk: Sequence[float],
    confidence_level: float,
    sims: int,
    seed: int,
) -> float:
    """
    Return the lower bound (alpha-quantile) of bootstrapped ROI where alpha=1-confidence_level.
    """
    if len(net) != len(risk):
        raise ValueError("net and risk must have same length")
    n = int(len(net))
    if n <= 0:
        return 0.0
    alpha = 1.0 - float(confidence_level)
    if alpha <= 0.0:
        alpha = 0.0
    if alpha >= 1.0:
        alpha = 1.0

    arr_net = np.array(net, dtype=float)
    arr_risk = np.array(risk, dtype=float)
    rng = np.random.default_rng(int(seed))
    sims_n = int(sims)
    rois = np.empty(sims_n, dtype=float)
    idx = np.arange(n)
    for i in range(sims_n):
        sample = rng.choice(idx, size=n, replace=True)
        sn = float(arr_net[sample].sum())
        sr = float(arr_risk[sample].sum())
        rois[i] = 0.0 if sr <= 0 else sn / sr
    return float(np.quantile(rois, alpha))


def learn_eligibility(
    *,
    league: str,
    trained_through: str,
    per_model_samples: Dict[str, List[Tuple[float, float]]],
    min_games: int,
    confidence_level: float,
    bootstrap_sims: int,
    seed: int,
) -> ModelEligibility:
    """
    Build per-league primary/secondary eligibility from per-model samples.

    per_model_samples maps model -> list of (net_pnl, risked) for 1-contract incremental trades.

    Primary models:
      - n >= min_games
      - ROI lower bound at confidence_level > 0

    Secondary models:
      - n >= min_games
      - not primary
    """
    perf: Dict[str, ModelPerf] = {}
    primary: List[str] = []
    secondary: List[str] = []

    def _seed_u32(*parts: object) -> int:
        s = "\x1f".join(str(p) for p in parts)
        return int(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)

    for model, samples in sorted(per_model_samples.items()):
        xs = list(samples)
        n = len(xs)
        if n <= 0:
            perf[model] = ModelPerf(model=model, n=0, net_pnl=0.0, risked=0.0, roi=0.0, roi_lb90=0.0)
            continue
        net = [float(a) for a, _ in xs]
        risk = [float(b) for _, b in xs]
        net_sum = float(sum(net))
        risk_sum = float(sum(risk))
        roi = 0.0 if risk_sum <= 0 else net_sum / risk_sum
        lb = bootstrap_roi_lower_bound(
            net=net,
            risk=risk,
            confidence_level=float(confidence_level),
            sims=int(bootstrap_sims),
            seed=int(_seed_u32(seed, league, model, n)),
        )
        perf[model] = ModelPerf(model=model, n=n, net_pnl=net_sum, risked=risk_sum, roi=float(roi), roi_lb90=float(lb))

    for model, mp in perf.items():
        if int(mp.n) < int(min_games):
            continue
        if float(mp.roi_lb90) > 0.0:
            primary.append(model)
        else:
            secondary.append(model)

    return ModelEligibility(
        league=str(league),
        trained_through=str(trained_through),
        min_games=int(min_games),
        confidence_level=float(confidence_level),
        bootstrap_sims=int(bootstrap_sims),
        primary_models=sorted(primary),
        secondary_models=sorted(secondary),
        perf_by_model=perf,
    )
