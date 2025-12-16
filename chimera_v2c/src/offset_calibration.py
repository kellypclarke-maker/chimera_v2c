from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


def clamp_prob(p: float) -> float:
    return float(min(1.0, max(0.0, p)))


@dataclass(frozen=True)
class OffsetCalibrationStats:
    """
    Summary stats for an additive bias calibration:

      residual r = y - p_raw
      bias_mean b = mean(r)
      p_cal = clamp(p_raw + b)

    `bias_ci95_half_width` is a 95% CI half-width for the *mean residual* (b),
    i.e. 1.96 * stdev(r) / sqrt(n).
    """

    n: int
    bias_mean: float
    bias_stdev: float
    bias_ci95_half_width: float
    brier_raw: Optional[float]
    brier_calibrated: Optional[float]


def apply_offset(p_raw: float, bias_mean: float) -> float:
    return clamp_prob(float(p_raw) + float(bias_mean))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def compute_offset_calibration(pairs: Iterable[Tuple[float, int]]) -> OffsetCalibrationStats:
    """
    Compute additive-bias calibration stats from (p_raw, y) pairs where y ∈ {0,1}.
    """
    pairs_list = [(float(p), int(y)) for p, y in pairs]
    n = len(pairs_list)
    if n == 0:
        return OffsetCalibrationStats(
            n=0,
            bias_mean=0.0,
            bias_stdev=0.0,
            bias_ci95_half_width=0.0,
            brier_raw=None,
            brier_calibrated=None,
        )

    residuals = [float(y) - float(p) for p, y in pairs_list]
    bias_mean = _mean(residuals)
    bias_stdev = _sample_stdev(residuals)
    if n < 2 or bias_stdev == 0.0:
        half_width = 0.0
    else:
        half_width = 1.96 * (bias_stdev / math.sqrt(n))

    brier_raw = _mean([(p - float(y)) ** 2 for p, y in pairs_list])
    brier_cal = _mean([(apply_offset(p, bias_mean) - float(y)) ** 2 for p, y in pairs_list])

    return OffsetCalibrationStats(
        n=n,
        bias_mean=bias_mean,
        bias_stdev=bias_stdev,
        bias_ci95_half_width=half_width,
        brier_raw=brier_raw,
        brier_calibrated=brier_cal,
    )

