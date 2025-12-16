from __future__ import annotations

import math

import pytest

from chimera_v2c.src.offset_calibration import apply_offset, clamp_prob, compute_offset_calibration


def test_offset_calibration_clamp_prob() -> None:
    assert clamp_prob(-0.5) == 0.0
    assert clamp_prob(1.5) == 1.0
    assert clamp_prob(0.25) == 0.25


def test_offset_calibration_apply_offset_clamps() -> None:
    assert apply_offset(0.99, 0.10) == 1.0
    assert apply_offset(0.01, -0.20) == 0.0


def test_offset_calibration_compute_stats_known_example() -> None:
    # Two games: model says 0.70 both times, outcomes split 1/0.
    # Residuals: (1 - 0.70)=+0.30, (0 - 0.70)=-0.70 => mean=-0.20
    stats = compute_offset_calibration([(0.70, 1), (0.70, 0)])
    assert stats.n == 2
    assert stats.bias_mean == pytest.approx(-0.20, abs=1e-9)

    # Sample stdev of [+0.30, -0.70] around mean -0.20 is sqrt(0.5) ~= 0.7071
    assert stats.bias_stdev == pytest.approx(math.sqrt(0.5), abs=1e-6)

    # CI half-width: 1.96 * stdev / sqrt(n) = 1.96 * 0.7071 / 1.4142 = 0.98
    assert stats.bias_ci95_half_width == pytest.approx(0.98, abs=1e-2)

    # Raw Brier: ((0.7-1)^2 + (0.7-0)^2)/2 = (0.09 + 0.49)/2 = 0.29
    assert stats.brier_raw == pytest.approx(0.29, abs=1e-6)

    # Calibrated offset sets p_cal = 0.5; Brier = (0.25 + 0.25)/2 = 0.25
    assert stats.brier_calibrated == pytest.approx(0.25, abs=1e-6)


def test_offset_calibration_empty_pairs_returns_n0() -> None:
    stats = compute_offset_calibration([])
    assert stats.n == 0
    assert stats.brier_raw is None
    assert stats.brier_calibrated is None

