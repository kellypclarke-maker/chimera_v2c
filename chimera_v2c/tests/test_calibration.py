from __future__ import annotations

import pytest

from chimera_v2c.src.calibration import PlattScaler, fit_platt


@pytest.mark.parametrize("p", [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
def test_platt_scaler_identity_mapping(p: float) -> None:
    scaler = PlattScaler(a=1.0, b=0.0)
    assert scaler.predict(p) == pytest.approx(p, abs=1e-6)


def test_fit_platt_empty_returns_identity() -> None:
    scaler = fit_platt([])
    assert scaler.a == pytest.approx(1.0)
    assert scaler.b == pytest.approx(0.0)

