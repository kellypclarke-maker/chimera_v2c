from __future__ import annotations

import datetime

from chimera_v2c.src.calibration import PlattScaler
from chimera_v2c.tools.walkforward_grok_mid_hybrid import (
    Sample,
    _cv_select_alpha,
    _predict_hybrid,
    walkforward_eval,
)


def test_predict_hybrid_identity_scaler() -> None:
    scaler = PlattScaler(a=1.0, b=0.0)
    p = _predict_hybrid(scaler=scaler, alpha=0.5, p_grok_raw=0.60, p_mid=0.40)
    assert abs(p - 0.50) < 1e-9


def test_cv_select_alpha_prefers_full_grok_when_helpful() -> None:
    alpha_grid = [0.0, 0.5, 1.0]
    d1 = datetime.date(2025, 12, 1)
    d2 = datetime.date(2025, 12, 2)

    # Symmetric samples around 0.5: mid=0.5, grok at 0.9 for home win and 0.1 for home loss.
    # With identity calibration (min_platt_samples high), best alpha is 1.0 on this dataset.
    samples = [
        Sample(d=d1, p_grok=0.90, p_mid=0.50, y=1),
        Sample(d=d2, p_grok=0.10, p_mid=0.50, y=0),
    ]
    alpha = _cv_select_alpha(samples, alpha_grid=alpha_grid, min_platt_samples=999)
    assert alpha == 1.0


def test_walkforward_eval_trains_only_after_two_prior_dates() -> None:
    samples = [
        Sample(d=datetime.date(2025, 12, 1), p_grok=0.90, p_mid=0.50, y=1),
        Sample(d=datetime.date(2025, 12, 2), p_grok=0.10, p_mid=0.50, y=0),
        Sample(d=datetime.date(2025, 12, 3), p_grok=0.90, p_mid=0.50, y=1),
    ]

    folds = walkforward_eval(
        samples,
        league="nba",
        alpha_step=0.5,
        min_platt_samples=2,
        train_days=0,
        skip_untrained=False,
    )
    assert len(folds) == 3
    assert folds[0].trained is False
    assert folds[1].trained is False
    assert folds[2].trained is True
    assert folds[2].alpha == 1.0

