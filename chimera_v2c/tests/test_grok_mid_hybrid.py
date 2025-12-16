from __future__ import annotations

import pytest

from chimera_v2c.src.grok_mid_hybrid import GrokMidHybridParams


def test_grok_mid_hybrid_alpha_endpoints() -> None:
    params_market = GrokMidHybridParams(a=1.0, b=0.0, alpha=0.0)
    assert params_market.predict(p_grok_raw=0.80, p_mid=0.42) == pytest.approx(0.42)

    params_grok = GrokMidHybridParams(a=1.0, b=0.0, alpha=1.0)
    assert params_grok.predict(p_grok_raw=0.80, p_mid=0.42) == pytest.approx(0.80)


def test_grok_mid_hybrid_midpoint_alpha() -> None:
    params = GrokMidHybridParams(a=1.0, b=0.0, alpha=0.5)
    assert params.predict(p_grok_raw=0.80, p_mid=0.40) == pytest.approx(0.60)


def test_grok_mid_hybrid_clamps() -> None:
    params = GrokMidHybridParams(a=1.0, b=0.0, alpha=1.0)
    assert params.predict(p_grok_raw=2.0, p_mid=0.40) == pytest.approx(1.0, abs=2e-6)
    assert params.predict(p_grok_raw=-1.0, p_mid=0.40) == pytest.approx(0.0, abs=2e-6)
