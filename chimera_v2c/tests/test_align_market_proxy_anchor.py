from __future__ import annotations

from chimera_v2c.tools.align_market_proxy_to_game_start import _anchor_iso


def test_anchor_iso_applies_minutes_before_start() -> None:
    assert _anchor_iso("2025-11-20T00:30:00Z", minutes_before_start=0) == "2025-11-20T00:30:00Z"
    assert _anchor_iso("2025-11-20T00:30:00Z", minutes_before_start=30) == "2025-11-20T00:00:00Z"

