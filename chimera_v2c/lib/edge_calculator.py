from __future__ import annotations

from typing import Optional, Tuple


def compute_edge(p_model: float, market_mid: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if market_mid is None:
        return None, None
    market_mid = max(0.0, min(1.0, market_mid))
    p_model = max(0.0, min(1.0, p_model))
    edge = p_model - market_mid
    return market_mid, edge
