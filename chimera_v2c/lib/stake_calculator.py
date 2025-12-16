from __future__ import annotations

from typing import Optional


def compute_stake_fraction(p_model: float, price: float, max_fraction: float = 0.01) -> Optional[float]:
    price = max(0.01, min(0.99, price))
    p_model = max(0.0, min(1.0, p_model))
    if p_model <= price:
        return None
    denom = 1.0 - price
    if denom <= 0.0:
        return None
    f_full = (p_model - price) / denom
    if f_full <= 0.0:
        return None
    f_safe = 0.25 * f_full
    return min(f_safe, max_fraction)
