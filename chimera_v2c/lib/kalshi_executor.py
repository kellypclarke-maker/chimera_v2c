from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PlannedOrder:
    league: str
    date: str
    game_id: str
    market_ticker: str
    side: str
    target_price: float
    stake_fraction: float


def plan_orders_for_game(
    league: str,
    date: str,
    game_id: str,
    market_ticker: Optional[str],
    p_model: float,
    market_mid: Optional[float],
    stake_fraction: Optional[float],
    target_price_override: Optional[float] = None,
    side: str = "yes",
) -> List[PlannedOrder]:
    if not market_ticker or market_mid is None or stake_fraction is None:
        return []
    p_model = max(0.0, min(1.0, p_model))

    if target_price_override is not None:
        target_price = target_price_override
    else:
        raw = min(p_model - 0.02, market_mid)
        target_price = max(0.01, min(0.99, raw))

    return [
        PlannedOrder(
            league=league,
            date=date,
            game_id=game_id,
            market_ticker=market_ticker,
            side=side,
            target_price=target_price,
            stake_fraction=stake_fraction,
        )
    ]
