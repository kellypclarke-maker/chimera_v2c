from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class TradeDecision:
    bucket: str
    side: str  # "home" or "away"
    edge_threshold: float


def select_trade_decision(
    *,
    p_mid: float,
    p_model: float,
    thresholds_by_bucket: Dict[str, float],
) -> Optional[TradeDecision]:
    """
    Decide which quadrant bucket (A/B/C/D) applies for this game and model,
    using per-bucket thresholds. Returns at most one decision.

    Buckets (home-win probabilities):
      - Market home-fav (p_mid >= 0.5)
        A: p_model <= p_mid - t_A (fade home => buy away)
        B: p_model >= p_mid + t_B (follow home => buy home)
      - Market away-fav (p_mid < 0.5)
        C: p_model <= p_mid - t_C (follow away => buy away)
        D: p_model >= p_mid + t_D (fade away => buy home)
    """
    pm = float(p_mid)
    p = float(p_model)

    if pm >= 0.5:
        t_a = thresholds_by_bucket.get("A")
        if t_a is not None and p <= pm - float(t_a):
            return TradeDecision(bucket="A", side="away", edge_threshold=float(t_a))
        t_b = thresholds_by_bucket.get("B")
        if t_b is not None and p >= pm + float(t_b):
            return TradeDecision(bucket="B", side="home", edge_threshold=float(t_b))
        return None

    t_c = thresholds_by_bucket.get("C")
    if t_c is not None and p <= pm - float(t_c):
        return TradeDecision(bucket="C", side="away", edge_threshold=float(t_c))
    t_d = thresholds_by_bucket.get("D")
    if t_d is not None and p >= pm + float(t_d):
        return TradeDecision(bucket="D", side="home", edge_threshold=float(t_d))
    return None

