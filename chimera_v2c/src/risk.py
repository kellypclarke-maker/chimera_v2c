from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from chimera_v2c.lib.stake_calculator import compute_stake_fraction
from chimera_v2c.src.doctrine import DoctrineConfig, doctrine_decide_trade


@dataclass
class RiskSettings:
    edge_min: float
    max_fraction: float


def size_trade(p_model: float, market_mid: float, cfg: RiskSettings) -> Optional[float]:
    if market_mid is None or p_model <= market_mid:
        return None
    edge = p_model - market_mid
    if edge < cfg.edge_min:
        return None
    return compute_stake_fraction(p_model, market_mid, max_fraction=cfg.max_fraction)


def compute_trade_decision(
    p_model: float,
    market_mid: Optional[float],
    cfg: RiskSettings,
    used_fraction: float,
    daily_cap: float,
    target_spread_bp: float,
    require_confluence: bool,
    internal_prob: Optional[float],
    market_signal: Optional[float],
    league: str,
    doctrine_cfg: Optional[dict] = None,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Delegate to doctrine_decide_trade (banded epsilon/delta_min + capped Kelly).
    Returns (stake_fraction, target_price, reason).
    """
    doc_cfg = DoctrineConfig(
        max_fraction=cfg.max_fraction,
        target_spread_bp=target_spread_bp,
        require_confluence=require_confluence,
        enable_bucket_guardrails=bool(doctrine_cfg.get("enable_bucket_guardrails", False)) if doctrine_cfg else False,
        require_positive_roi_buckets=bool(doctrine_cfg.get("require_positive_roi_buckets", False)) if doctrine_cfg else False,
        bucket_guardrails_path=(doctrine_cfg.get("bucket_guardrails_path") if doctrine_cfg else "reports/roi_by_bucket.csv"),
        league_min_samples=(doctrine_cfg.get("league_min_samples") if doctrine_cfg else None),
        paper_mode_enforce=bool(doctrine_cfg.get("paper_mode_enforce", True)) if doctrine_cfg else True,
        league=league,
        negative_roi_buckets=doctrine_cfg.get("negative_roi_buckets") if doctrine_cfg else None,
    )
    # NHL-specific confluence gap: if model vs market signal is too wide, skip (log-only reason)
    confluence_gap = None
    if doctrine_cfg:
        confluence_gap = doctrine_cfg.get("confluence_gap")
    if league.lower() == "nhl" and confluence_gap:
        signal = market_signal if market_signal is not None else market_mid
        if signal is not None and abs(p_model - signal) > confluence_gap:
            return None, None, f"confluence_gap>{confluence_gap}"

    return doctrine_decide_trade(
        p_model=p_model,
        p_market=market_mid,
        cfg=doc_cfg,
        used_fraction=used_fraction,
        daily_cap=daily_cap,
        internal_prob=internal_prob,
        market_signal=market_signal,
    )
