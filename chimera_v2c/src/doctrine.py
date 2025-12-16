from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from chimera_v2c.lib.stake_calculator import compute_stake_fraction


# Simple banded epsilon/delta_min defaults; tune via calibration later.
BANDS = [
    (0.50, 0.60, 0.08, 0.06),
    (0.60, 0.70, 0.06, 0.05),
    (0.70, 0.80, 0.05, 0.04),
    (0.80, 1.01, 0.04, 0.03),
]
GUARDRAIL_BUCKET_WIDTH = 0.05
FEE_BUFFER = 0.02
KELLY_FRACTION = 0.25  # quarter-Kelly


@dataclass
class DoctrineConfig:
    max_fraction: float
    fee_buffer: float = FEE_BUFFER
    kelly_fraction: float = KELLY_FRACTION
    target_spread_bp: float = 2.0
    require_confluence: bool = False
    enable_bucket_guardrails: bool = False
    require_positive_roi_buckets: bool = False
    bucket_guardrails_path: str = "reports/roi_by_bucket.csv"
    league_min_samples: Optional[Dict[str, int]] = None
    paper_mode_enforce: bool = True
    league: str = "nba"
    negative_roi_buckets: Optional[Dict[str, list]] = None


def band_for_p(p: float) -> Tuple[float, float]:
    for lo, hi, eps, delta_min in BANDS:
        if lo <= p < hi:
            return eps, delta_min
    return BANDS[-1][2], BANDS[-1][3]


def bucket_for_p(p: float) -> str:
    if p is None:
        return "unknown"
    try:
        x = float(p)
    except Exception:
        return "unknown"
    x = max(0.0, min(1.0, x))
    w = GUARDRAIL_BUCKET_WIDTH
    if w <= 0:
        return "unknown"
    idx = int(math.floor((x + 1e-12) / w))
    lo = idx * w
    lo = max(0.0, min(lo, 1.0 - w))
    hi = min(1.0, lo + w)
    return f"[{lo:.2f},{hi:.2f})"


@lru_cache(maxsize=4)
def _load_roi_by_bucket(path: str) -> Dict[str, float]:
    file_path = Path(path)
    rois: Dict[str, float] = {}
    if not file_path.exists():
        return rois
    try:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            bucket = row.get("bucket") or row.get("p_true_bucket")
            roi = row.get("roi_estimate")
            if roi is None:
                roi = row.get("roi")
            if roi is None:
                roi = row.get("roi_estimate_all")
            if bucket is None or roi is None:
                continue
            try:
                rois[str(bucket)] = float(roi)
            except Exception:
                continue
    except Exception:
        return rois
    return rois


def _load_negative_roi_buckets(path: str) -> Dict[str, float]:
    rois = _load_roi_by_bucket(path)
    return {bucket: roi for bucket, roi in rois.items() if roi < 0}


def _count_league_samples(league: str) -> int:
    # prefer per-league ledger if present
    base_dir = Path("reports/specialist_performance")
    candidates = [
        base_dir / f"specialist_manual_ledger_{league}.csv",
        base_dir / "specialist_manual_ledger.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if "league" in df.columns:
                return int((df["league"].str.lower() == league.lower()).sum())
            return len(df)
        except Exception:
            continue
    return 0


def doctrine_decide_trade(
    p_model: float,
    p_market: Optional[float],
    cfg: DoctrineConfig,
    used_fraction: float,
    daily_cap: float,
    internal_prob: Optional[float] = None,
    market_signal: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], str]:
    reason_tags = []
    bucket = bucket_for_p(p_model)
    if p_market is None:
        return None, None, "no_market"

    if cfg.require_confluence and internal_prob is not None and market_signal is not None:
        if (internal_prob > 0.5) != (market_signal > 0.5):
            return None, None, "confluence_mismatch"

    # League sample guardrail (paper-only for under-sampled leagues)
    min_samples = 0
    if cfg.league_min_samples:
        min_samples = cfg.league_min_samples.get(cfg.league, 0) or 0
    if min_samples > 0:
        count = _count_league_samples(cfg.league)
        if count < min_samples:
            reason = f"paper_only_league_min_samples({count}<{min_samples})"
            if cfg.paper_mode_enforce:
                return None, None, reason
            reason_tags.append(reason)

    # Bucket guardrail (config-driven; can be log-only or blocking)
    configured_bad = (cfg.negative_roi_buckets or {}).get(cfg.league, [])
    csv_bad = _load_negative_roi_buckets(cfg.bucket_guardrails_path)
    blocked_buckets = set(configured_bad) | set(csv_bad.keys())

    csv_rois = _load_roi_by_bucket(cfg.bucket_guardrails_path)
    if cfg.enable_bucket_guardrails:
        if bucket in blocked_buckets:
            reason = f"bucket_negative_roi_blocked({bucket})"
            return None, None, reason
        if cfg.require_positive_roi_buckets:
            roi = csv_rois.get(bucket)
            if roi is None:
                return None, None, f"bucket_roi_unknown({bucket})"
            if roi <= 0:
                return None, None, f"bucket_roi_not_positive({bucket},{roi:.4f})"
    else:
        if bucket in blocked_buckets:
            reason_tags.append(f"bucket_negative_roi_would_block({bucket})")
        if cfg.require_positive_roi_buckets:
            roi = csv_rois.get(bucket)
            if roi is None:
                reason_tags.append(f"bucket_roi_unknown_would_block({bucket})")
            elif roi <= 0:
                reason_tags.append(f"bucket_roi_not_positive_would_block({bucket},{roi:.4f})")

    eps, delta_min = band_for_p(p_model)
    q_min = max(0.0, p_model - eps)
    edge_req = delta_min + cfg.fee_buffer
    if q_min <= p_market + edge_req:
        return None, None, "below_delta_min"

    stake = compute_stake_fraction(q_min, p_market, max_fraction=cfg.max_fraction)
    stake *= cfg.kelly_fraction
    if stake <= 0:
        return None, None, "kelly_zero"
    if used_fraction + stake > daily_cap:
        return None, None, "daily_cap"

    target_price = max(0.01, min(p_market, p_model - (cfg.target_spread_bp / 100.0)))
    reason_final = "ok"
    if reason_tags:
        reason_final = f"ok({'|'.join(reason_tags)})"
    return stake, target_price, reason_final
