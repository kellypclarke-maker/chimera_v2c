from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from chimera_v2c.src.ledger_analysis import GameRow


# Default bucket set used by the quadrant evaluator tool.
# Letters are intentionally stable so operator notes and docs remain consistent.
DEFAULT_BUCKETS = ["A", "B", "C", "D", "I", "J", "K", "L", "M", "N", "O", "P"]


@dataclass
class BucketStats:
    bets: int = 0
    wins: int = 0
    total_pnl: float = 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.bets if self.bets else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.bets if self.bets else 0.0


def iter_graded_games(games: Iterable[GameRow]) -> Iterable[GameRow]:
    for g in games:
        if g.home_win is None or g.home_win == 0.5:
            continue
        if g.kalshi_mid is None:
            continue
        yield g


def pnl_buy_home(p_mid: float, home_win: float) -> float:
    """
    PnL (1 unit) for buying HOME YES at price p_mid.
      - home win:  +(1 - p_mid)
      - home loss: -p_mid
    """
    p_mid = max(0.01, min(0.99, float(p_mid)))
    return (1.0 - p_mid) if home_win == 1.0 else (-p_mid)


def pnl_buy_away(p_mid: float, home_win: float) -> float:
    """
    PnL (1 unit) for buying AWAY YES at price (1 - p_mid).
      - away win (home loss): +p_mid
      - away loss (home win): -(1 - p_mid)
    """
    p_mid = max(0.01, min(0.99, float(p_mid)))
    p_away = 1.0 - p_mid
    return (1.0 - p_away) if home_win == 0.0 else (-p_away)


def bucket_letters(
    *,
    p_mid: float,
    p_model: float,
    edge_threshold: float,
) -> List[str]:
    """
    Return the bucket letters triggered by this (market,model) pair.

    All probabilities are *home* win probabilities:
      - Market baseline: p_mid = kalshi_mid = p_home(mid)
      - Model:           p_model = p_home(model)

    Edge threshold t:
      - "home rich":  p_model <= p_mid - t  (model less bullish on home)
      - "home cheap": p_model >= p_mid + t  (model more bullish on home)

    Buckets:
      Market home-fav (p_mid >= 0.5)
        A: Fade home (buy away) where home rich (I + J)
          I: p_model < 0.5
          J: p_model >= 0.5
        B: Follow home (buy home) where home cheap (M + N)
          M: p_model >= 0.5
          N: p_model < 0.5

      Market away-fav (p_mid < 0.5)
        C: Follow away (buy away) where home rich (O + P)
          O: p_model < 0.5
          P: p_model >= 0.5
        D: Fade away (buy home) where home cheap (K + L)
          K: p_model >= 0.5
          L: p_model < 0.5

    Note: some sub-buckets (N, P) are theoretically unreachable given the
    parent bucket conditions, but are kept as stable labels for ops/docs.
    """
    t = float(edge_threshold)
    if t <= 0:
        raise ValueError("edge_threshold must be > 0")

    p_mid_f = float(p_mid)
    p_model_f = float(p_model)
    # Protect against float drift when the ledgers store rounded probabilities.
    eps = 1e-12

    if p_mid_f >= 0.5:
        if p_model_f <= (p_mid_f - t + eps):
            return ["A", "I" if p_model_f < 0.5 else "J"]
        if p_model_f >= (p_mid_f + t - eps):
            return ["B", "M" if p_model_f >= 0.5 else "N"]
        return []

    if p_model_f <= (p_mid_f - t + eps):
        return ["C", "O" if p_model_f < 0.5 else "P"]
    if p_model_f >= (p_mid_f + t - eps):
        return ["D", "K" if p_model_f >= 0.5 else "L"]
    return []


def trade_side(bucket: str) -> str:
    """
    Return which YES contract is bought for a bucket:
      - 'home' => buy HOME YES @ p_mid
      - 'away' => buy AWAY YES @ (1 - p_mid)
    """
    if bucket in {"A", "C", "I", "J", "O", "P"}:
        return "away"
    return "home"


def compute_bucket_stats(
    games: Iterable[GameRow],
    *,
    models: List[str],
    edge_threshold: float,
    buckets: List[str],
) -> Dict[Tuple[str, str, str], BucketStats]:
    stats: Dict[Tuple[str, str, str], BucketStats] = {}
    bucket_set = set(buckets)

    for g in iter_graded_games(games):
        p_mid = float(g.kalshi_mid)
        y = float(g.home_win)
        for model in models:
            p_model = g.probs.get(model)
            if p_model is None:
                continue
            letters = [
                b
                for b in bucket_letters(p_mid=p_mid, p_model=float(p_model), edge_threshold=float(edge_threshold))
                if b in bucket_set
            ]
            if not letters:
                continue
            for b in letters:
                key = (g.league, model, b)
                s = stats.setdefault(key, BucketStats())
                s.bets += 1
                if trade_side(b) == "home":
                    pnl = pnl_buy_home(p_mid=p_mid, home_win=y)
                    if y == 1.0:
                        s.wins += 1
                else:
                    pnl = pnl_buy_away(p_mid=p_mid, home_win=y)
                    if y == 0.0:
                        s.wins += 1
                s.total_pnl += pnl

    return stats


def is_allowed_bucket(*, stats: BucketStats, min_bets: int, ev_threshold: float) -> bool:
    return stats.bets >= int(min_bets) and stats.avg_pnl >= float(ev_threshold)


def select_threshold_for_bucket(
    *,
    stats_by_threshold: Dict[float, BucketStats],
    min_bets: int,
    ev_threshold: float,
    mode: str = "min_edge",
) -> Optional[float]:
    """
    Select a threshold from a sweep for a single (league,model,bucket).

    Only thresholds that pass the allow-gate (min_bets + ev_threshold) are eligible.

    Modes:
      - min_edge:    smallest threshold among eligible (volume-first)
      - max_avg_pnl: highest avg_pnl among eligible (quality-first)
      - max_total_pnl: highest total_pnl among eligible (pnl-first)
    """
    eligible = [
        (t, s)
        for t, s in stats_by_threshold.items()
        if is_allowed_bucket(stats=s, min_bets=min_bets, ev_threshold=ev_threshold)
    ]
    if not eligible:
        return None

    mode_norm = (mode or "").strip().lower()
    if mode_norm == "min_edge":
        return min(t for t, _ in eligible)
    if mode_norm == "max_avg_pnl":
        eligible.sort(key=lambda x: (x[1].avg_pnl, x[1].bets, -x[0]), reverse=True)
        return eligible[0][0]
    if mode_norm == "max_total_pnl":
        eligible.sort(key=lambda x: (x[1].total_pnl, x[1].avg_pnl, -x[0]), reverse=True)
        return eligible[0][0]
    raise ValueError(f"unknown selection mode: {mode}")
