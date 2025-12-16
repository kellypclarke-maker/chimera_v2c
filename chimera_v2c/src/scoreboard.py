from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from chimera_v2c.src.ledger_analysis import GameRow


def prob_bucket(p: float, bucket_width: float = 0.1) -> str:
    """
    Map a probability p in [0,1] to a bucket label like [0.2,0.3).
    The top bucket is closed at 1.0: [0.9,1.0].
    """
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    if p >= 1.0 - 1e-9:
        hi = 1.0
        lo = hi - bucket_width
        if lo < 0.0:
            lo = 0.0
        return f"[{lo:.1f},{hi:.1f}]"
    idx = int(p // bucket_width)
    lo = idx * bucket_width
    hi = lo + bucket_width
    if hi > 1.0:
        hi = 1.0
    return f"[{lo:.1f},{hi:.1f})"


def sanitize_games(games: Iterable[GameRow], models: Sequence[str]) -> List[GameRow]:
    """
    Ensure probability-like fields are in [0,1] so downstream metrics don't
    accidentally treat moneylines/garbage as probabilities.
    """
    keep = set(models)
    out: List[GameRow] = []
    for g in games:
        kalshi_mid = g.kalshi_mid
        if kalshi_mid is not None:
            try:
                km = float(kalshi_mid)
                if 0.0 <= km <= 1.0:
                    kalshi_mid = km
                else:
                    kalshi_mid = None
            except Exception:
                kalshi_mid = None

        probs: Dict[str, float] = {}
        for m, p in (g.probs or {}).items():
            if m not in keep:
                continue
            try:
                x = float(p)
            except Exception:
                continue
            if 0.0 <= x <= 1.0:
                probs[m] = x
        out.append(
            GameRow(
                date=g.date,
                league=g.league,
                matchup=g.matchup,
                kalshi_mid=kalshi_mid,
                probs=probs,
                home_win=g.home_win,
            )
        )
    return out


@dataclass
class AccuracyStats:
    n: int = 0
    wins: int = 0

    @property
    def acc(self) -> Optional[float]:
        if self.n == 0:
            return None
        return self.wins / self.n


def compute_accuracy(games: Iterable[GameRow], models: Sequence[str]) -> Dict[str, AccuracyStats]:
    stats: Dict[str, AccuracyStats] = {m: AccuracyStats() for m in models}
    for g in games:
        y = g.home_win
        if y is None or y == 0.5:
            continue
        for m in models:
            p = g.probs.get(m)
            if p is None:
                continue
            s = stats[m]
            s.n += 1
            pick_home = p >= 0.5
            correct = (y == 1.0 and pick_home) or (y == 0.0 and not pick_home)
            if correct:
                s.wins += 1
    return stats


@dataclass
class BucketStats:
    n: int = 0
    sum_p: float = 0.0
    sum_y: float = 0.0
    sum_sq: float = 0.0
    bets: int = 0
    pnl_sum: float = 0.0

    @property
    def avg_p(self) -> Optional[float]:
        if self.n == 0:
            return None
        return self.sum_p / self.n

    @property
    def actual_rate(self) -> Optional[float]:
        if self.n == 0:
            return None
        return self.sum_y / self.n

    @property
    def brier(self) -> Optional[float]:
        if self.n == 0:
            return None
        return self.sum_sq / self.n

    @property
    def avg_pnl(self) -> Optional[float]:
        if self.bets == 0:
            return None
        return self.pnl_sum / self.bets


def compute_reliability_by_bucket(
    games: Iterable[GameRow],
    models: Sequence[str],
    *,
    bucket_width: float = 0.1,
) -> Dict[str, Dict[str, BucketStats]]:
    """
    Compute per-model, per-probability-bucket reliability and EV-vs-mid stats.

    EV convention matches chimera_v2c/src/ledger_analysis.py:
      - If p_model > kalshi_mid: treat as long home at price kalshi_mid.
      - Else: treat as effectively long away at price kalshi_mid.
    """
    stats: Dict[str, Dict[str, BucketStats]] = {m: {} for m in models}
    for g in games:
        y = g.home_win
        if y is None or y == 0.5:
            continue
        kalshi = g.kalshi_mid
        for m in models:
            p = g.probs.get(m)
            if p is None:
                continue
            b = prob_bucket(p, bucket_width=bucket_width)
            s = stats[m].setdefault(b, BucketStats())
            s.n += 1
            s.sum_p += p
            s.sum_y += y
            s.sum_sq += (p - y) ** 2

            if m == "kalshi_mid" or kalshi is None or p == kalshi:
                continue

            price = float(kalshi)
            price = max(0.01, min(0.99, price))
            if p > price:
                pnl = (1.0 - price) if y == 1.0 else -price
            else:
                pnl = price if y == 0.0 else -(1.0 - price)
            s.bets += 1
            s.pnl_sum += pnl
    return stats

