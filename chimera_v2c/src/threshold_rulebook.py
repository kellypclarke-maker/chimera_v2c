from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from chimera_v2c.src.ledger_analysis import GameRow
from chimera_v2c.src.offset_calibration import clamp_prob
from chimera_v2c.src.rulebook_quadrants import BucketStats, compute_bucket_stats, select_threshold_for_bucket


def edge_thresholds(min_edge: float, max_edge: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if max_edge < min_edge:
        raise ValueError("max_edge must be >= min_edge")
    if min_edge <= 0:
        raise ValueError("min_edge must be > 0")
    out: List[float] = []
    cur = float(min_edge)
    while cur <= float(max_edge) + 1e-9:
        out.append(round(cur, 3))
        cur += float(step)
    return out


def apply_offset_biases(
    games: Iterable[GameRow],
    *,
    bias_by_model: Dict[str, float],
) -> List[GameRow]:
    """
    Return a new list of GameRow with p_model replaced by clamp(p + bias) for any model in bias_by_model.
    """
    out: List[GameRow] = []
    for g in games:
        if not g.probs or not bias_by_model:
            out.append(g)
            continue
        new_probs = dict(g.probs)
        changed = False
        for model, bias in bias_by_model.items():
            if model not in new_probs:
                continue
            new_probs[model] = clamp_prob(float(new_probs[model]) + float(bias))
            changed = True
        if not changed:
            out.append(g)
        else:
            out.append(
                GameRow(
                    date=g.date,
                    league=g.league,
                    matchup=g.matchup,
                    kalshi_mid=g.kalshi_mid,
                    probs=new_probs,
                    home_win=g.home_win,
                )
            )
    return out


def sweep_rulebook_stats(
    games: Iterable[GameRow],
    *,
    thresholds: List[float],
    models: List[str],
    buckets: List[str],
) -> Dict[Tuple[float, str, str, str], BucketStats]:
    """
    Return stats keyed by (edge_threshold, league, model, bucket).
    """
    stats_grid: Dict[Tuple[float, str, str, str], BucketStats] = {}
    for t in thresholds:
        stats = compute_bucket_stats(games, models=models, edge_threshold=t, buckets=buckets)
        for (league, model, bucket), s in stats.items():
            stats_grid[(float(t), league, model, bucket)] = s
    return stats_grid


@dataclass(frozen=True)
class SelectedThreshold:
    league: str
    model: str
    bucket: str
    edge_threshold: float
    stats: BucketStats


def select_thresholds(
    stats_grid: Dict[Tuple[float, str, str, str], BucketStats],
    *,
    thresholds: List[float],
    min_bets: int,
    ev_threshold: float,
    mode: str,
) -> List[SelectedThreshold]:
    """
    Select one threshold per (league,model,bucket) using allow-gates + selection mode.
    """
    # Gather keys present in the grid.
    keys = {(league, model, bucket) for (_, league, model, bucket) in stats_grid.keys()}
    selected: List[SelectedThreshold] = []
    for league, model, bucket in sorted(keys):
        by_t: Dict[float, BucketStats] = {}
        for t in thresholds:
            s = stats_grid.get((float(t), league, model, bucket))
            if s is None:
                continue
            by_t[float(t)] = s
        if not by_t:
            continue
        t_star = select_threshold_for_bucket(
            stats_by_threshold=by_t,
            min_bets=min_bets,
            ev_threshold=ev_threshold,
            mode=mode,
        )
        if t_star is None:
            continue
        s_star = by_t.get(float(t_star))
        if s_star is None:
            continue
        selected.append(
            SelectedThreshold(
                league=league,
                model=model,
                bucket=bucket,
                edge_threshold=float(t_star),
                stats=s_star,
            )
        )
    return selected

