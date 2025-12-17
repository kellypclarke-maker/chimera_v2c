from __future__ import annotations

from datetime import datetime

from chimera_v2c.src.ledger_analysis import GameRow
from chimera_v2c.tools.analyze_positive_quadrant_policy import (
    _policy_summaries,
    is_positive_bucket,
)
from chimera_v2c.src.rulebook_quadrants import BucketStats, compute_bucket_stats


def test_is_positive_bucket_respects_gates() -> None:
    s = BucketStats(bets=9, wins=0, total_pnl=1.0)
    assert is_positive_bucket(s, min_bets=10, ev_threshold=0.0) is False

    s2 = BucketStats(bets=10, wins=0, total_pnl=-0.1)
    assert is_positive_bucket(s2, min_bets=10, ev_threshold=0.0) is False

    s3 = BucketStats(bets=10, wins=0, total_pnl=0.0)
    assert is_positive_bucket(s3, min_bets=10, ev_threshold=0.0) is True


def test_policy_summaries_selects_positive_buckets_and_sums() -> None:
    # Build a tiny synthetic set where only bucket A is positive for grok.
    # Use t=0.10, p_mid=0.60:
    #  - grok=0.40 triggers bucket A (buy away).
    #  - away win yields +p_mid = +0.60 each.
    games = [
        GameRow(
            date=datetime.strptime("2025-12-01", "%Y-%m-%d"),
            league="nba",
            matchup="A@B",
            kalshi_mid=0.60,
            probs={"grok": 0.40},
            home_win=0.0,
        ),
        GameRow(
            date=datetime.strptime("2025-12-02", "%Y-%m-%d"),
            league="nba",
            matchup="C@D",
            kalshi_mid=0.60,
            probs={"grok": 0.40},
            home_win=0.0,
        ),
    ]
    stats = compute_bucket_stats(games, models=["grok"], edge_threshold=0.10, buckets=["A", "B", "C", "D"])
    summaries = _policy_summaries(
        stats_by_key=stats,
        leagues=["nba"],
        models=["grok"],
        edge_threshold=0.10,
        min_bets=2,
        ev_threshold=0.0,
    )
    assert len(summaries) == 1
    s = summaries[0]
    assert s.league == "nba"
    assert s.model == "grok"
    assert s.positive_buckets == ("A",)
    assert s.bets == 2
    assert abs(s.total_pnl - 1.2) < 1e-12

