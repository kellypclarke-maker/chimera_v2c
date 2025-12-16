from __future__ import annotations

from datetime import datetime

from chimera_v2c.src.ledger_analysis import GameRow
from chimera_v2c.src.rulebook_quadrants import (
    BucketStats,
    bucket_letters,
    compute_bucket_stats,
    select_threshold_for_bucket,
)


def test_rulebook_quadrants_bucket_letters_home_fav_fade_hard_flip() -> None:
    assert bucket_letters(p_mid=0.55, p_model=0.44, edge_threshold=0.05) == ["A", "I"]


def test_rulebook_quadrants_bucket_letters_home_fav_fade_soft() -> None:
    assert bucket_letters(p_mid=0.65, p_model=0.56, edge_threshold=0.05) == ["A", "J"]


def test_rulebook_quadrants_bucket_letters_away_fav_fade_soft() -> None:
    # Market favors away (p_mid < 0.5), model says home is cheap by >= t => buy home (D).
    assert bucket_letters(p_mid=0.45, p_model=0.49, edge_threshold=0.03) == ["D", "L"]


def test_rulebook_quadrants_compute_bucket_stats_pnl_home_fav_fade() -> None:
    games = [
        GameRow(
            date=datetime(2025, 12, 13),
            league="nhl",
            matchup="X@Y",
            kalshi_mid=0.60,
            probs={"grok": 0.50},
            home_win=0.0,  # away wins
        )
    ]
    stats = compute_bucket_stats(games, models=["grok"], edge_threshold=0.05, buckets=["A", "J"])
    assert stats[("nhl", "grok", "A")].bets == 1
    assert stats[("nhl", "grok", "A")].wins == 1
    assert stats[("nhl", "grok", "A")].total_pnl == 0.60
    assert stats[("nhl", "grok", "J")].total_pnl == 0.60


def test_rulebook_quadrants_select_threshold_for_bucket_modes() -> None:
    stats_by_threshold = {
        0.03: BucketStats(bets=15, wins=0, total_pnl=15 * 0.12),
        0.05: BucketStats(bets=10, wins=0, total_pnl=10 * 0.15),
    }
    assert (
        select_threshold_for_bucket(
            stats_by_threshold=stats_by_threshold, min_bets=10, ev_threshold=0.10, mode="min_edge"
        )
        == 0.03
    )
    assert (
        select_threshold_for_bucket(
            stats_by_threshold=stats_by_threshold, min_bets=10, ev_threshold=0.10, mode="max_avg_pnl"
        )
        == 0.05
    )
    assert (
        select_threshold_for_bucket(
            stats_by_threshold=stats_by_threshold, min_bets=10, ev_threshold=0.10, mode="max_total_pnl"
        )
        == 0.03
    )

