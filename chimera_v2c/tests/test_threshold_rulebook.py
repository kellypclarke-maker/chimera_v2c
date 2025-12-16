from __future__ import annotations

from datetime import datetime

import pytest

from chimera_v2c.src.ledger_analysis import GameRow
from chimera_v2c.src.threshold_rulebook import (
    apply_offset_biases,
    edge_thresholds,
    select_thresholds,
    sweep_rulebook_stats,
)


def test_threshold_rulebook_edge_thresholds_grid() -> None:
    assert edge_thresholds(0.01, 0.03, 0.01) == [0.01, 0.02, 0.03]


def test_threshold_rulebook_apply_offset_biases_clamps() -> None:
    games = [
        GameRow(
            date=datetime(2025, 12, 1),
            league="nba",
            matchup="A@B",
            kalshi_mid=0.55,
            probs={"grok": 0.95},
            home_win=1.0,
        )
    ]
    out = apply_offset_biases(games, bias_by_model={"grok": 0.10})
    assert out[0].probs["grok"] == 1.0


def test_threshold_rulebook_select_thresholds_modes() -> None:
    # Construct a dataset where:
    # - At t=0.02, bucket A includes 20 bets with avg_pnl=+0.10.
    # - At t=0.05, bucket A includes 10 bets with avg_pnl=+0.60.
    games = []
    for i in range(10):
        games.append(
            GameRow(
                date=datetime(2025, 12, 1),
                league="nba",
                matchup=f"X{i}@Y{i}",
                kalshi_mid=0.60,
                probs={"grok": 0.55},  # edge=-0.05
                home_win=0.0,  # away wins => fading home is profitable
            )
        )
    for i in range(10):
        games.append(
            GameRow(
                date=datetime(2025, 12, 1),
                league="nba",
                matchup=f"Z{i}@W{i}",
                kalshi_mid=0.60,
                probs={"grok": 0.58},  # edge=-0.02
                home_win=1.0,  # home wins => fading home is unprofitable
            )
        )

    thresholds = [0.02, 0.05]
    stats_grid = sweep_rulebook_stats(games, thresholds=thresholds, models=["grok"], buckets=["A"])

    selected_min = select_thresholds(stats_grid, thresholds=thresholds, min_bets=10, ev_threshold=0.05, mode="min_edge")
    assert len(selected_min) == 1
    assert selected_min[0].bucket == "A"
    assert selected_min[0].edge_threshold == pytest.approx(0.02)

    selected_max = select_thresholds(stats_grid, thresholds=thresholds, min_bets=10, ev_threshold=0.05, mode="max_avg_pnl")
    assert len(selected_max) == 1
    assert selected_max[0].edge_threshold == pytest.approx(0.05)

