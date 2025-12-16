from __future__ import annotations

from datetime import datetime

import pytest

from chimera_v2c.src.ledger_analysis import GameRow
from chimera_v2c.src.scoreboard import compute_accuracy, compute_reliability_by_bucket, prob_bucket, sanitize_games


def _g(
    *,
    league: str = "nhl",
    matchup: str = "AAA@BBB",
    kalshi_mid: float | None = 0.5,
    probs: dict[str, float] | None = None,
    home_win: float | None = 1.0,
) -> GameRow:
    return GameRow(
        date=datetime(2025, 12, 13),
        league=league,
        matchup=matchup,
        kalshi_mid=kalshi_mid,
        probs=probs or {},
        home_win=home_win,
    )


def test_prob_bucket_edges() -> None:
    assert prob_bucket(1.0, bucket_width=0.1) == "[0.9,1.0]"
    assert prob_bucket(0.0, bucket_width=0.1) == "[0.0,0.1)"
    assert prob_bucket(0.25, bucket_width=0.1) == "[0.2,0.3)"


def test_sanitize_games_drops_out_of_range_probs_and_mids() -> None:
    games = [
        _g(kalshi_mid=0.52, probs={"v2c": 0.6, "books_home_ml": -120.0}),
        _g(kalshi_mid=1.5, probs={"v2c": 0.7}),
    ]
    sanitized = sanitize_games(games, models=["v2c", "books_home_ml", "kalshi_mid"])
    assert sanitized[0].kalshi_mid == pytest.approx(0.52)
    assert sanitized[0].probs == {"v2c": pytest.approx(0.6)}
    assert sanitized[1].kalshi_mid is None
    assert sanitized[1].probs == {"v2c": pytest.approx(0.7)}


def test_compute_accuracy_counts_only_resolved_non_push() -> None:
    games = [
        _g(probs={"v2c": 0.7}, home_win=1.0),
        _g(probs={"v2c": 0.7}, home_win=0.0),
        _g(probs={"v2c": 0.7}, home_win=None),
        _g(probs={"v2c": 0.7}, home_win=0.5),
    ]
    stats = compute_accuracy(games, models=["v2c"])
    assert stats["v2c"].n == 2
    assert stats["v2c"].wins == 1
    assert stats["v2c"].acc == pytest.approx(0.5)


def test_compute_reliability_by_bucket_accumulates_stats() -> None:
    games = [
        _g(kalshi_mid=0.5, probs={"v2c": 0.7, "kalshi_mid": 0.5}, home_win=1.0),
        _g(kalshi_mid=0.5, probs={"v2c": 0.7, "kalshi_mid": 0.5}, home_win=0.0),
    ]
    stats = compute_reliability_by_bucket(games, models=["v2c", "kalshi_mid"], bucket_width=0.1)
    bucket = prob_bucket(0.7, bucket_width=0.1)
    s = stats["v2c"][bucket]
    assert s.n == 2
    assert s.bets == 2  # p != mid in both games
    assert s.avg_p == pytest.approx(0.7)
    assert s.actual_rate == pytest.approx(0.5)

