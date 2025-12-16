from __future__ import annotations

from datetime import datetime

from chimera_v2c.src.ledger_analysis import GameRow
from chimera_v2c.tools.online_grok_mid_hybrid_walkthrough import iter_samples, online_walkthrough


def test_online_walkthrough_learns_alpha_from_history_without_leakage() -> None:
    # Three graded games in chronological order (as load_games would provide by ledger date).
    games = [
        GameRow(
            date=datetime(2025, 12, 1),
            league="nba",
            matchup="AAA@BBB",
            kalshi_mid=0.50,
            probs={"grok": 0.90},
            home_win=1.0,
        ),
        GameRow(
            date=datetime(2025, 12, 2),
            league="nba",
            matchup="CCC@DDD",
            kalshi_mid=0.50,
            probs={"grok": 0.10},
            home_win=0.0,
        ),
        GameRow(
            date=datetime(2025, 12, 3),
            league="nba",
            matchup="EEE@FFF",
            kalshi_mid=0.50,
            probs={"grok": 0.90},
            home_win=1.0,
        ),
    ]

    samples = iter_samples(games)
    rows, summary = online_walkthrough(samples, min_train_samples=999, alpha_step=0.5)

    # Basic sanity: one row per game.
    assert summary["n"] == 3
    assert len(rows) == 3

    # First game: no training history => alpha must be 0.
    assert rows[0]["n_train"] == 0
    assert rows[0]["alpha"] == "0.000"

    # Second/third games: alpha should move toward Grok (1.0 on this toy dataset).
    assert rows[1]["alpha"] == "1.000"
    assert rows[2]["alpha"] == "1.000"


def test_online_walkthrough_date_cv_requires_two_dates() -> None:
    games = [
        GameRow(
            date=datetime(2025, 12, 1),
            league="nba",
            matchup="AAA@BBB",
            kalshi_mid=0.50,
            probs={"grok": 0.90},
            home_win=1.0,
        ),
        GameRow(
            date=datetime(2025, 12, 1),
            league="nba",
            matchup="CCC@DDD",
            kalshi_mid=0.50,
            probs={"grok": 0.10},
            home_win=0.0,
        ),
        GameRow(
            date=datetime(2025, 12, 2),
            league="nba",
            matchup="EEE@FFF",
            kalshi_mid=0.50,
            probs={"grok": 0.90},
            home_win=1.0,
        ),
    ]

    samples = iter_samples(games)
    rows, summary = online_walkthrough(samples, min_train_samples=999, alpha_step=0.5, alpha_mode="date_cv")
    assert summary["n"] == 3
    # First game: no history.
    assert rows[0]["alpha"] == "0.000"
    # Second game: train dates < 2 (only 2025-12-01) => alpha must still be 0.
    assert rows[1]["alpha"] == "0.000"
    # Third game: train has 2 games but only one distinct date (both 2025-12-01) => still 0.
    assert rows[2]["alpha"] == "0.000"
