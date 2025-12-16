from __future__ import annotations

from datetime import datetime

from chimera_v2c.src.ledger_analysis import GameRow
from chimera_v2c.tools.plan_grok_mid_hybrid_trades import plan_trades


def _row(
    *,
    date_iso: str,
    league: str,
    matchup: str,
    p_mid: float,
    p_grok: float,
    home_win: float | None,
) -> GameRow:
    return GameRow(
        date=datetime.strptime(date_iso, "%Y-%m-%d"),
        league=league,
        matchup=matchup,
        kalshi_mid=p_mid,
        probs={"grok": p_grok},
        home_win=home_win,
    )


def test_plan_trades_respects_train_days_window() -> None:
    games = [
        _row(date_iso="2025-12-01", league="nba", matchup="AAA@BBB", p_mid=0.80, p_grok=0.40, home_win=0.0),
        _row(date_iso="2025-12-02", league="nba", matchup="CCC@DDD", p_mid=0.80, p_grok=0.40, home_win=0.0),
        _row(date_iso="2025-12-03", league="nba", matchup="EEE@FFF", p_mid=0.80, p_grok=0.40, home_win=None),
    ]

    rows, meta = plan_trades(
        games,
        target_date="2025-12-03",
        league_for_stats="nba",
        train_days=1,
        min_train_days=2,
        min_train_samples=2,
        alpha_step=1.0,
        t_min=0.01,
        t_max=0.15,
        t_step=0.01,
        min_bets=1,
        ev_threshold=0.0,
        select_mode="min_edge",
    )

    assert rows  # test set exists
    assert meta["train_days"] == 1
    assert meta["train_samples"] == 1
    assert meta["train_start"] == "2025-12-02"
    assert meta["train_end"] == "2025-12-02"
    assert meta["trained"] is False


def test_plan_trades_produces_bucket_a_trade_on_synthetic_data() -> None:
    games = [
        _row(date_iso="2025-12-01", league="nba", matchup="AAA@BBB", p_mid=0.80, p_grok=0.40, home_win=0.0),
        _row(date_iso="2025-12-02", league="nba", matchup="CCC@DDD", p_mid=0.80, p_grok=0.40, home_win=0.0),
        _row(date_iso="2025-12-03", league="nba", matchup="EEE@FFF", p_mid=0.80, p_grok=0.40, home_win=None),
    ]

    rows, meta = plan_trades(
        games,
        target_date="2025-12-03",
        league_for_stats="nba",
        train_days=0,
        min_train_days=2,
        min_train_samples=2,
        alpha_step=1.0,
        t_min=0.01,
        t_max=0.15,
        t_step=0.01,
        min_bets=1,
        ev_threshold=0.0,
        select_mode="min_edge",
    )

    assert meta["trained"] is True
    assert rows and rows[0]["matchup"] == "EEE@FFF"
    assert rows[0]["decision_bucket"] == "A"
    assert rows[0]["decision_side"] == "away"

