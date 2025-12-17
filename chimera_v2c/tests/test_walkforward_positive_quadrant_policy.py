from __future__ import annotations

from chimera_v2c.tools.walkforward_positive_quadrant_policy import Sample, _walkforward_rows


def test_walkforward_rows_per_league_vs_pooled() -> None:
    # Train day: NBA bucket A profitable, NHL bucket A unprofitable.
    # Test day: same pattern repeats.
    samples = [
        Sample(date="2025-12-01", league="nba", matchup="A@B", p_mid=0.60, y=0, probs={"grok": 0.58}),
        Sample(date="2025-12-01", league="nhl", matchup="C@D", p_mid=0.60, y=1, probs={"grok": 0.58}),
        Sample(date="2025-12-02", league="nba", matchup="E@F", p_mid=0.60, y=0, probs={"grok": 0.58}),
        Sample(date="2025-12-02", league="nhl", matchup="G@H", p_mid=0.60, y=1, probs={"grok": 0.58}),
    ]

    rows, by_model, combined = _walkforward_rows(
        samples_all=samples,
        models=["grok"],
        league_filter=None,
        pool_leagues=False,
        train_days=0,
        min_train_days=1,
        min_bets=1,
        ev_threshold=0.0,
        candidates=[0.02],
        threshold_select_mode="max_total_pnl",
    )
    # Only the 2nd date emits rows (min_train_days gate); one row per league.
    assert len(rows) == 2
    nba_row = next(r for r in rows if r["league"] == "nba")
    nhl_row = next(r for r in rows if r["league"] == "nhl")
    assert nba_row["bets_grok"] == 1
    assert nhl_row["bets_grok"] == 0
    assert by_model["grok"].bets == 1
    assert abs(by_model["grok"].total_pnl - 0.60) < 1e-12
    assert combined.bets == 1

    pooled_rows, pooled_by_model, pooled_combined = _walkforward_rows(
        samples_all=samples,
        models=["grok"],
        league_filter=None,
        pool_leagues=True,
        train_days=0,
        min_train_days=1,
        min_bets=1,
        ev_threshold=0.0,
        candidates=[0.02],
        threshold_select_mode="max_total_pnl",
    )
    assert len(pooled_rows) == 1
    r = pooled_rows[0]
    assert r["league"] == "overall"
    assert r["bets_grok"] == 2
    assert pooled_by_model["grok"].bets == 2
    assert abs(pooled_by_model["grok"].total_pnl - 0.20) < 1e-12
    assert pooled_combined.bets == 2


def test_walkforward_selects_threshold_by_train_pnl() -> None:
    samples = [
        Sample(date="2025-12-01", league="nba", matchup="A@B", p_mid=0.60, y=0, probs={"grok": 0.58}),
        Sample(date="2025-12-02", league="nba", matchup="E@F", p_mid=0.60, y=0, probs={"grok": 0.58}),
    ]
    rows, _, _ = _walkforward_rows(
        samples_all=samples,
        models=["grok"],
        league_filter="nba",
        pool_leagues=False,
        train_days=0,
        min_train_days=1,
        min_bets=1,
        ev_threshold=0.0,
        candidates=[0.02, 0.05],
        threshold_select_mode="max_total_pnl",
    )
    assert len(rows) == 1
    assert rows[0]["t_grok"] == "0.020"

