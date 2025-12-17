from __future__ import annotations

from chimera_v2c.tools.walkforward_agreement_stacking_policy import Sample, _bucket_for_side, _walkforward_rows


def test_bucket_for_side_matches_quadrant_definitions() -> None:
    assert _bucket_for_side(p_mid=0.60, side="away") == "A"
    assert _bucket_for_side(p_mid=0.60, side="home") == "B"
    assert _bucket_for_side(p_mid=0.40, side="away") == "C"
    assert _bucket_for_side(p_mid=0.40, side="home") == "D"


def test_disabling_buckets_prevents_trades() -> None:
    # Train: bucket B would be positive for both models.
    # Test: both models would signal home (bucket B), but if we disable B then nothing trades.
    samples = [
        Sample(date="2025-12-01", league="nba", matchup="A@B", p_mid=0.60, y=1, probs={"m1": 0.62, "m2": 0.62}),
        Sample(date="2025-12-02", league="nba", matchup="C@D", p_mid=0.60, y=1, probs={"m1": 0.62, "m2": 0.62}),
    ]
    rows, totals_agree, totals_stack = _walkforward_rows(
        samples_all=samples,
        models=["m1", "m2"],
        league_filter="nba",
        pool_leagues=False,
        train_days=0,
        min_train_days=1,
        min_bets=1,
        ev_threshold=0.0,
        candidates=[0.02],
        threshold_select_mode="max_total_pnl",
        min_agree=2,
        max_units=0,
        active_buckets={"A", "C"},
    )
    day_rows = [r for r in rows if r["test_date"] != "OVERALL"]
    assert len(day_rows) == 1
    assert day_rows[0]["AGREE_ONLY_units"] == 0
    assert day_rows[0]["STACK_ALL_units"] == 0
    assert totals_agree.units == 0
    assert totals_stack.units == 0


def test_agreement_policy_skips_disagreement_and_sizes_by_count() -> None:
    # Train day: allow bucket A for both models (positive), so both can signal "away".
    # Test day: one model signals "away", the other signals "home" -> disagreement -> skip AGREE_ONLY.
    # Baseline STACK_ALL still counts both units.
    samples = [
        # Train bucket A: both models signal away and away wins => +0.60 per unit.
        Sample(date="2025-12-01", league="nba", matchup="A@B", p_mid=0.60, y=0, probs={"m1": 0.58, "m2": 0.58}),
        # Train bucket B for m2: m2 signals home and home wins => +0.40 per unit.
        Sample(date="2025-12-01", league="nba", matchup="E@F", p_mid=0.60, y=1, probs={"m1": 0.58, "m2": 0.62}),
        Sample(date="2025-12-02", league="nba", matchup="C@D", p_mid=0.60, y=0, probs={"m1": 0.58, "m2": 0.62}),
    ]
    rows, totals_agree, totals_stack = _walkforward_rows(
        samples_all=samples,
        models=["m1", "m2"],
        league_filter="nba",
        pool_leagues=False,
        train_days=0,
        min_train_days=1,
        min_bets=1,
        ev_threshold=0.0,
        candidates=[0.02],
        threshold_select_mode="max_total_pnl",
        min_agree=2,
        max_units=0,
        active_buckets={"A", "B", "C", "D"},
    )
    day_rows = [r for r in rows if r["test_date"] != "OVERALL"]
    assert len(day_rows) == 1
    r = day_rows[0]
    assert r["AGREE_ONLY_bets"] == 0
    assert r["STACK_ALL_bets"] == 1
    assert r["STACK_ALL_units"] == 2
    assert totals_agree.bets == 0
    assert totals_stack.units == 2


def test_agreement_policy_trades_when_both_agree_and_can_cap_units() -> None:
    # Train: both models bucket A positive.
    # Test: both models signal "away" -> agreement, min_agree=2 => trade.
    samples = [
        Sample(date="2025-12-01", league="nba", matchup="A@B", p_mid=0.60, y=0, probs={"m1": 0.58, "m2": 0.58}),
        Sample(date="2025-12-02", league="nba", matchup="C@D", p_mid=0.60, y=0, probs={"m1": 0.58, "m2": 0.58}),
    ]
    rows, totals_agree, _ = _walkforward_rows(
        samples_all=samples,
        models=["m1", "m2"],
        league_filter="nba",
        pool_leagues=False,
        train_days=0,
        min_train_days=1,
        min_bets=1,
        ev_threshold=0.0,
        candidates=[0.02],
        threshold_select_mode="max_total_pnl",
        min_agree=2,
        max_units=1,
        active_buckets={"A", "B", "C", "D"},
    )
    day_rows = [r for r in rows if r["test_date"] != "OVERALL"]
    assert len(day_rows) == 1
    r = day_rows[0]
    assert r["AGREE_ONLY_bets"] == 1
    assert r["AGREE_ONLY_units"] == 1  # capped from 2 to 1
    assert totals_agree.bets == 1
    assert totals_agree.units == 1
    assert abs(totals_agree.total_pnl - 0.60) < 1e-12
