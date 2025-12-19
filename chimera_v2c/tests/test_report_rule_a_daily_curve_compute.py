from __future__ import annotations

from chimera_v2c.src.rule_a_unit_policy import RuleAGame
from chimera_v2c.src.rule_a_vote_calibration import VoteDeltaCalibration
from chimera_v2c.tools.report_rule_a_daily_curve import WeakBucketPolicy, _compute_daily_rows


def test_compute_daily_rows_applies_weak_bucket_cap() -> None:
    g = RuleAGame(
        date="2025-12-01",
        league="nhl",
        matchup="AAA@BBB",
        mid_home=0.62,  # bucket 0.60-0.65
        price_away=0.40,
        home_win=0,
        probs={"v2c": 0.55},
    )
    cfg = VoteDeltaCalibration(
        league="nhl",
        trained_through="2025-12-01",
        models=["v2c"],
        vote_delta_by_model={"v2c": 0.0},
        vote_delta_default=0.0,
        vote_edge_by_model={},
        vote_edge_default=0.0,
        flip_delta_mode="none",
        vote_weight=1,
        flip_weight=2,
        cap_units=10,
        base_units=1,
        roi_guardrail_mode="soft",
        roi_epsilon=0.0,
    )

    # Without weak cap, contracts = base 1 + vote 1 => 2.
    rows = _compute_daily_rows(
        [g],
        league="nhl",
        models=["v2c"],
        vote_cfg=cfg,
        eligibility=None,
        weak_policy=None,
        default_vote_delta=0.01,
        default_vote_edge=0.0,
        default_vote_weight=1,
        default_flip_weight=2,
        default_flip_delta_mode="none",
        default_base_units=1,
        default_cap_units=10,
    )
    assert rows[0]["contracts"] == 2

    # With weak cap=1 in that bucket, contracts should be 1.
    rows2 = _compute_daily_rows(
        [g],
        league="nhl",
        models=["v2c"],
        vote_cfg=cfg,
        eligibility=None,
        weak_policy=WeakBucketPolicy(weak_buckets=["0.60-0.65"], weak_bucket_cap=1),
        default_vote_delta=0.01,
        default_vote_edge=0.0,
        default_vote_weight=1,
        default_flip_weight=2,
        default_flip_delta_mode="none",
        default_base_units=1,
        default_cap_units=10,
    )
    assert rows2[0]["contracts"] == 1
