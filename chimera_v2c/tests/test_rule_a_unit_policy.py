from __future__ import annotations

from chimera_v2c.src.rule_a_unit_policy import (
    ModelPolicy,
    PolicyCalibration,
    RuleAGame,
    SignalMetrics,
    net_pnl_taker,
    select_model_policy,
    select_unit_scale,
    signal_metrics,
    units_for_game,
)


def _g(
    *,
    date: str = "2025-12-01",
    league: str = "nba",
    matchup: str = "AAA@BBB",
    mid_home: float = 0.60,
    price_away: float = 0.40,
    home_win: int | None = 0,
    probs: dict[str, float] | None = None,
) -> RuleAGame:
    return RuleAGame(
        date=date,
        league=league,
        matchup=matchup,
        mid_home=float(mid_home),
        price_away=float(price_away),
        home_win=None if home_win is None else int(home_win),
        probs=dict(probs or {}),
    )


def test_rule_a_unit_policy_signal_metrics_constant_samples() -> None:
    m = signal_metrics(
        [0.05] * 10,
        threshold=0.02,
        confidence_level=0.9,
        bootstrap_sims=100,
        seed=123,
    )
    assert m is not None
    assert m.n == 10
    assert abs(m.mean_net - 0.05) < 1e-12
    assert abs(m.mean_low - 0.05) < 1e-12
    assert abs(m.conf_gt0 - 1.0) < 1e-12


def test_rule_a_unit_policy_select_model_policy_picks_threshold() -> None:
    gross_win, fees_win, _ = net_pnl_taker(contracts=1, price_away=0.40, home_win=0)
    net_win = gross_win - fees_win
    gross_lose, fees_lose, _ = net_pnl_taker(contracts=1, price_away=0.40, home_win=1)
    net_lose = gross_lose - fees_lose

    # Strong signals have positive expected edge and win; weak signals have small positive expected edge but lose.
    train = [
        _g(probs={"m": 0.50}, home_win=0),
        _g(probs={"m": 0.50}, home_win=0),
        _g(probs={"m": 0.57}, home_win=1),
        _g(probs={"m": 0.57}, home_win=1),
        _g(probs={"m": 0.57}, home_win=1),
        _g(probs={"m": 0.57}, home_win=1),
    ]

    mp = select_model_policy(
        train,
        model="m",
        thresholds=[0.0, 0.02, 0.12],
        min_signals=2,
        confidence_level=0.9,
        bootstrap_sims=200,
        seed=7,
        select_mode="max_total_mean_low",
    )
    assert mp is not None
    assert abs(mp.threshold - 0.02) < 1e-12
    assert abs(mp.weight - net_win) < 1e-12
    assert mp.metrics.n == 2
    assert abs(mp.metrics.mean_net - net_win) < 1e-12
    assert abs(mp.metrics.mean_low - net_win) < 1e-12
    assert abs(net_lose) > 0  # sanity check: the weak-signal loss is not accidentally zero


def test_rule_a_unit_policy_units_for_game_scales_and_caps() -> None:
    policy = PolicyCalibration(
        league="nba",
        confidence_level=0.9,
        unit_scale=0.10,
        models={
            "m1": ModelPolicy(model="m1", threshold=0.02, weight=0.15, metrics=SignalMetrics(0.02, 10, 0.15, 0.15, 1.0)),
            "m2": ModelPolicy(model="m2", threshold=0.02, weight=0.15, metrics=SignalMetrics(0.02, 10, 0.15, 0.15, 1.0)),
        },
    )
    g = _g(probs={"m1": 0.40, "m2": 0.40}, mid_home=0.60, home_win=0)
    contracts, score, triggering = units_for_game(g, policy=policy, cap_units=10)
    assert triggering == ["m1", "m2"]
    assert abs(score - 0.36) < 1e-12
    assert contracts == 4  # 1 + floor(0.36 / 0.10)

    contracts_cap, _, _ = units_for_game(g, policy=policy, cap_units=3)
    assert contracts_cap == 3


def test_rule_a_unit_policy_select_unit_scale_respects_roi_floor() -> None:
    # Baseline: 1 unit on each game -> positive ROI.
    train = [
        _g(date="2025-12-01", matchup="A@B", home_win=0, probs={"m1": 0.40}),  # away win (only m1 triggers)
        _g(date="2025-12-01", matchup="C@D", home_win=1, probs={"m1": 0.40, "m2": 0.40}),  # home win
    ]
    models = {
        # m2 is missing on the winning game, so a small unit_scale over-sizes the losing game.
        "m1": ModelPolicy(model="m1", threshold=0.0, weight=0.10, metrics=SignalMetrics(0.0, 2, 0.10, 0.10, 1.0)),
        "m2": ModelPolicy(model="m2", threshold=0.0, weight=0.10, metrics=SignalMetrics(0.0, 2, 0.10, 0.10, 1.0)),
    }
    # Force extra units on both games; ROI-floor should reject if it underperforms baseline ROI.
    best = select_unit_scale(train, models=models, unit_scales=[0.05], cap_units=10, roi_floor_mult=0.9)
    assert best == 1e9


def test_rule_a_unit_policy_select_unit_scale_can_choose_profitable_scale() -> None:
    train = [
        _g(date="2025-12-01", matchup="A@B", home_win=0, probs={"m1": 0.40}),
        _g(date="2025-12-01", matchup="C@D", home_win=0, probs={"m1": 0.40}),
    ]
    models = {
        "m1": ModelPolicy(model="m1", threshold=0.0, weight=0.20, metrics=SignalMetrics(0.0, 2, 0.20, 0.20, 1.0)),
    }
    best = select_unit_scale(train, models=models, unit_scales=[0.10], cap_units=10, roi_floor_mult=0.9)
    assert abs(best - 0.10) < 1e-12
