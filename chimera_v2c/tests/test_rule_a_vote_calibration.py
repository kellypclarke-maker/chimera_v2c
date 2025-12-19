from __future__ import annotations

from chimera_v2c.src.rule_a_unit_policy import RuleAGame
from chimera_v2c.src.rule_a_vote_calibration import (
    choose_flip_delta_mode,
    contracts_for_game,
    train_vote_delta_calibration,
)


def _game(
    *,
    date: str,
    league: str,
    matchup: str = "AAA@BBB",
    mid_home: float,
    price_away: float,
    home_win: int,
    probs: dict[str, float],
) -> RuleAGame:
    return RuleAGame(
        date=date,
        league=league,
        matchup=matchup,
        mid_home=float(mid_home),
        price_away=float(price_away),
        home_win=int(home_win),
        probs=dict(probs),
    )


def test_train_vote_delta_calibration_picks_higher_delta() -> None:
    train = []
    # 10 "good" vote games at delta~0.06 where away wins (should pass a 0.05 threshold).
    for i in range(10):
        train.append(
            _game(
                date=f"2025-11-{19 + i:02d}",
                league="nhl",
                mid_home=0.60,
                price_away=0.55,
                home_win=0,
                probs={"grok": 0.54},
            )
        )
    # 20 "bad" vote games at delta~0.015 where home wins (passes 0.01 but fails 0.05).
    for i in range(20):
        train.append(
            _game(
                date=f"2025-12-{1 + i:02d}",
                league="nhl",
                mid_home=0.60,
                price_away=0.55,
                home_win=1,
                probs={"grok": 0.585},
            )
        )

    calib, totals_c, totals_b, counts = train_vote_delta_calibration(
        train,
        league="nhl",
        trained_through="2025-12-15",
        models=["grok"],
        vote_deltas=[0.0, 0.01, 0.05],
        flip_delta_mode="none",
        cap_units=10,
        base_units=1,
        vote_weight=1,
        flip_weight=2,
        vote_edge_default=-1.0,  # ignore edge gating for this synthetic test
        roi_guardrail_mode="soft",
        roi_epsilon=1.0,
        min_signals=10,
        max_iters=2,
    )

    assert calib.vote_delta_by_model["grok"] == 0.05
    assert counts["grok"] >= 10
    assert totals_c.net_pnl > totals_b.net_pnl


def test_choose_flip_delta_mode_prefers_none_when_flips_profitable() -> None:
    train = []
    # 10 flip games: small delta (0.02), away wins.
    for i in range(10):
        train.append(
            _game(
                date=f"2025-11-{19 + i:02d}",
                league="nhl",
                mid_home=0.51,
                price_away=0.55,
                home_win=0,
                probs={"grok": 0.49},
            )
        )
    # 10 good vote games: delta~0.06, away wins.
    for i in range(10):
        train.append(
            _game(
                date=f"2025-12-{1 + i:02d}",
                league="nhl",
                mid_home=0.60,
                price_away=0.55,
                home_win=0,
                probs={"grok": 0.54},
            )
        )
    # 20 bad vote games: delta~0.015, home wins.
    for i in range(20):
        train.append(
            _game(
                date=f"2025-12-{11 + i:02d}",
                league="nhl",
                mid_home=0.60,
                price_away=0.55,
                home_win=1,
                probs={"grok": 0.585},
            )
        )

    calib, totals_c, totals_b, counts = choose_flip_delta_mode(
        train,
        league="nhl",
        trained_through="2025-12-31",
        models=["grok"],
        vote_deltas=[0.0, 0.01, 0.02, 0.05],
        flip_delta_modes=["none", "same"],
        cap_units=10,
        base_units=1,
        vote_weight=1,
        flip_weight=2,
        vote_edge_default=-1.0,  # ignore edge gating for this synthetic test
        roi_guardrail_mode="soft",
        roi_epsilon=1.0,
        min_signals=10,
        max_iters=2,
    )

    assert counts["grok"] >= 10
    assert calib.vote_delta_by_model["grok"] >= 0.02
    assert calib.flip_delta_mode == "none"
    assert totals_c.net_pnl > totals_b.net_pnl


def test_contracts_for_game_emits_weighted_triggers() -> None:
    g = _game(
        date="2025-12-01",
        league="nhl",
        mid_home=0.60,
        price_away=0.55,
        home_win=0,
        probs={"grok": 0.49, "v2c": 0.54},
    )

    contracts, triggers = contracts_for_game(
        g,
        models=["grok", "v2c"],
        vote_delta_default=0.05,
        vote_delta_by_model={"v2c": 0.05},
        vote_edge_default=-1.0,
        vote_edge_by_model={},
        flip_delta_mode="none",
        vote_weight=1,
        flip_weight=2,
        cap_units=10,
        base_units=1,
    )
    # grok flips => +2, v2c votes => +1, plus base 1
    assert contracts == 4
    assert "grok:flip:2" in triggers
    assert "v2c:vote:1" in triggers
