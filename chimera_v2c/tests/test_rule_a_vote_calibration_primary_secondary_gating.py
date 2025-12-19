from __future__ import annotations

from chimera_v2c.src.rule_a_model_eligibility import ModelEligibility
from chimera_v2c.src.rule_a_unit_policy import RuleAGame
from chimera_v2c.src.rule_a_vote_calibration import contracts_for_game_with_eligibility


def test_secondary_triggers_only_when_primary_triggers() -> None:
    g = RuleAGame(
        date="2025-12-01",
        league="nhl",
        matchup="AAA@BBB",
        mid_home=0.60,
        price_away=0.40,
        home_win=0,
        probs={"primary": 0.50, "secondary": 0.50},
    )
    elig = ModelEligibility(
        league="nhl",
        trained_through="2025-12-01",
        min_games=10,
        confidence_level=0.9,
        bootstrap_sims=2000,
        primary_models=["primary"],
        secondary_models=["secondary"],
        perf_by_model={},
    )

    # Configure gates such that only secondary would vote.
    c, triggers = contracts_for_game_with_eligibility(
        g,
        models=["primary", "secondary"],
        vote_delta_default=0.05,
        vote_delta_by_model={"primary": 0.20, "secondary": 0.05},
        vote_edge_default=-1.0,
        vote_edge_by_model={},
        flip_delta_mode="none",
        vote_weight=1,
        flip_weight=2,
        cap_units=10,
        base_units=0,
        eligibility=elig,
    )
    assert c == 0
    assert triggers == []

    # Now allow primary to vote too; secondary should be counted.
    c2, triggers2 = contracts_for_game_with_eligibility(
        g,
        models=["primary", "secondary"],
        vote_delta_default=0.05,
        vote_delta_by_model={"primary": 0.05, "secondary": 0.05},
        vote_edge_default=-1.0,
        vote_edge_by_model={},
        flip_delta_mode="none",
        vote_weight=1,
        flip_weight=2,
        cap_units=10,
        base_units=0,
        eligibility=elig,
    )
    assert c2 == 2
    assert any(t.endswith(":primary") for t in triggers2)
    assert any(t.endswith(":secondary") for t in triggers2)

