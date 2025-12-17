from __future__ import annotations

from chimera_v2c.src.rulebook_policy import TradeDecision
from chimera_v2c.tools.walkforward_grok_mid_hybrid_consensus_backtest import (
    Sample,
    _confirmer_agrees,
    _evaluate_consensus,
    _units_for_models,
)


def test_units_for_models_mapping() -> None:
    assert _units_for_models(0, (1, 3, 5)) == 0
    assert _units_for_models(1, (1, 3, 5)) == 1
    assert _units_for_models(2, (1, 3, 5)) == 3
    assert _units_for_models(3, (1, 3, 5)) == 5
    assert _units_for_models(10, (1, 3, 5)) == 5


def test_confirmer_agrees_uses_bucket_threshold_and_side() -> None:
    # Home side agreement: p_model must be >= p_mid + t.
    dec_home = TradeDecision(bucket="D", side="home", edge_threshold=0.05)
    assert _confirmer_agrees(p_mid=0.45, p_model=0.51, decision=dec_home) is True
    assert _confirmer_agrees(p_mid=0.45, p_model=0.49, decision=dec_home) is False

    # Away side agreement: p_model must be <= p_mid - t.
    dec_away = TradeDecision(bucket="A", side="away", edge_threshold=0.10)
    assert _confirmer_agrees(p_mid=0.60, p_model=0.49, decision=dec_away) is True
    assert _confirmer_agrees(p_mid=0.60, p_model=0.55, decision=dec_away) is False


def test_evaluate_consensus_gates_on_confirmer_and_positive_bucket_filter() -> None:
    # Two samples on the same day; the primary model triggers bucket A (buy away).
    test = [
        Sample(
            date="2025-12-02",
            league="nba",
            matchup="A@B",
            p_grok=0.40,
            p_mid=0.60,
            y=0,  # away wins
            p_confirmers={"market_proxy": 0.45},  # agrees (<= 0.50)
        ),
        Sample(
            date="2025-12-02",
            league="nba",
            matchup="C@D",
            p_grok=0.40,
            p_mid=0.60,
            y=0,  # away wins
            p_confirmers={"market_proxy": 0.55},  # disagrees
        ),
    ]

    # Primary p_model always triggers bucket A at t=0.10.
    p_primary = {
        ("2025-12-02", "nba", "A@B"): 0.40,
        ("2025-12-02", "nba", "C@D"): 0.40,
    }
    thresholds = {"A": 0.10}

    bets, units, pnl = _evaluate_consensus(
        test,
        p_primary_by_matchup=p_primary,
        thresholds_by_bucket=thresholds,
        confirmers=["market_proxy"],
        consensus_min_models=2,  # require at least one confirmer
        sizing=(1, 1, 1),
        confirmer_positive_bucket_only=False,
        confirmer_allowed={},
        trade_rows_out=None,
    )
    assert bets == 1
    assert units == 1
    assert abs(pnl - 0.60) < 1e-12

    # If we don't require a confirmer, both trades are taken (1 unit each).
    bets2, units2, pnl2 = _evaluate_consensus(
        test,
        p_primary_by_matchup=p_primary,
        thresholds_by_bucket=thresholds,
        confirmers=["market_proxy"],
        consensus_min_models=1,
        sizing=(1, 1, 1),
        confirmer_positive_bucket_only=False,
        confirmer_allowed={},
        trade_rows_out=None,
    )
    assert bets2 == 2
    assert units2 == 2
    assert abs(pnl2 - 1.20) < 1e-12

    # If the positive-bucket filter disallows the confirmer in bucket A, the trade is skipped.
    bets3, units3, pnl3 = _evaluate_consensus(
        test,
        p_primary_by_matchup=p_primary,
        thresholds_by_bucket=thresholds,
        confirmers=["market_proxy"],
        consensus_min_models=2,
        sizing=(1, 1, 1),
        confirmer_positive_bucket_only=True,
        confirmer_allowed={("market_proxy", "A"): False},
        trade_rows_out=None,
    )
    assert bets3 == 0
    assert units3 == 0
    assert abs(pnl3 - 0.0) < 1e-12

