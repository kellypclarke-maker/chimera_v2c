from __future__ import annotations

from chimera_v2c.tools.walkforward_rule_a_ij_tiered_policy import Sample, _pick_threshold, _train_score_for_t


def test_pick_threshold_prefers_net_pnl_on_train() -> None:
    # Build a tiny train set where t=0.01 triggers trades and t=0.03 triggers none.
    train = [
        # Home-fav, model is rich by 0.02 -> triggers at 0.01 but not 0.03.
        Sample(date="2025-12-01", league="nba", matchup="A@B", p_mid=0.60, y=0, probs={"grok": 0.58, "market_proxy": 0.58}),
    ]
    t = _pick_threshold(
        train,
        primary="grok",
        confirmer="market_proxy",
        candidates=[0.01, 0.03],
        units_i=2,
        units_jj=1,
        fee_mode="none",
        mode="max_net_pnl",
    )
    assert t == 0.01


def test_train_score_counts_contracts_and_fees() -> None:
    train = [
        Sample(date="2025-12-01", league="nba", matchup="A@B", p_mid=0.60, y=0, probs={"grok": 0.40, "market_proxy": 0.58}),
    ]
    # grok is I (p<0.5) so contracts=2, away win => +0.60 per contract.
    stats = _train_score_for_t(
        train,
        primary="grok",
        confirmer="market_proxy",
        t=0.02,
        units_i=2,
        units_jj=1,
        fee_mode="maker",
    )
    assert stats.bets == 1
    assert stats.contracts == 2
    assert abs(stats.gross_pnl - 1.2) < 1e-12
    # Maker fee should be > 0 for a non-zero price.
    assert stats.fees > 0.0

