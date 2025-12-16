from __future__ import annotations

from chimera_v2c.tools.walkforward_grok_mid_hybrid_backtest import (
    Sample,
    _cv_alpha_pnl,
    _select_best_alpha_from_scores,
)


def test_cv_alpha_pnl_prefers_smallest_alpha_on_tie() -> None:
    # Construct two dated samples where a trade only triggers once alpha is large enough.
    # With threshold t=0.1 and p_mid=0.6, we need p_hybrid <= 0.5 to trigger bucket A.
    # p_hybrid = 0.6 + alpha*(0.4 - 0.6) = 0.6 - 0.2*alpha.
    train = [
        Sample(date="2025-12-01", league="nba", matchup="A@B", p_grok=0.40, p_mid=0.60, y=0),
        Sample(date="2025-12-02", league="nba", matchup="C@D", p_grok=0.40, p_mid=0.60, y=0),
    ]
    alphas = [0.4, 0.5, 1.0]
    score_by_alpha = _cv_alpha_pnl(
        train,
        alphas=alphas,
        min_train_samples=999,  # force identity Platt
        thresholds=[0.1],
        league_for_stats="all",
        min_bets=1,
        ev_threshold=-1.0,
        select_mode="min_edge",
        metric="avg_pnl_per_bet",
    )

    # alpha=0.4 never triggers the trade => score -inf via min_bets gate.
    assert score_by_alpha[0.4] < -1e9
    # alpha=0.5 and alpha=1.0 both trigger and have the same avg PnL per bet.
    assert abs(score_by_alpha[0.5] - score_by_alpha[1.0]) < 1e-12

    # Tie-break is conservative: pick the smaller alpha.
    best = _select_best_alpha_from_scores(score_by_alpha, larger_is_better=True)
    assert best == 0.5

