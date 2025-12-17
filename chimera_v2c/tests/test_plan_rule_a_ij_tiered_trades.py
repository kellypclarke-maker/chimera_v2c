from __future__ import annotations

from datetime import datetime

from chimera_v2c.src.ledger_analysis import GameRow
from chimera_v2c.tools.plan_rule_a_ij_tiered_trades import plan_trades


def _row(
    *,
    date_iso: str,
    league: str,
    matchup: str,
    p_mid: float,
    p_primary: float,
    p_confirmer: float,
    home_win: float | None,
) -> GameRow:
    return GameRow(
        date=datetime.strptime(date_iso, "%Y-%m-%d"),
        league=league,
        matchup=matchup,
        kalshi_mid=p_mid,
        probs={"grok": p_primary, "market_proxy": p_confirmer},
        home_win=home_win,
    )


def test_plan_trades_emits_i_trade_without_confirmer_requirement() -> None:
    games = [
        _row(
            date_iso="2025-12-01",
            league="nba",
            matchup="AAA@BBB",
            p_mid=0.80,
            p_primary=0.40,  # I
            p_confirmer=0.60,  # not J at t=0.02
            home_win=0.0,
        ),
        _row(
            date_iso="2025-12-02",
            league="nba",
            matchup="CCC@DDD",
            p_mid=0.80,
            p_primary=0.40,  # I
            p_confirmer=0.60,
            home_win=0.0,
        ),
        _row(
            date_iso="2025-12-03",
            league="nba",
            matchup="EEE@FFF",
            p_mid=0.80,
            p_primary=0.40,  # I => should trade regardless of confirmer
            p_confirmer=0.60,
            home_win=None,
        ),
    ]

    rows, meta = plan_trades(
        games,
        target_date="2025-12-03",
        league_for_stats="nba",
        primary="grok",
        confirmer="market_proxy",
        edge_threshold=0.02,
        edge_thresholds=None,
        threshold_select_mode="max_net_pnl",
        units_i=3,
        units_jj=1,
        fee_mode="none",
        train_days=0,
        min_train_days=2,
    )

    assert meta["trained"] is True
    assert meta["t_selected"] == 0.02
    assert len(rows) == 1
    assert rows[0]["matchup"] == "EEE@FFF"
    assert rows[0]["policy_subbucket_primary"] == "I"
    assert rows[0]["contracts"] == 3
    assert rows[0]["fee_estimate"] == 0.0


def test_plan_trades_requires_confirmer_for_j_trades() -> None:
    games = [
        _row(
            date_iso="2025-12-01",
            league="nba",
            matchup="AAA@BBB",
            p_mid=0.60,
            p_primary=0.55,  # J at t=0.02 (0.55 <= 0.58)
            p_confirmer=0.59,  # not J at t=0.02 (0.59 > 0.58)
            home_win=1.0,
        ),
        _row(
            date_iso="2025-12-02",
            league="nba",
            matchup="CCC@DDD",
            p_mid=0.60,
            p_primary=0.55,
            p_confirmer=0.59,
            home_win=1.0,
        ),
        _row(
            date_iso="2025-12-03",
            league="nba",
            matchup="EEE@FFF",
            p_mid=0.60,
            p_primary=0.55,  # primary J
            p_confirmer=0.59,  # confirmer fails J => should skip
            home_win=None,
        ),
    ]

    rows, meta = plan_trades(
        games,
        target_date="2025-12-03",
        league_for_stats="nba",
        primary="grok",
        confirmer="market_proxy",
        edge_threshold=0.02,
        edge_thresholds=None,
        threshold_select_mode="max_net_pnl",
        units_i=3,
        units_jj=1,
        fee_mode="none",
        train_days=0,
        min_train_days=2,
    )

    assert meta["trained"] is True
    assert rows == []

