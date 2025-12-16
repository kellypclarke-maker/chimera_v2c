from __future__ import annotations

import pytest

from chimera_v2c.src.rulebook_policy import select_trade_decision


def test_rulebook_policy_select_trade_decision_home_fav_A() -> None:
    d = select_trade_decision(p_mid=0.60, p_model=0.54, thresholds_by_bucket={"A": 0.05})
    assert d is not None
    assert d.bucket == "A"
    assert d.side == "away"


def test_rulebook_policy_select_trade_decision_home_fav_B() -> None:
    d = select_trade_decision(p_mid=0.60, p_model=0.66, thresholds_by_bucket={"B": 0.05})
    assert d is not None
    assert d.bucket == "B"
    assert d.side == "home"


def test_rulebook_policy_select_trade_decision_away_fav_C() -> None:
    d = select_trade_decision(p_mid=0.40, p_model=0.35, thresholds_by_bucket={"C": 0.05})
    assert d is not None
    assert d.bucket == "C"
    assert d.side == "away"


def test_rulebook_policy_select_trade_decision_away_fav_D() -> None:
    d = select_trade_decision(p_mid=0.40, p_model=0.46, thresholds_by_bucket={"D": 0.05})
    assert d is not None
    assert d.bucket == "D"
    assert d.side == "home"


def test_rulebook_policy_select_trade_decision_none() -> None:
    assert select_trade_decision(p_mid=0.60, p_model=0.58, thresholds_by_bucket={"A": 0.05, "B": 0.05}) is None
    assert select_trade_decision(p_mid=0.60, p_model=0.80, thresholds_by_bucket={"A": 0.05}) is None
    assert select_trade_decision(p_mid=0.40, p_model=0.10, thresholds_by_bucket={"D": 0.05}) is None

