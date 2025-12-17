from __future__ import annotations

from chimera_v2c.src.kalshi_fees import maker_fee_dollars, round_up_to_cent, taker_fee_dollars


def test_round_up_to_cent() -> None:
    assert round_up_to_cent(0.0) == 0.0
    assert round_up_to_cent(0.0100) == 0.01
    assert round_up_to_cent(0.010000000000000002) == 0.01
    assert round_up_to_cent(0.0101) == 0.02


def test_maker_fee_matches_formula_examples() -> None:
    # For 1 contract, maker fee is always at least 1 cent (round up).
    assert maker_fee_dollars(contracts=1, price=0.50) == 0.01
    assert maker_fee_dollars(contracts=1, price=0.10) == 0.01

    # For 3 contracts at 0.50:
    # raw = 0.0175 * 3 * 0.5 * 0.5 = 0.013125 -> round up to 0.02
    assert maker_fee_dollars(contracts=3, price=0.50) == 0.02


def test_taker_fee_is_higher_than_maker() -> None:
    # raw taker = 0.07 * 3 * 0.25 = 0.0525 -> 0.06
    assert taker_fee_dollars(contracts=3, price=0.50) == 0.06

