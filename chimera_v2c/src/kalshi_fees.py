from __future__ import annotations

import math
from dataclasses import dataclass


# Kalshi fee schedule (Oct 1, 2025 update).
# Source: https://kalshi.com/docs/kalshi-fee-schedule.pdf
#
# General trading fees (taker):
#   fees = round_up(0.07 * C * P * (1 - P))
#
# Maker fees (where applicable; e.g., series fee_type=quadratic_with_maker_fees):
#   fees = round_up(0.0175 * C * P * (1 - P))
#
# Where:
#   P = contract price in dollars (0.50 = 50 cents)
#   C = number of contracts executed
#   round_up = round up to the next cent
#
# Notes:
# - This module implements the fee formulas above; it does not determine whether a
#   given series applies maker fees (that is series-specific).


TAKER_FEE_RATE = 0.07
MAKER_FEE_RATE = 0.0175


def _clamp_price(p: float) -> float:
    # Kalshi prices are in [0,1] but our ledgers may include rounded values;
    # avoid 0/1 edge cases.
    return max(0.0, min(1.0, float(p)))


def round_up_to_cent(dollars: float) -> float:
    """
    Round up to the next cent (two decimals).

    Example: 0.0001 -> 0.01, 0.0100 -> 0.01, 0.0101 -> 0.02.
    """
    x = float(dollars)
    # Protect against float drift when value is already at a cent boundary.
    return math.ceil(x * 100.0 - 1e-12) / 100.0


def fee_dollars(*, rate: float, contracts: int, price: float) -> float:
    """
    Compute the fee in dollars for an executed trade.

    Parameters:
      - rate: fee rate (e.g., 0.07 taker, 0.0175 maker)
      - contracts: number of contracts executed (C)
      - price: contract price in dollars (P)
    """
    c = int(contracts)
    if c <= 0:
        return 0.0
    p = _clamp_price(price)
    raw = float(rate) * float(c) * p * (1.0 - p)
    return float(round_up_to_cent(raw))


def maker_fee_dollars(*, contracts: int, price: float) -> float:
    return fee_dollars(rate=MAKER_FEE_RATE, contracts=contracts, price=price)


def taker_fee_dollars(*, contracts: int, price: float) -> float:
    return fee_dollars(rate=TAKER_FEE_RATE, contracts=contracts, price=price)


@dataclass(frozen=True)
class FeeBreakdown:
    gross_pnl: float
    fees: float

    @property
    def net_pnl(self) -> float:
        return float(self.gross_pnl - self.fees)

