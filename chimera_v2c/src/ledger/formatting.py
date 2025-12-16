from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Final, Optional


MISSING_SENTINEL: Final[str] = "NR"

# Canonical daily-ledger schema for operator readability (one game per row).
DAILY_LEDGER_COLUMNS: Final[list[str]] = [
    "date",
    "league",
    "matchup",
    "v2c",
    "grok",
    "gemini",
    "gpt",
    "kalshi_mid",
    "market_proxy",
    "moneypuck",
    "actual_outcome",
]

DAILY_LEDGER_PROB_COLUMNS: Final[set[str]] = {
    "v2c",
    "grok",
    "gemini",
    "gpt",
    "kalshi_mid",
    "market_proxy",
    "moneypuck",
}


def _to_decimal(text: str) -> Optional[Decimal]:
    s = text.strip()
    if not s:
        return None
    try:
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


def format_prob_cell(
    value: object,
    *,
    decimals: int = 2,
    drop_leading_zero: bool = True,
) -> str:
    """
    Format a probability-like cell to a canonical string representation.

    - Preserves the missing sentinel "NR" (case-insensitive).
    - Returns "" for blanks/unparseable/out-of-range values.
    - Rounds with half-up semantics.
    - Optionally drops the leading zero for values in [0, 1): 0.85 -> .85
    """
    s = str(value).strip() if value is not None else ""
    if not s:
        return ""

    if s.strip().upper() == MISSING_SENTINEL:
        return MISSING_SENTINEL

    d = _to_decimal(s)
    if d is None:
        return ""

    if d < Decimal("0") or d > Decimal("1"):
        return ""

    q = Decimal("1").scaleb(-decimals)  # 10^-decimals
    rounded = d.quantize(q, rounding=ROUND_HALF_UP)

    out = f"{rounded:.{decimals}f}"
    if drop_leading_zero and out.startswith("0.") and rounded < Decimal("1"):
        return out[1:]
    return out

