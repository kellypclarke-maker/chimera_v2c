from __future__ import annotations

import datetime as dt

from chimera_v2c.tools.backfill_kalshi_mid_from_candlesticks import (
    _candlestick_mid_prob,
    _kalshi_date_token,
    _parse_utc_iso,
    _team_tokens,
)


def test_kalshi_date_token() -> None:
    assert _kalshi_date_token(dt.date(2025, 11, 19)) == "25NOV19"


def test_team_tokens_includes_known_short_aliases() -> None:
    tokens = _team_tokens("TBL", "nhl")
    assert tokens[0] == "TBL"
    assert "TB" in tokens

    tokens = _team_tokens("NJD", "nhl")
    assert tokens[0] == "NJD"
    assert "NJ" in tokens


def test_parse_utc_iso_z_suffix() -> None:
    parsed = _parse_utc_iso("2025-11-20T00:30Z")
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == dt.timedelta(0)
    assert parsed.hour == 0 and parsed.minute == 30


def test_candlestick_mid_prob_uses_yes_bid_ask_close() -> None:
    candle = {"yes_bid": {"close": 78}, "yes_ask": {"close": 79}}
    mid = _candlestick_mid_prob(candle)
    assert mid is not None
    assert abs(mid - 0.785) < 1e-9

