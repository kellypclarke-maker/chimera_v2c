from __future__ import annotations

import pandas as pd

from chimera_v2c.tools.build_rule_a_maker_order_sheet import _maker_limit_price_cents, build_maker_order_sheet


def test_maker_limit_price_cents_clamps_to_ask_minus_one() -> None:
    d = _maker_limit_price_cents(bid_cents=8, ask_cents=10, improve_cents=1)
    assert d.limit_price_cents == 9
    assert d.reason == "maker_inside_spread"

    d2 = _maker_limit_price_cents(bid_cents=8, ask_cents=9, improve_cents=1)
    assert d2.limit_price_cents == 8
    assert d2.reason == "maker_at_bid"


def test_build_maker_order_sheet_keeps_unpriceable_rows(tmp_path) -> None:
    plan = pd.DataFrame(
        [
            {
                "date": "2025-12-18",
                "league": "nhl",
                "matchup": "AAA@BBB",
                "start_time_utc": "2025-12-19T00:00:00Z",
                "market_ticker_away": "KXNHLGAME-FAKE-AWAY",
                "contracts_planned": 2,
                "away_yes_bid_cents": 10,
                "away_yes_ask_cents": 11,
            },
            {
                "date": "2025-12-18",
                "league": "nhl",
                "matchup": "CCC@DDD",
                "start_time_utc": "2025-12-19T00:00:00Z",
                "market_ticker_away": "KXNHLGAME-FAKE-AWAY2",
                "contracts_planned": 1,
                "away_yes_bid_cents": 10,
                "away_yes_ask_cents": None,
            },
        ]
    )
    plan_path = tmp_path / "plan.csv"
    plan.to_csv(plan_path, index=False)

    sheet = build_maker_order_sheet(plan_paths=[plan_path], maker_improve_cents=0)
    assert len(sheet) == 2
    assert sheet.loc[0, "maker_limit_price_cents"] == 10
    assert sheet.loc[1, "maker_limit_price_cents"] != sheet.loc[1, "maker_limit_price_cents"]  # NaN

