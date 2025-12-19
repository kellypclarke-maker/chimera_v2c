from __future__ import annotations

import pandas as pd

from chimera_v2c.tools.place_rule_a_maker_orders_from_sheet import _parse_sheet_orders


def test_parse_sheet_orders_filters_and_normalizes(tmp_path) -> None:
    df = pd.DataFrame(
        [
            {"market_ticker_away": "kxnblgame-abc", "contracts": 2, "maker_limit_price_cents": 10},
            {"market_ticker_away": "KXNHLGAME-def", "contracts": 0, "maker_limit_price_cents": 41},
            {"market_ticker_away": "KXNHLGAME-ghi", "contracts": 1, "maker_limit_price_cents": None},
        ]
    )
    p = tmp_path / "sheet.csv"
    df.to_csv(p, index=False)

    orders = _parse_sheet_orders(p)
    assert len(orders) == 1
    assert orders[0].ticker == "KXNBLGAME-ABC"
    assert orders[0].count == 2
    assert orders[0].price_cents == 10

