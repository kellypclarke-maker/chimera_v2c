from __future__ import annotations

from pathlib import Path

import pandas as pd

from chimera_v2c.tools.fill_daily_ledger_from_plan import apply_updates


def test_fill_daily_ledger_from_plan_fills_nr_cells(tmp_path: Path) -> None:
    ledger_path = tmp_path / "20990101_daily_game_ledger.csv"
    ledger_path.write_text(
        "date,league,matchup,v2c,grok,gemini,gpt,kalshi_mid,market_proxy,moneypuck,actual_outcome\n"
        "2099-01-01,nba,AAA@BBB,NR,NR,NR,NR,.52,NR,NR,\n",
        encoding="utf-8",
    )

    updates = [
        {
            "date": "2099-01-01",
            "league": "nba",
            "matchup": "AAA@BBB",
            "v2c": ".66",
            "kalshi_mid": ".51",
        }
    ]

    added, filled = apply_updates(
        ledger_path=ledger_path,
        league="nba",
        updates=updates,
        apply=True,
        force=False,
    )
    assert added == 0
    assert filled == 1

    df = pd.read_csv(ledger_path, dtype=str, keep_default_na=False).fillna("")
    assert df.loc[0, "v2c"] == ".66"
    assert df.loc[0, "kalshi_mid"] == ".52"


def test_fill_daily_ledger_from_plan_new_rows_get_nr_defaults(tmp_path: Path) -> None:
    ledger_path = tmp_path / "20990102_daily_game_ledger.csv"
    ledger_path.write_text(
        "date,league,matchup,v2c,grok,gemini,gpt,kalshi_mid,market_proxy,moneypuck,actual_outcome\n"
        "2099-01-02,nba,AAA@BBB,.50,NR,NR,NR,NR,NR,NR,\n",
        encoding="utf-8",
    )

    updates = [
        {
            "date": "2099-01-02",
            "league": "nba",
            "matchup": "CCC@DDD",
            "v2c": ".61",
            "kalshi_mid": ".60",
        }
    ]

    added, filled = apply_updates(
        ledger_path=ledger_path,
        league="nba",
        updates=updates,
        apply=True,
        force=False,
    )
    assert added == 1
    assert filled == 2

    df = pd.read_csv(ledger_path, dtype=str, keep_default_na=False).fillna("")
    new_row = df[df["matchup"] == "CCC@DDD"].iloc[0].to_dict()
    assert new_row["v2c"] == ".61"
    assert new_row["kalshi_mid"] == ".60"
    assert new_row["grok"] == "NR"
    assert new_row["gemini"] == "NR"
    assert new_row["gpt"] == "NR"
    assert new_row["market_proxy"] == "NR"
    assert new_row["moneypuck"] == "NR"
    assert new_row["actual_outcome"] == ""
