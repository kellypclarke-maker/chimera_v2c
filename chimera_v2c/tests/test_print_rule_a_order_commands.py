from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from chimera_v2c.tools import print_rule_a_order_commands


def test_print_rule_a_order_commands_emits_expected_line(tmp_path: Path, capsys) -> None:
    sheet = pd.DataFrame(
        [
            {
                "market_ticker_away": "KXNBAGAME-FAKE-AWAY",
                "contracts": 3,
                "maker_limit_price_cents": 12,
            }
        ]
    )
    sheet_path = tmp_path / "sheet.csv"
    sheet.to_csv(sheet_path, index=False)

    argv = [
        "print_rule_a_order_commands.py",
        "--sheet",
        str(sheet_path),
        "--require-maker",
    ]
    old = sys.argv
    try:
        sys.argv = argv
        print_rule_a_order_commands.main()
    finally:
        sys.argv = old

    out = capsys.readouterr().out
    assert "kalshi_place_limit_order.py" in out
    assert "--ticker KXNBAGAME-FAKE-AWAY" in out
    assert "--count 3" in out
    assert "--price-cents 12" in out
    assert "--require-maker" in out
    cmd_lines = [line for line in out.splitlines() if line.startswith("PYTHONPATH=")]
    assert len(cmd_lines) == 1
    assert "--confirm" not in cmd_lines[0]


def test_print_rule_a_order_commands_live_includes_confirm(tmp_path: Path, capsys) -> None:
    sheet = pd.DataFrame(
        [
            {
                "market_ticker_away": "KXNHLGAME-FAKE-AWAY",
                "contracts": 1,
                "maker_limit_price_cents": 41,
            }
        ]
    )
    sheet_path = tmp_path / "sheet.csv"
    sheet.to_csv(sheet_path, index=False)

    argv = [
        "print_rule_a_order_commands.py",
        "--sheet",
        str(sheet_path),
        "--live",
    ]
    old = sys.argv
    try:
        sys.argv = argv
        print_rule_a_order_commands.main()
    finally:
        sys.argv = old

    out = capsys.readouterr().out
    assert "--confirm" in out
