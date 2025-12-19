from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_grade_rule_a_votes_plan_backfills_from_fills_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chimera_v2c.tools import grade_rule_a_votes_plan as grader

    daily_dir = tmp_path / "reports" / "daily_ledgers"
    daily_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = daily_dir / "20251217_daily_game_ledger.csv"
    _write_csv(
        ledger_path,
        [
            {
                "date": "2025-12-17",
                "league": "nhl",
                "matchup": "ANA@NJD",
                "kalshi_mid": "0.60",
                "actual_outcome": "ANA 1-4 NJD",
            }
        ],
    )

    plan_path = tmp_path / "plan.csv"
    _write_csv(
        plan_path,
        [
            {
                "date": "2025-12-17",
                "league": "nhl",
                "matchup": "ANA@NJD",
                "market_ticker_away": "KXNHLGAME-25DEC17ANANJD-ANA",
                "contracts_planned": "3",
                "price_away_planned": "0.40",
                "contracts_filled": "",
                "price_away_filled": "",
                "fees_filled": "",
            }
        ],
    )

    fills_path = tmp_path / "fills.csv"
    _write_csv(
        fills_path,
        [
            {
                "created_time_utc": "2025-12-17T23:59:59Z",
                "date_utc": "2025-12-17",
                "date_local": "2025-12-17",
                "ticker": "KXNHLGAME-25DEC17ANANJD-ANA",
                "action": "buy",
                "side": "yes",
                "count": "3",
                "price_cents": "41",
            }
        ],
    )

    monkeypatch.setattr(grader, "LEDGER_DIR", daily_dir)

    out_path = tmp_path / "out.csv"
    argv = [
        "grade_rule_a_votes_plan.py",
        "--plan-csv",
        str(plan_path),
        "--fills-csv",
        str(fills_path),
        "--out",
        str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    grader.main()

    out_rows = list(csv.DictReader(out_path.open("r", encoding="utf-8", newline="")))
    assert len(out_rows) == 1
    r = out_rows[0]
    assert r["contracts_used"] == "3"
    assert float(r["price_used"]) == pytest.approx(0.41, abs=1e-6)

    # Home won, so buying away YES loses price per contract (gross = -c*price).
    gross_expected = -3 * 0.41
    fee_expected = grader.taker_fee_dollars(contracts=3, price=0.41)
    net_expected = gross_expected - fee_expected
    assert float(r["gross_pnl"]) == pytest.approx(gross_expected, abs=1e-6)
    assert float(r["net_pnl"]) == pytest.approx(net_expected, abs=1e-6)


def test_grade_rule_a_votes_plan_counts_sell_no_as_buy_yes_equivalent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chimera_v2c.tools import grade_rule_a_votes_plan as grader

    daily_dir = tmp_path / "reports" / "daily_ledgers"
    daily_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = daily_dir / "20251217_daily_game_ledger.csv"
    _write_csv(
        ledger_path,
        [
            {
                "date": "2025-12-17",
                "league": "nba",
                "matchup": "MEM@MIN",
                "kalshi_mid": "0.60",
                "actual_outcome": "MEM 100-90 MIN",  # away wins, so YES on MEM wins
            }
        ],
    )

    plan_path = tmp_path / "plan.csv"
    _write_csv(
        plan_path,
        [
            {
                "date": "2025-12-17",
                "league": "nba",
                "matchup": "MEM@MIN",
                "market_ticker_away": "KXNBAGAME-25DEC17MEMMIN-MEM",
                "contracts_planned": "0",
                "price_away_planned": "0.40",
                "contracts_filled": "",
                "price_away_filled": "",
                "fees_filled": "",
            }
        ],
    )

    fills_path = tmp_path / "fills.csv"
    _write_csv(
        fills_path,
        [
            # buy YES 4 @ 0.26
            {
                "created_time_utc": "2025-12-17T21:00:00Z",
                "date_utc": "2025-12-17",
                "date_local": "2025-12-17",
                "ticker": "KXNBAGAME-25DEC17MEMMIN-MEM",
                "action": "buy",
                "side": "yes",
                "count": "4",
                "price_cents": "26",
            },
            # sell NO 2 @ 0.74 (YES-equivalent buy 2 @ 0.26)
            {
                "created_time_utc": "2025-12-17T22:00:00Z",
                "date_utc": "2025-12-17",
                "date_local": "2025-12-17",
                "ticker": "KXNBAGAME-25DEC17MEMMIN-MEM",
                "action": "sell",
                "side": "no",
                "count": "2",
                "price_cents": "74",
            },
        ],
    )

    monkeypatch.setattr(grader, "LEDGER_DIR", daily_dir)

    out_path = tmp_path / "out.csv"
    argv = [
        "grade_rule_a_votes_plan.py",
        "--plan-csv",
        str(plan_path),
        "--fills-csv",
        str(fills_path),
        "--out",
        str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    grader.main()

    out_rows = list(csv.DictReader(out_path.open("r", encoding="utf-8", newline="")))
    assert len(out_rows) == 1
    r = out_rows[0]
    assert r["contracts_used"] == "6"
    assert float(r["price_used"]) == pytest.approx(0.26, abs=1e-6)

