from __future__ import annotations

from pathlib import Path

from chimera_v2c.src.ledger.formatting import DAILY_LEDGER_COLUMNS, MISSING_SENTINEL
from chimera_v2c.src.ledger.guard import load_csv_records, write_csv
from chimera_v2c.tools.seed_daily_ledger_from_espn_schedule import rows_from_scoreboard, seed_daily_ledger_from_rows


def test_seed_daily_ledger_appends_missing_rows(tmp_path: Path) -> None:
    ledger_path = tmp_path / "20251216_daily_game_ledger.csv"
    write_csv(ledger_path, [], DAILY_LEDGER_COLUMNS)

    scoreboard = {
        "events": [
            {
                "id": "1",
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "BOS"}},
                            {"homeAway": "away", "team": {"abbreviation": "NYK"}},
                        ]
                    }
                ],
            }
        ]
    }

    rows = rows_from_scoreboard(league="nba", date_iso="2025-12-16", scoreboard=scoreboard)
    added, prior = seed_daily_ledger_from_rows(ledger_path=ledger_path, rows_to_add=rows, apply=True, force=False)
    assert prior == 0
    assert added == 1

    records = load_csv_records(ledger_path)
    assert len(records) == 1
    row = records[0]
    assert row["date"] == "2025-12-16"
    assert row["league"] == "nba"
    assert row["matchup"] == "NYK@BOS"
    assert row["v2c"] == MISSING_SENTINEL
    assert row["kalshi_mid"] == MISSING_SENTINEL


def test_seed_daily_ledger_is_idempotent(tmp_path: Path) -> None:
    ledger_path = tmp_path / "20251216_daily_game_ledger.csv"
    write_csv(
        ledger_path,
        [
            {
                "date": "2025-12-16",
                "league": "nba",
                "matchup": "NYK@BOS",
                "v2c": MISSING_SENTINEL,
                "grok": MISSING_SENTINEL,
                "gemini": MISSING_SENTINEL,
                "gpt": MISSING_SENTINEL,
                "kalshi_mid": MISSING_SENTINEL,
                "market_proxy": MISSING_SENTINEL,
                "moneypuck": MISSING_SENTINEL,
                "actual_outcome": "",
            }
        ],
        DAILY_LEDGER_COLUMNS,
    )

    scoreboard = {
        "events": [
            {
                "id": "1",
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "BOS"}},
                            {"homeAway": "away", "team": {"abbreviation": "NYK"}},
                        ]
                    }
                ],
            }
        ]
    }
    rows = rows_from_scoreboard(league="nba", date_iso="2025-12-16", scoreboard=scoreboard)
    added, prior = seed_daily_ledger_from_rows(ledger_path=ledger_path, rows_to_add=rows, apply=True, force=False)
    assert prior == 1
    assert added == 0

