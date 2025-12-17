from __future__ import annotations

from chimera_v2c.tools.build_schedule_csv_from_espn import scoreboard_to_schedule_rows


def test_scoreboard_to_schedule_rows_final_game() -> None:
    sb = {
        "status": "ok",
        "games": [
            {
                "teams": {"away": {"alias": "NYK"}, "home": {"alias": "BOS"}},
                "status": {"state": "post"},
                "scores": {"away": "101", "home": "99"},
            }
        ],
    }
    rows = scoreboard_to_schedule_rows(league="nba", date_iso="2025-12-15", sb=sb)
    assert len(rows) == 1
    row = rows[0]
    assert row["date"] == "2025-12-15"
    assert row["away"] == "NYK"
    assert row["home"] == "BOS"
    assert row["away_score"] == 101.0
    assert row["home_score"] == 99.0
    assert row["outcome"] == "NYK 101-99 BOS"


def test_scoreboard_to_schedule_rows_pregame_keeps_key_with_zero_scores() -> None:
    sb = {
        "status": "ok",
        "games": [
            {
                "teams": {"away": {"alias": "SAS"}, "home": {"alias": "NYK"}},
                "status": {"state": "pre"},
                "scores": {"away": "0", "home": "0"},
            }
        ],
    }
    rows = scoreboard_to_schedule_rows(league="nba", date_iso="2025-12-16", sb=sb)
    assert len(rows) == 1
    row = rows[0]
    assert row["away"] == "SAS"
    assert row["home"] == "NYK"
    assert row["away_score"] == 0.0
    assert row["home_score"] == 0.0
    assert row["outcome"] == ""

