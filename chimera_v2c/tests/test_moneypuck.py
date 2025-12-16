from __future__ import annotations

from datetime import date

import pytest

from chimera_v2c.lib.moneypuck import games_for_date, parse_pregame_csv, season_string_for_date


def test_moneypuck_season_string_for_date_spans_two_years() -> None:
    assert season_string_for_date(date(2025, 10, 1)) == "20252026"
    assert season_string_for_date(date(2026, 3, 1)) == "20252026"
    assert season_string_for_date(date(2025, 1, 1)) == "20242025"


def test_moneypuck_games_for_date_filters_by_est_prefix() -> None:
    schedule = [
        {"id": 2025020495, "a": "ANA", "h": "NJD", "est": "20251213 1900"},
        {"id": 2025020496, "a": "OTT", "h": "MIN", "est": "20251213 2000"},
        {"id": 2025020497, "a": "BOS", "h": "BUF", "est": "20251214 1900"},
    ]

    games = games_for_date(schedule, "2025-12-13")
    matchups = sorted(g.matchup for g in games)
    assert matchups == ["ANA@NJD", "OTT@MIN"]


def test_moneypuck_parse_pregame_csv_extracts_fields() -> None:
    csv_text = (
        "gameID,homeTeamCode,roadTeamCode,preGameMoneyPuckHomeWinPrediction,preGameBettingOddsHomeWinPrediction,startingGoalie\n"
        "2025020495,NJD,ANA,0.621,0.605,1\n"
    )

    pre = parse_pregame_csv(csv_text)
    assert pre.game_id == 2025020495
    assert pre.home == "NJD"
    assert pre.away == "ANA"
    assert pre.moneypuck_home_win == pytest.approx(0.621)
    assert pre.betting_odds_home_win == pytest.approx(0.605)
    assert pre.starting_goalie == 1

