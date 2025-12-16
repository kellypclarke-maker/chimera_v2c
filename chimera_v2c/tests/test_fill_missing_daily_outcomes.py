from __future__ import annotations

from chimera_v2c.tools.fill_missing_daily_outcomes import match_game


def test_match_game_requires_final_state() -> None:
    sb = {
        "status": "ok",
        "games": [
            {
                "teams": {"home": {"alias": "NJD"}, "away": {"alias": "ANA"}},
                "scores": {"home": "0", "away": "0"},
                "status": {"state": "in", "detail": "2nd"},
            },
            {
                "teams": {"home": {"alias": "NJD"}, "away": {"alias": "ANA"}},
                "scores": {"home": "4", "away": "1"},
                "status": {"state": "post", "detail": "Final"},
            },
        ],
    }

    assert match_game(sb, "nhl", "ANA", "NJD") == (1.0, 4.0)
