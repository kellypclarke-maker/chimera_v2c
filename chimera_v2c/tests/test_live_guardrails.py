from pathlib import Path

import chimera_v2c.tools.live_guardrails as lg


def test_parse_ticker_teams_basic():
    res = lg.parse_ticker_teams("KXNBAGAME-25DEC04BOSLAL-BOS", "nba")
    assert res is not None
    home, away, pick = res
    assert pick == "BOS"
    assert {home, away} == {"BOS", "LAL"}


def test_trigger_guardrail_trips_on_margin():
    sb = {
        "games": [
            {
                "league": "nba",
                "teams": {"home": {"alias": "BOS"}, "away": {"alias": "NYK"}},
                "scores": {"home": 80, "away": 95},
            }
        ]
    }
    assert lg.trigger_guardrail("BOS", "NYK", sb, margin_thresh=10) is True
    assert lg.trigger_guardrail("NYK", "BOS", sb, margin_thresh=20) is False


def test_live_prob_from_score_handles_diff():
    sb = {
        "games": [
            {
                "league": "nba",
                "teams": {"home": {"alias": "BOS"}, "away": {"alias": "NYK"}},
                "scores": {"home": 100, "away": 90},
            }
        ]
    }
    p = lg.live_prob_from_score("BOS", "NYK", sb, scale=10.0)
    assert p is not None
    assert p > 0.5
    p2 = lg.live_prob_from_score("NYK", "BOS", sb, scale=10.0)
    assert p2 is not None
    assert p2 < 0.5
