from __future__ import annotations

from pathlib import Path

from chimera_v2c.tools.auto_rule_a_tminus_llm import render_legacy_helios_block
from chimera_v2c.tools.auto_rule_a_tminus_llm import _coerce_json_object
from chimera_v2c.tools.ingest_raw_specialist_reports_v2c import (
    parse_compact_prediction_headers,
    parse_legacy_helios_prediction_headers,
)


def test_auto_rule_a_tminus_llm_renders_parseable_helios_blocks(tmp_path: Path) -> None:
    text = (
        render_legacy_helios_block(
            ts_pacific="2025-12-18 12:00 PST",
            league="nhl",
            matchup="UTA@DET",
            model_label="gpt-4o-mini",
            winner="DET",
            p_home=0.5234,
            confidence=0.77,
        )
        + render_legacy_helios_block(
            ts_pacific="2025-12-18 12:00 PST",
            league="nhl",
            matchup="LAK@SJS",
            model_label="gpt-4o-mini",
            winner="SJS",
            p_home=0.6123,
            confidence=None,
        )
    )

    raw = tmp_path / "2025-12-18 gpt nhl.txt"
    raw.write_text(text, encoding="utf-8")

    games = parse_legacy_helios_prediction_headers(text, raw)
    assert len(games) == 2
    assert games[0].date == "2025-12-18"
    assert games[0].league == "nhl"
    assert games[0].matchup == "UTA@DET"
    assert games[0].home == "DET"
    assert games[0].away == "UTA"
    assert games[0].p_home is not None


def test_auto_rule_a_tminus_llm_coerces_list_wrapped_json() -> None:
    obj = {"matchup": "LAC@OKC", "p_home": 0.55, "confidence": 0.7, "winner": "OKC"}
    assert _coerce_json_object(obj) == obj
    assert _coerce_json_object([obj]) == obj
    assert _coerce_json_object([1, "x", obj]) == obj
    assert _coerce_json_object([]) is None


def test_parse_legacy_helios_accepts_parenthesized_bold_p_home(tmp_path: Path) -> None:
    text = (
        "HELIOS_PREDICTION_HEADER\n"
        "LEAGUE: NHL\n"
        "DATE: 2025-12-17\n"
        "MATCHUP: UTA@DET\n"
        "P_HOME (DET): **0.540**\n"
        "\n---\n"
    )
    raw = tmp_path / "12-17 gpt NHL.txt"
    raw.write_text(text, encoding="utf-8")
    games = parse_legacy_helios_prediction_headers(text, raw)
    assert len(games) == 1
    assert games[0].matchup == "UTA@DET"
    assert games[0].p_home == 0.54


def test_parse_compact_supports_game_line_format(tmp_path: Path) -> None:
    text = (
        "HELIOS_PREDICTION_HEADER\n"
        "GAME: NYR @ STL — 2025-12-18 — Pre-game\n"
        "WINNER: NYR\n"
        "p_home: 0.41\n"
    )
    raw = tmp_path / "12-18 gpt NHL.txt"
    raw.write_text(text, encoding="utf-8")
    games = parse_compact_prediction_headers(text, raw)
    assert len(games) == 1
    assert games[0].date == "2025-12-18"
    assert games[0].league == "nhl"
    assert games[0].matchup == "NYR@STL"
    assert games[0].p_home == 0.41
