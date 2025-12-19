from __future__ import annotations

from pathlib import Path

from chimera_v2c.tools.ingest_raw_specialist_reports_v2c import ingest_file, parse_compact_prediction_headers


def test_parse_compact_prediction_headers_infers_league_from_filename(tmp_path: Path) -> None:
    raw = tmp_path / "12-16 gpt NHL.txt"
    raw.write_text(
        "\n".join(
            [
                "HELIOS_PREDICTION_HEADER",
                "DATE: 2025-12-16",
                "MATCHUP: UTA@BOS",
                "P_HOME: 0.505",
                "WINNER: BOS",
                "",
                "HELIOS_PREDICTION_HEADER",
                "DATE: 2025-12-16",
                "MATCHUP: CHI@TOR",
                "P_HOME: 0.649",
                "WINNER: TOR",
            ]
        ),
        encoding="utf-8",
    )

    games = parse_compact_prediction_headers(raw.read_text(encoding="utf-8"), raw)
    assert len(games) == 2
    assert games[0].league == "nhl"
    assert games[0].matchup == "UTA@BOS"
    assert games[0].p_home is not None
    assert abs(games[0].p_home - 0.505) < 1e-9


def test_ingest_file_merges_multiple_formats(tmp_path: Path, monkeypatch) -> None:
    # Avoid any accidental OpenAI fallbacks.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    text = (
        "HELIOS_PREDICTION_HEADER_START\n"
        "Game: 2025-12-18 NHL CHI@MTL\n"
        "Model: GPT_Scientist_v8.5\n"
        "Prediction_Moneyline:\n"
        "Winner: MTL\n"
        "p_home: 0.620\n"
        "HELIOS_PREDICTION_HEADER_END\n"
        "\n---\n\n"
        "HELIOS_PREDICTION_HEADER\n"
        "GAME: NYR @ STL — 2025-12-18 — Pre-game\n"
        "WINNER: NYR\n"
        "p_home: 0.41\n"
    )
    raw = tmp_path / "12-18 gpt NHL.txt"
    raw.write_text(text, encoding="utf-8")
    result = ingest_file(raw, model_override=None, openai_model="gpt-4o-mini", apply=False, force=False)
    assert result["error"] is None
    assert len(result["raw_games"]) == 2
