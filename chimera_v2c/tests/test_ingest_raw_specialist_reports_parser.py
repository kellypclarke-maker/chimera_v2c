from __future__ import annotations

from pathlib import Path

from chimera_v2c.tools.ingest_raw_specialist_reports_v2c import parse_compact_prediction_headers


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

