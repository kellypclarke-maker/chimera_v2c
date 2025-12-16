import csv
from pathlib import Path

import chimera_v2c.tools.ingest_results as ir


class DummyScoreboard:
    @staticmethod
    def build():
        return {
            "games": [
                {
                    "league": "nba",
                    "teams": {"home": {"alias": "BOS"}, "away": {"alias": "NYK"}},
                    "scores": {"home": 100, "away": 90},
                }
            ]
        }


def test_ingest_results_writes_outcome(tmp_path, monkeypatch):
    # Prepare exec log
    exec_log = tmp_path / "exec_log.csv"
    with exec_log.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts_utc", "date", "ticker", "side", "count", "price_cents", "edge", "stake_fraction", "status", "message"])
        writer.writerow(["t1", "2025-12-03", "KXNBAGAME-25DEC04BOSNYK-BOS", "yes", "1", "50", "0.05", "0.01", "placed", "ok"])
    res_log = tmp_path / "results.csv"
    monkeypatch.setattr(ir, "EXEC_LOG", exec_log)
    monkeypatch.setattr(ir, "RESULTS_LOG", res_log)

    # Stub scoreboard fetch
    monkeypatch.setattr(ir, "fetch_scoreboard", lambda league, date_str: DummyScoreboard.build())

    ir.ingest_results("2025-12-03", "nba")

    assert res_log.exists()
    rows = list(csv.DictReader(res_log.open("r", encoding="utf-8")))
    assert len(rows) == 1
    r = rows[0]
    assert r["result"] == "win"
    assert set(r["matchup"].split("@")) == {"BOS", "NYK"}
