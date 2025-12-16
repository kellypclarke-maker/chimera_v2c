import json
from pathlib import Path

from chimera_v2c.src import logging as v2c_logging
import chimera_v2c.tools.execute_plan as exec_mod


def test_check_sentinel_respects_flag(tmp_path, monkeypatch):
    flag = tmp_path / "STOP_TRADING.flag"
    monkeypatch.setattr(exec_mod, "SENTINEL_PATH", flag)
    assert exec_mod.check_sentinel() is True
    flag.write_text("halt")
    assert exec_mod.check_sentinel() is False


def test_append_log_creates_header(tmp_path):
    log_path = tmp_path / "v2c_execution_log.csv"
    rows = [
        {
            "date": "2025-12-04",
            "ticker": "KXNBAGAME-TEST",
            "side": "yes",
            "count": 1,
            "price_cents": 50,
            "edge": 0.05,
            "stake_fraction": 0.01,
            "status": "dry_run",
            "message": "not placed",
        }
    ]
    v2c_logging.append_log(rows, path=log_path)
    contents = log_path.read_text(encoding="utf-8").strip().splitlines()
    # header + 1 row
    assert len(contents) == 2
    assert "ticker" in contents[0]
    assert "KXNBAGAME-TEST" in contents[1]
