from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Legacy folder name `betting_ladders` was retired; execution logs now live in reports/execution_logs.
LOG_PATH = Path("reports/execution_logs/v2c_execution_log.csv")


def ensure_log_header(path: Path = LOG_PATH) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ts_utc",
                "date",
                "ticker",
                "side",
                "count",
                "price_cents",
                "edge",
                "stake_fraction",
                "status",
                "message",
            ]
        )


def append_log(rows: List[Dict], path: Path = LOG_PATH) -> None:
    ensure_log_header(path)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(
                [
                    datetime.now(timezone.utc).isoformat(),
                    row.get("date"),
                    row.get("ticker"),
                    row.get("side"),
                    row.get("count"),
                    row.get("price_cents"),
                    row.get("edge"),
                    row.get("stake_fraction"),
                    row.get("status"),
                    row.get("message"),
                ]
            )
