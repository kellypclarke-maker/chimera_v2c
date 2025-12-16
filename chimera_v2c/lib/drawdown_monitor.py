from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Optional

LEDGER_PATH = Path("reports/specialist_performance/specialist_manual_ledger.csv")


def _parse_date(val: str) -> Optional[date]:
    try:
        return datetime.fromisoformat(val).date()
    except Exception:
        return None


def should_halt_trading(window_days: int = 7, max_drawdown_pct: float = 0.25) -> bool:
    if not LEDGER_PATH.exists():
        return False
    today = datetime.utcnow().date()
    window_start = today.replace(day=max(1, today.day - window_days))
    equity_points = []
    with LEDGER_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = _parse_date(row.get("game_date") or row.get("date") or "")
            if not d or d < window_start or d > today:
                continue
            res = (row.get("ml_result") or row.get("result") or "").lower()
            if res not in {"win", "loss"}:
                continue
            equity_points.append(1.0 if res == "win" else -1.0)
    if not equity_points:
        return False
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for delta in equity_points:
        equity += delta
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    if peak <= 0.0:
        return False
    drawdown_ratio = max_dd / max(1.0, peak)
    return drawdown_ratio >= max_drawdown_pct
