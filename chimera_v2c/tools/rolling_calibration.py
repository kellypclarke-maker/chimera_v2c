"""
Rolling calibration/accuracy from daily_ledgers (non-destructive).

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/rolling_calibration.py --days 14 --league nba

Reads `reports/daily_ledgers/*.csv` and computes accuracy/Brier
for v2c, Gemini, Grok, and Kalshi over the last N days (by filename date).
"""
from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from chimera_v2c.src.ledger.outcomes import parse_home_win

LEDGER_DIR = Path("reports/daily_ledgers")
MODELS = ["v2c", "gemini", "grok", "kalshi_mid"]


def outcome_home_win(s: str) -> float | None:
    return parse_home_win(s)


def accuracy(prob_str: str, home_win: float | None) -> float | None:
    if home_win is None:
        return None
    if prob_str is None or str(prob_str).strip() == "" or str(prob_str).lower() in {"nan", "none"}:
        return None
    p = float(prob_str)
    if home_win == 0.5:
        return None
    pred = 1 if p >= 0.5 else 0
    return 1.0 if pred == home_win else 0.0


def brier(prob_str: str, home_win: float | None) -> float | None:
    if home_win is None:
        return None
    if prob_str is None or str(prob_str).strip() == "" or str(prob_str).lower() in {"nan", "none"}:
        return None
    p = float(prob_str)
    return (p - home_win) ** 2


def parse_date_from_name(path: Path) -> datetime:
    # filenames like 20251208_daily_game_ledger.csv
    stem = path.stem
    m = re.match(r"(\\d{8})_", stem)
    if not m:
        return datetime.min
    return datetime.strptime(m.group(1), "%Y%m%d")


def load_ledgers(days: int, league_filter: str | None) -> List[Dict]:
    files = sorted(LEDGER_DIR.glob("*_daily_game_ledger.csv"), key=parse_date_from_name)
    if days:
        files = files[-days:]
    rows: List[Dict] = []
    for f in files:
        with f.open() as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                if league_filter and row.get("league", "").lower() != league_filter:
                    continue
                row["_file"] = f.name
                rows.append(row)
    return rows


def summarize(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    res = {m: {"acc_sum": 0.0, "acc_n": 0, "brier_sum": 0.0, "brier_n": 0} for m in MODELS}
    for r in rows:
        hw = outcome_home_win(r.get("actual_outcome", ""))
        for m in MODELS:
            acc = accuracy(r.get(m, ""), hw)
            if acc is not None:
                res[m]["acc_sum"] += acc
                res[m]["acc_n"] += 1
            br = brier(r.get(m, ""), hw)
            if br is not None:
                res[m]["brier_sum"] += br
                res[m]["brier_n"] += 1
    out = {}
    for m in MODELS:
        acc_n = res[m]["acc_n"]
        br_n = res[m]["brier_n"]
        acc = res[m]["acc_sum"] / acc_n if acc_n else None
        br = res[m]["brier_sum"] / br_n if br_n else None
        out[m] = {"games": acc_n, "acc": acc, "brier": br}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Rolling calibration over daily ledgers.")
    ap.add_argument("--days", type=int, default=14, help="How many most recent ledger days to include (default: 14).")
    ap.add_argument("--league", help="Optional league filter (nba|nhl|nfl).")
    args = ap.parse_args()

    rows = load_ledgers(args.days, args.league.lower() if args.league else None)
    if not rows:
        raise SystemExit("[error] no rows found for given filters")
    summary = summarize(rows)
    print(f"[info] rows included: {len(rows)} from last {args.days} day(s)")
    for m, stats in summary.items():
        acc = stats["acc"]
        br = stats["brier"]
        if acc is None:
            print(f"{m}: no data")
            continue
        print(f"{m}: acc={acc:.3f} ({stats['games']} games), brier={br:.3f}")


if __name__ == "__main__":
    main()
