"""
Build a filtered game-level ledger that only includes games with all key models:
  - v2c
  - Gemini
  - Grok
  - Market mid

GPT stays as a column and may be blank.

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/build_complete_game_ledger.py \
    --source reports/specialist_performance/game_level_ml_master.csv \
    --out reports/specialist_performance/game_level_ml_complete.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict

DEFAULT_SOURCE = Path("reports/specialist_performance/game_level_ml_master.csv")
DEFAULT_OUT = Path("reports/specialist_performance/game_level_ml_complete.csv")

REQUIRED_COLS = ["p_home_v2c", "p_home_gemini", "p_home_grok", "p_home_market"]


def is_present(val: str) -> bool:
    if val is None:
        return False
    s = str(val).strip()
    if not s:
        return False
    if s.lower() in {"nan", "none"}:
        return False
    return True


def build_complete_ledger(source: Path, out_path: Path) -> None:
    if not source.exists():
        raise SystemExit(f"[error] source ledger not found: {source}")

    with source.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLS if c not in fieldnames]
        if missing:
            raise SystemExit(f"[error] required columns missing from source: {missing}")

        kept_rows: List[Dict[str, str]] = []
        for row in reader:
            if all(is_present(row.get(col)) for col in REQUIRED_COLS):
                kept_rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"[info] kept rows (all required model fields present): {len(kept_rows)}")
    print(f"[info] wrote complete ledger: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a filtered ledger containing only games with v2c, Gemini, Grok, and market mids.")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="Source ledger path (default: game_level_ml_master.csv)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output path for the filtered ledger")
    args = ap.parse_args()

    build_complete_ledger(Path(args.source), Path(args.out))


if __name__ == "__main__":
    main()
