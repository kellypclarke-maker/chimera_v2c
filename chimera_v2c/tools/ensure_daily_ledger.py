"""
Ensure today's daily game ledger exists (one file per date, no overwrites by default).

This is intended to be "step #1" each day:
  PYTHONPATH=. python chimera_v2c/tools/ensure_daily_ledger.py

Behavior:
  - Determines the target date (default: today, local time).
  - If the per-day ledger already exists, it exits with a skip message (unless --overwrite).
  - Otherwise, it writes the daily ledger using build_daily_game_ledgers with --allow-empty
    so the file exists even if predictions haven't landed yet.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from chimera_v2c.tools.build_daily_game_ledgers import build_daily_ledgers, DEFAULT_OUT_DIR, DEFAULT_SOURCE


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensure today's daily ledger exists (header-only if no rows yet).")
    ap.add_argument(
        "--date",
        help="Target date (YYYY-MM-DD). Defaults to today (local time).",
    )
    ap.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Source ledger (default: game_level_ml_master.csv)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for daily ledgers",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing daily file (default: skip if present).",
    )
    args = ap.parse_args()

    target_date = args.date or date.today().isoformat()

    build_daily_ledgers(
        source=Path(args.source),
        out_dir=Path(args.out_dir),
        dates={target_date},
        overwrite=args.overwrite,
        allow_empty=True,
    )


if __name__ == "__main__":
    main()
