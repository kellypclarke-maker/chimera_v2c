"""
Snapshot the game-level master ledger without mutating it.

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/snapshot_game_level_ledger.py
  PYTHONPATH=. python chimera_v2c/tools/snapshot_game_level_ledger.py --out reports/specialist_performance/snapshots/custom.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil

LEDGER_PATH = Path("reports/specialist_performance/game_level_ml_master.csv")
SNAPSHOT_DIR = Path("reports/specialist_performance/snapshots")


def snapshot(out_path: Path) -> None:
    if not LEDGER_PATH.exists():
        raise SystemExit(f"[error] ledger not found at {LEDGER_PATH}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LEDGER_PATH, out_path)
    print(f"[info] snapshot written: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy the game-level master ledger to a timestamped snapshot.")
    ap.add_argument(
        "--out",
        help="Optional snapshot path. Defaults to snapshots/game_level_ml_master_<timestamp>.csv",
        default=None,
    )
    args = ap.parse_args()

    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = SNAPSHOT_DIR / f"game_level_ml_master_{ts}.csv"

    snapshot(out_path)


if __name__ == "__main__":
    main()
