"""
Fit Platt calibration parameters from ledger outcomes and write JSON for pipeline use.

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/fit_calibration.py --league nhl --out chimera_v2c/data/calibration_params_nhl.json

Reads: reports/master_ledger/master_game_ledger.csv
Requires: columns `v2c` and `actual_outcome` (final score like `AWAY 2-3 HOME`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from chimera_v2c.src.calibration import fit_platt
from chimera_v2c.src.ledger.outcomes import parse_home_win


LEDGER_PATH = Path("reports/master_ledger/master_game_ledger.csv")


def _parse_home_win(outcome: object) -> Optional[int]:
    hw = parse_home_win(outcome)
    if hw == 1.0:
        return 1
    if hw == 0.0:
        return 0
    return None


def load_pairs(league: str) -> List[Tuple[float, int]]:
    if not LEDGER_PATH.exists():
        raise SystemExit(f"[error] ledger missing at {LEDGER_PATH}")
    df = pd.read_csv(LEDGER_PATH)
    if "v2c" not in df.columns:
        raise SystemExit("[error] ledger missing v2c column")
    if "actual_outcome" not in df.columns:
        raise SystemExit("[error] ledger missing actual_outcome column")

    df = df[df["league"].astype(str).str.lower() == league.lower()].copy()
    preds = pd.to_numeric(df["v2c"], errors="coerce")
    ys = df["actual_outcome"].apply(_parse_home_win)
    pairs = []
    for p, y in zip(preds, ys):
        if pd.isna(p):
            continue
        if y in (0, 1):
            pairs.append((float(p), int(y)))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit Platt calibration params from ledger.")
    ap.add_argument("--league", default="nhl", help="League filter (default nhl).")
    ap.add_argument("--out", default="chimera_v2c/data/calibration_params_nhl.json", help="Output JSON path.")
    ap.add_argument("--min-samples", type=int, default=20, help="Minimum samples to fit (default 20).")
    args = ap.parse_args()

    pairs = load_pairs(args.league)
    if len(pairs) < args.min_samples:
        print(f"[warn] only {len(pairs)} samples; writing identity calibration.")
        payload = {"a": 1.0, "b": 0.0, "n": len(pairs)}
    else:
        scaler = fit_platt(pairs)
        payload = {"a": scaler.a, "b": scaler.b, "n": len(pairs)}
        print(f"[ok] fit calibration on n={len(pairs)}: a={scaler.a:.4f}, b={scaler.b:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ok] wrote calibration params to {out_path}")


if __name__ == "__main__":
    main()
