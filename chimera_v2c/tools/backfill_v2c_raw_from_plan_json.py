#!/usr/bin/env python
"""
Backfill v2c_raw (pre-calibration probability) from a saved v2c plan JSON (append-only).

This is useful when you have a stored plan JSON that contains:
  - per-game `key` (AWAY@HOME), and
  - `components.pre_calibration_final` (or `components.final`)
but your daily ledger only stored the calibrated `v2c`.

Safety:
- Respects `reports/daily_ledgers/locked/YYYYMMDD.lock` unless --force.
- Never overwrites non-blank cells; fills `v2c_raw` only when blank.
- Snapshots the daily ledger to `reports/daily_ledgers/snapshots/` before writing.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/backfill_v2c_raw_from_plan_json.py \\
    --league nhl --date 2025-12-12 --plan-json reports/execution_logs/v2c_plan_20251212_nhl.json --apply
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from chimera_v2c.src.ledger.guard import LedgerGuardError, compute_append_only_diff, load_locked_dates, snapshot_file


DAILY_DIR = Path("reports/daily_ledgers")
LOCK_DIR = DAILY_DIR / "locked"
SNAPSHOT_DIR = DAILY_DIR / "snapshots"


def _ledger_path(date_iso: str) -> Path:
    return DAILY_DIR / f"{date_iso.replace('-', '')}_daily_game_ledger.csv"


def _cell_blank(val: object) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    return str(val).strip() == ""


def _format_prob(val: object, decimals: int = 3) -> str:
    if val is None:
        return ""
    try:
        x = float(val)
    except Exception:
        return ""
    if x < 0.0 or x > 1.0:
        return ""
    return f"{x:.{decimals}f}"


def _extract_v2c_raw_map(plan_json_path: Path) -> Dict[str, str]:
    data = json.loads(plan_json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"[error] unexpected plan JSON (expected object): {plan_json_path}")
    plans = data.get("plans")
    if not isinstance(plans, list):
        raise SystemExit(f"[error] plan JSON missing 'plans' list: {plan_json_path}")

    out: Dict[str, str] = {}
    for p in plans:
        if not isinstance(p, dict):
            continue
        matchup = str(p.get("key") or "").strip()
        if not matchup or "@" not in matchup:
            continue
        comps = p.get("components") or {}
        if not isinstance(comps, dict):
            comps = {}
        raw = comps.get("pre_calibration_final")
        if raw is None:
            raw = comps.get("final")
        val = _format_prob(raw, 3)
        if not val:
            continue
        out[matchup] = val
    return out


def apply_backfill(
    *,
    ledger_path: Path,
    date_iso: str,
    league: str,
    v2c_raw_map: Dict[str, str],
    apply: bool,
    force: bool,
) -> Tuple[int, int]:
    if not ledger_path.exists():
        raise SystemExit(f"[error] daily ledger missing: {ledger_path}")

    ymd = ledger_path.name.split("_")[0]
    locked = load_locked_dates(LOCK_DIR)
    if ymd in locked and not force:
        raise SystemExit(f"[error] {ymd} is locked; refusing to modify {ledger_path} (pass --force to override)")

    original = pd.read_csv(ledger_path).fillna("")
    df = original.copy()
    for col in ("date", "league", "matchup"):
        if col not in df.columns:
            raise SystemExit(f"[error] ledger missing required column '{col}': {ledger_path}")
    if "v2c_raw" not in df.columns:
        df["v2c_raw"] = ""

    added_rows = 0
    filled_cells = 0

    for matchup, raw_val in v2c_raw_map.items():
        mask = (
            (df["date"].astype(str) == date_iso)
            & (df["league"].astype(str).str.lower() == league.lower())
            & (df["matchup"].astype(str) == matchup)
        )
        if not mask.any():
            new_row = {c: "" for c in df.columns}
            new_row.update({"date": date_iso, "league": league.lower(), "matchup": matchup})
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            added_rows += 1
            mask = (
                (df["date"].astype(str) == date_iso)
                & (df["league"].astype(str).str.lower() == league.lower())
                & (df["matchup"].astype(str) == matchup)
            )

        idx = df.index[mask][0]
        if _cell_blank(df.at[idx, "v2c_raw"]) and raw_val:
            df.at[idx, "v2c_raw"] = raw_val
            filled_cells += 1

    if not apply:
        return added_rows, filled_cells

    try:
        old_rows = original.fillna("").astype(str).to_dict("records")
        new_rows = df.fillna("").astype(str).to_dict("records")
        key_fields = ["date", "league", "matchup"]
        compute_append_only_diff(
            old_rows=old_rows,
            new_rows=new_rows,
            key_fields=key_fields,
            value_fields=[c for c in df.columns if c not in key_fields],
        )
    except LedgerGuardError as exc:
        raise SystemExit(f"[error] append-only guard failed: {exc}") from exc

    snapshot_file(ledger_path, SNAPSHOT_DIR)
    df.to_csv(ledger_path, index=False)
    return added_rows, filled_cells


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill v2c_raw from a saved v2c plan JSON (append-only).")
    ap.add_argument("--league", required=True, help="League (nba|nhl|nfl).")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD.")
    ap.add_argument("--plan-json", required=True, help="Path to v2c plan JSON with components/pre_calibration_final.")
    ap.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run).")
    ap.add_argument("--force", action="store_true", help="Allow edits to locked ledgers (still fills blanks only).")
    args = ap.parse_args()

    date_iso = datetime.strptime(args.date, "%Y-%m-%d").date().isoformat()
    ledger_path = _ledger_path(date_iso)
    plan_path = Path(args.plan_json)
    if not plan_path.exists():
        raise SystemExit(f"[error] plan JSON missing: {plan_path}")

    v2c_raw_map = _extract_v2c_raw_map(plan_path)
    if not v2c_raw_map:
        raise SystemExit(f"[error] no v2c_raw values found in plan JSON: {plan_path}")

    added, filled = apply_backfill(
        ledger_path=ledger_path,
        date_iso=date_iso,
        league=args.league.lower(),
        v2c_raw_map=v2c_raw_map,
        apply=bool(args.apply),
        force=bool(args.force),
    )

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] {ledger_path} rows_added={added} cells_filled={filled} games={len(v2c_raw_map)}")


if __name__ == "__main__":
    main()

