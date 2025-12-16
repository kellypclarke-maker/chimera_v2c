#!/usr/bin/env python
"""
Backfill MoneyPuck pre-game win probabilities into daily ledgers (append-only).

MoneyPuck publishes per-game pregame predictions at:
  https://moneypuck.com/moneypuck/predictions/<gameID>.csv

This tool:
- Finds NHL gameIDs for a date from the MoneyPuck season schedule JSON.
- Fetches each game’s `preGameMoneyPuckHomeWinPrediction`.
- Fills blank `moneypuck` cells in `reports/daily_ledgers/YYYYMMDD_daily_game_ledger.csv`.
- Rich provenance fields belong in `reports/market_snapshots/` via `external_snapshot.py`.

Safety:
- Respects `reports/daily_ledgers/locked/YYYYMMDD.lock` unless --force.
- Never overwrites non-blank cells.
- Dry-run by default; pass --apply to write.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/backfill_moneypuck_pregame.py --date YYYY-MM-DD --apply
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.lib.moneypuck import fetch_pregame, fetch_schedule, games_for_date, season_string_for_date
from chimera_v2c.src.ledger.guard import (
    LedgerGuardError,
    compute_append_only_diff,
    load_locked_dates,
    snapshot_file,
)


DAILY_DIR = Path("reports/daily_ledgers")
LOCK_DIR = DAILY_DIR / "locked"
LEDGER_SNAPSHOT_DIR = DAILY_DIR / "snapshots"


def _ledger_path(date_iso: str) -> Path:
    return DAILY_DIR / f"{date_iso.replace('-', '')}_daily_game_ledger.csv"


def ensure_daily_ledger(date_iso: str) -> None:
    path = _ledger_path(date_iso)
    if path.exists():
        return
    cmd = [
        sys.executable,
        "chimera_v2c/tools/ensure_daily_ledger.py",
        "--date",
        date_iso,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
    subprocess.run(cmd, check=True, env=env)


def _cell_blank(val: object) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    return str(val).strip() == ""


def _fmt_prob(p: float, decimals: int = 4) -> str:
    from chimera_v2c.src.ledger.formatting import format_prob_cell

    return format_prob_cell(p, decimals=decimals, drop_leading_zero=True)


def build_map(date_iso: str) -> Dict[str, Dict[str, str]]:
    dt = datetime.strptime(date_iso, "%Y-%m-%d").date()
    season = season_string_for_date(dt)
    schedule = fetch_schedule(season)
    games = games_for_date(schedule, date_iso)
    out: Dict[str, Dict[str, str]] = {}
    for g in games:
        pre = fetch_pregame(g.game_id)
        if pre.moneypuck_home_win is None:
            continue
        out[g.matchup] = {
            "moneypuck": _fmt_prob(pre.moneypuck_home_win, 2),
        }
    return out


def apply_to_ledger(
    *,
    ledger_path: Path,
    date_iso: str,
    mapping: Dict[str, Dict[str, str]],
    apply: bool,
    force: bool,
) -> Tuple[int, int]:
    if not ledger_path.exists():
        raise SystemExit(f"[error] daily ledger missing: {ledger_path}")
    ymd = ledger_path.name.split("_")[0]
    locked = load_locked_dates(LOCK_DIR)
    if apply and ymd in locked and not force:
        raise SystemExit(f"[error] {ymd} is locked; refusing to modify {ledger_path} (pass --force to override)")

    original = pd.read_csv(ledger_path).fillna("")
    df = original.copy()

    for col in ["moneypuck"]:
        if col not in df.columns:
            df[col] = ""

    added_rows = 0
    filled_cells = 0

    for matchup, payload in mapping.items():
        mask = (
            (df["date"].astype(str) == date_iso)
            & (df["league"].astype(str).str.lower() == "nhl")
            & (df["matchup"].astype(str) == matchup)
        )
        if not mask.any():
            # Append minimal row for this matchup.
            new_row = {c: "" for c in df.columns}
            new_row.update({"date": date_iso, "league": "nhl", "matchup": matchup})
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            added_rows += 1
            mask = (
                (df["date"].astype(str) == date_iso)
                & (df["league"].astype(str).str.lower() == "nhl")
                & (df["matchup"].astype(str) == matchup)
            )

        idx = df.index[mask][0]
        new_val = (payload.get("moneypuck") or "").strip()
        if new_val and _cell_blank(df.at[idx, "moneypuck"]):
            df.at[idx, "moneypuck"] = new_val
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

    snapshot_file(ledger_path, LEDGER_SNAPSHOT_DIR)
    df.to_csv(ledger_path, index=False)
    return added_rows, filled_cells


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill MoneyPuck pregame win probs into daily ledgers (append-only).")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD.")
    ap.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run).")
    ap.add_argument("--force", action="store_true", help="Allow edits to locked ledgers (still fills blanks only).")
    args = ap.parse_args()

    load_env_from_env_list()
    date_iso = datetime.strptime(args.date, "%Y-%m-%d").date().isoformat()
    ledger_path = _ledger_path(date_iso)
    if args.apply:
        ensure_daily_ledger(date_iso)
    if not ledger_path.exists():
        print(f"[warn] daily ledger missing for {date_iso}: {ledger_path}")
        if not args.apply:
            print("[info] run with --apply to create/fill the ledger")
        return

    mapping = build_map(date_iso)
    if not mapping:
        print(f"[warn] no MoneyPuck games/probabilities found for {date_iso}")
        return

    added, filled = apply_to_ledger(
        ledger_path=ledger_path,
        date_iso=date_iso,
        mapping=mapping,
        apply=bool(args.apply),
        force=bool(args.force),
    )
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] {ledger_path} rows_added={added} cells_filled={filled} games={len(mapping)}")


if __name__ == "__main__":
    main()
