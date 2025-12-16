#!/usr/bin/env python
"""
Format a single daily ledger to the canonical, operator-readable schema.

This tool is intentionally *single-date* and explicit. It is meant to be run
when a day's ledger is being finalized for readability / consistency:
- Enforces a fixed column set/order (drops noisy/deprecated columns).
- Formats probability-like fields to 2 decimals and drops the leading zero
  (e.g., 0.846 -> .85).
- Replaces blank cells with "NR" for probability-like columns; leaves
  `actual_outcome` blank until the game is final.

Safety:
- Refuses to modify locked dates unless --force.
- Snapshots the prior ledger file under reports/daily_ledgers/snapshots/.
- Does not delete rows; errors on duplicate (date,league,matchup) keys.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/format_daily_ledger.py --date YYYY-MM-DD --apply
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from chimera_v2c.src.ledger.formatting import (
    DAILY_LEDGER_COLUMNS,
    DAILY_LEDGER_PROB_COLUMNS,
    MISSING_SENTINEL,
    format_prob_cell,
)
from chimera_v2c.src.ledger.guard import load_locked_dates, snapshot_file
from chimera_v2c.src.ledger.outcomes import format_final_score, is_placeholder_outcome, parse_scores


DAILY_DIR = Path("reports/daily_ledgers")
LOCK_DIR = DAILY_DIR / "locked"
SNAPSHOT_DIR = DAILY_DIR / "snapshots"


def _ledger_path(date_iso: str) -> Path:
    ymd = date_iso.replace("-", "")
    return DAILY_DIR / f"{ymd}_daily_game_ledger.csv"


def _normalize_row(row: Dict[str, str], *, date_iso: str, decimals: int) -> Dict[str, str]:
    league = (row.get("league") or "").strip().lower()
    matchup = (row.get("matchup") or "").strip().upper()
    away_team = ""
    home_team = ""
    if "@" in matchup:
        away_team, home_team = matchup.split("@", 1)
        away_team = away_team.strip()
        home_team = home_team.strip()

    out: Dict[str, str] = {}
    out["date"] = (row.get("date") or "").strip() or date_iso
    out["league"] = league
    out["matchup"] = matchup

    for col in DAILY_LEDGER_COLUMNS:
        if col in ("date", "league", "matchup"):
            continue

        raw = (row.get(col) or "").strip()
        if col in DAILY_LEDGER_PROB_COLUMNS:
            if col == "moneypuck" and league != "nhl":
                raw = MISSING_SENTINEL
            if not raw:
                raw = MISSING_SENTINEL
            formatted = format_prob_cell(raw, decimals=decimals, drop_leading_zero=True)
            out[col] = formatted or MISSING_SENTINEL
            continue

        if col == "actual_outcome":
            if not raw:
                out[col] = ""
                continue
            if is_placeholder_outcome(raw):
                out[col] = ""
                continue
            scores = parse_scores(raw)
            if scores is None or not away_team or not home_team:
                out[col] = raw
                continue
            away_score, home_score = scores
            out[col] = format_final_score(away_team, home_team, away_score, home_score)
            continue

        out[col] = raw or MISSING_SENTINEL

    return out


def format_daily_ledger(
    *,
    ledger_path: Path,
    date_iso: str,
    decimals: int,
    apply: bool,
    force: bool,
) -> Tuple[int, int, List[str]]:
    if not ledger_path.exists():
        raise SystemExit(f"[error] missing daily ledger: {ledger_path}")

    ymd = ledger_path.name.split("_", 1)[0]
    locked = load_locked_dates(LOCK_DIR)
    if ymd in locked and not force:
        raise SystemExit(f"[error] {ymd} is locked; refusing to modify {ledger_path} (pass --force to override)")

    with ledger_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        in_columns = list(reader.fieldnames or [])
        rows_in = list(reader)

    rows_out: List[Dict[str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    changes = 0

    for r in rows_in:
        normalized = _normalize_row(r, date_iso=date_iso, decimals=decimals)
        key = (normalized["date"], normalized["league"], normalized["matchup"])
        if key in seen:
            raise SystemExit(f"[error] duplicate key in {ledger_path}: {key}")
        seen.add(key)

        # Count changes only on canonical columns.
        for col in DAILY_LEDGER_COLUMNS:
            before = (r.get(col) or "").strip()
            after = (normalized.get(col) or "").strip()
            if before != after:
                changes += 1
        rows_out.append(normalized)

    # Validate: no blank cells (except actual_outcome, which may be blank)
    for idx, r in enumerate(rows_out, start=1):
        for col in DAILY_LEDGER_COLUMNS:
            if str(r.get(col) or "").strip() == "":
                if col == "actual_outcome":
                    continue
                raise SystemExit(f"[error] blank cell after formatting: row={idx} col={col} key={r.get('matchup')}")

    dropped = [c for c in in_columns if c not in DAILY_LEDGER_COLUMNS]

    if not apply:
        return len(rows_out), changes, dropped

    snapshot_file(ledger_path, SNAPSHOT_DIR)
    with ledger_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DAILY_LEDGER_COLUMNS)
        writer.writeheader()
        for r in rows_out:
            writer.writerow({k: r.get(k, "") for k in DAILY_LEDGER_COLUMNS})

    return len(rows_out), changes, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description="Format a single daily ledger to the canonical schema.")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD.")
    ap.add_argument("--decimals", type=int, default=2, help="Decimals for probabilities (default: 2).")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run).")
    ap.add_argument("--force", action="store_true", help="Allow edits to locked ledgers.")
    args = ap.parse_args()

    date_iso = args.date.strip()
    ledger_path = _ledger_path(date_iso)

    rows, changes, dropped = format_daily_ledger(
        ledger_path=ledger_path,
        date_iso=date_iso,
        decimals=int(args.decimals),
        apply=bool(args.apply),
        force=bool(args.force),
    )

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] {ledger_path} rows={rows} cells_changed={changes} dropped_cols={len(dropped)}")
    if dropped:
        print(f"  dropped: {', '.join(dropped)}")


if __name__ == "__main__":
    main()
