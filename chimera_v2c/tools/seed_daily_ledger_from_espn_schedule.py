#!/usr/bin/env python
"""
Seed a daily ledger with today's scheduled games from the ESPN scoreboard (append-only).

Purpose
-------
`ensure_daily_ledger.py` creates the per-day file, but for "live" dates (>= 2025-12-04)
there is no archived master source to populate rows. This tool fills that gap by
appending missing (date, league, matchup) rows based on ESPN's schedule.

Rules
-----
- Append-only: adds missing rows only; never overwrites existing non-blank values.
- Probability-like columns are initialized to `NR` to preserve the daily-ledger schema
  invariant ("no blank prob cells"). `actual_outcome` stays blank.
- Respects `reports/daily_ledgers/locked/YYYYMMDD.lock` unless `--force`.

Usage
-----
  PYTHONPATH=. python chimera_v2c/tools/seed_daily_ledger_from_espn_schedule.py \
    --date YYYY-MM-DD --leagues nba,nhl,nfl --apply
"""

from __future__ import annotations

import argparse
from datetime import date as date_type
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from chimera_v2c.lib import espn_schedule, team_mapper
from chimera_v2c.src.ledger.formatting import DAILY_LEDGER_COLUMNS, DAILY_LEDGER_PROB_COLUMNS, MISSING_SENTINEL
from chimera_v2c.src.ledger.guard import (
    LedgerGuardError,
    compute_append_only_diff,
    load_csv_records,
    load_locked_dates,
    snapshot_file,
    write_csv,
)


DAILY_DIR = Path("reports/daily_ledgers")
LOCK_DIR = DAILY_DIR / "locked"
SNAPSHOT_DIR = DAILY_DIR / "snapshots"


def _ledger_path(date_iso: str) -> Path:
    ymd = date_iso.replace("-", "")
    return DAILY_DIR / f"{ymd}_daily_game_ledger.csv"


def _key(date_iso: str, league: str, matchup: str) -> Tuple[str, str, str]:
    return (date_iso, league.lower(), matchup.strip())


def normalize_matchup_key(league: str, away_abbr: str, home_abbr: str) -> str:
    away_norm = team_mapper.normalize_team_code(away_abbr, league)
    home_norm = team_mapper.normalize_team_code(home_abbr, league)
    return f"{away_norm}@{home_norm}".upper()


def rows_from_scoreboard(*, league: str, date_iso: str, scoreboard: Dict) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for m in espn_schedule.extract_matchups(league, scoreboard):
        matchup = normalize_matchup_key(league, m.get("away_abbr", ""), m.get("home_abbr", ""))
        row = {c: "" for c in DAILY_LEDGER_COLUMNS}
        row["date"] = date_iso
        row["league"] = league.lower()
        row["matchup"] = matchup
        for col in DAILY_LEDGER_PROB_COLUMNS:
            row[col] = MISSING_SENTINEL
        row["actual_outcome"] = ""
        rows.append(row)
    return rows


def seed_daily_ledger_from_rows(
    *,
    ledger_path: Path,
    rows_to_add: Sequence[Dict[str, str]],
    apply: bool,
    force: bool,
) -> Tuple[int, int]:
    if not ledger_path.exists():
        raise SystemExit(f"[error] daily ledger missing: {ledger_path}")

    ymd = ledger_path.name.split("_")[0]
    lock_dir = ledger_path.parent / "locked"
    locked = load_locked_dates(lock_dir)
    if ymd in locked and not force:
        raise SystemExit(f"[error] {ymd} is locked; refusing to modify {ledger_path} (pass --force to override)")

    old_rows = load_csv_records(ledger_path)
    existing_keys = {_key(r.get("date", ""), r.get("league", ""), r.get("matchup", "")) for r in old_rows}

    added = 0
    new_rows: List[Dict[str, str]] = list(old_rows)
    for r in rows_to_add:
        k = _key(r.get("date", ""), r.get("league", ""), r.get("matchup", ""))
        if k in existing_keys:
            continue
        new_rows.append({c: (r.get(c) or "") for c in DAILY_LEDGER_COLUMNS})
        existing_keys.add(k)
        added += 1

    if not apply:
        return added, len(old_rows)

    snapshot_dir = ledger_path.parent / "snapshots"
    snapshot_file(ledger_path, snapshot_dir)
    try:
        compute_append_only_diff(
            old_rows=old_rows,
            new_rows=new_rows,
            key_fields=["date", "league", "matchup"],
            value_fields=[c for c in DAILY_LEDGER_COLUMNS if c not in {"date", "league", "matchup"}],
            blank_sentinels={MISSING_SENTINEL},
        )
    except LedgerGuardError as exc:
        raise SystemExit(f"[error] append-only guard failed: {exc}") from exc

    write_csv(ledger_path, new_rows, DAILY_LEDGER_COLUMNS)
    return added, len(old_rows)


def _parse_leagues(value: str) -> List[str]:
    parts = [p.strip().lower() for p in (value or "").split(",") if p.strip()]
    if not parts:
        return []
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description="Append missing daily-ledger rows from ESPN scoreboard schedule.")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD.")
    ap.add_argument("--leagues", default="nba,nhl,nfl", help="Comma-separated leagues (default: nba,nhl,nfl).")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run).")
    ap.add_argument("--force", action="store_true", help="Allow edits to locked ledgers (still append-only).")
    args = ap.parse_args()

    date_iso = datetime.strptime(args.date, "%Y-%m-%d").date().isoformat()
    ledger_path = _ledger_path(date_iso)
    if not ledger_path.exists():
        raise SystemExit(f"[error] missing daily ledger; run ensure_daily_ledger.py first: {ledger_path}")

    leagues = _parse_leagues(args.leagues)
    if not leagues:
        raise SystemExit("[error] no leagues specified")

    dt = date_type.fromisoformat(date_iso)
    seed_rows: List[Dict[str, str]] = []
    for league in leagues:
        sb = espn_schedule.get_scoreboard(league, dt)
        seed_rows.extend(rows_from_scoreboard(league=league, date_iso=date_iso, scoreboard=sb))

    added, prior = seed_daily_ledger_from_rows(
        ledger_path=ledger_path,
        rows_to_add=seed_rows,
        apply=bool(args.apply),
        force=bool(args.force),
    )
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] {ledger_path} prior_rows={prior} rows_added={added}")


if __name__ == "__main__":
    main()
