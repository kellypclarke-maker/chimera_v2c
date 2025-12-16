#!/usr/bin/env python
"""
Fill per-day daily ledgers from the v2c planner output (append-only).

This tool is meant to remove manual copy/paste steps after running the planner:
- Ensures `reports/daily_ledgers/YYYYMMDD_daily_game_ledger.csv` exists (no overwrite).
- Adds missing rows for the league/date matchups on the slate.
- Fills blank cells only:
  - `v2c` (home win prob; uses GamePlan.p_final)
  - `kalshi_mid` (home-implied mid from the HOME-YES market when available)

Safety:
- Respects `reports/daily_ledgers/locked/YYYYMMDD.lock` unless --force.
- Never overwrites non-blank cells.
- Dry-run by default; pass --apply to write.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/fill_daily_ledger_from_plan.py \
    --date YYYY-MM-DD --config chimera_v2c/config/nhl_defaults.yaml --apply
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from chimera_v2c.src.config_loader import V2CConfig
from chimera_v2c.src.ledger.guard import LedgerGuardError, compute_append_only_diff, load_locked_dates
from chimera_v2c.src.pipeline import build_daily_plan


DAILY_DIR = Path("reports/daily_ledgers")
LOCK_DIR = DAILY_DIR / "locked"

DEFAULT_COLUMNS = [
    "date",
    "league",
    "matchup",
    "v2c",
    "grok",
    "gemini",
    "gpt",
    "kalshi_mid",
    "market_proxy",
    "moneypuck",
    "actual_outcome",
]


def _ledger_path(date_iso: str) -> Path:
    ymd = date_iso.replace("-", "")
    return DAILY_DIR / f"{ymd}_daily_game_ledger.csv"


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
    s = str(val).strip()
    if s == "":
        return True
    # Daily ledgers use `NR` as a canonical "reviewed missing" sentinel for
    # probability-like columns. Treat it as fillable for v2c/kalshi seeding.
    return s.upper() == "NR"


def _format_prob(val: Optional[float], decimals: int) -> str:
    from chimera_v2c.src.ledger.formatting import format_prob_cell

    if val is None:
        return ""
    return format_prob_cell(val, decimals=decimals, drop_leading_zero=True)


def _home_mid_from_plan(plan) -> Optional[float]:
    """
    Prefer the HOME-YES market mid; fall back to (1 - away_mid) if only away is present.
    """
    home = (plan.home or "").upper()
    away = (plan.away or "").upper()
    home_mid = None
    away_mid = None
    for s in plan.sides or []:
        yes_team = (s.yes_team or "").upper()
        mid = s.market.mid if s.market else None
        if yes_team == home and mid is not None:
            home_mid = float(mid)
        if yes_team == away and mid is not None:
            away_mid = float(mid)
    if home_mid is not None:
        return home_mid
    if away_mid is not None:
        return 1.0 - away_mid
    return None


def build_updates(cfg: V2CConfig, date_iso: str) -> List[Dict[str, str]]:
    target = datetime.strptime(date_iso, "%Y-%m-%d").date()
    plans = build_daily_plan(cfg, target)
    rows: List[Dict[str, str]] = []
    for p in plans:
        rows.append(
            {
                "date": date_iso,
                "league": cfg.league.lower(),
                "matchup": p.key,
                "v2c": _format_prob(float(p.p_final), 2),
                "kalshi_mid": _format_prob(_home_mid_from_plan(p), 2),
            }
        )
    return rows


def apply_updates(
    *,
    ledger_path: Path,
    league: str,
    updates: List[Dict[str, str]],
    apply: bool,
    force: bool,
) -> Tuple[int, int]:
    if not ledger_path.exists():
        raise SystemExit(f"[error] daily ledger missing: {ledger_path}")

    ymd = ledger_path.name.split("_")[0]
    locked = load_locked_dates(LOCK_DIR)
    if ymd in locked and not force:
        raise SystemExit(f"[error] {ymd} is locked; refusing to modify {ledger_path} (pass --force to override)")

    original = pd.read_csv(ledger_path, dtype=str, keep_default_na=False).fillna("")
    df = original.copy()
    for col in DEFAULT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    added_rows = 0
    filled_cells = 0
    for u in updates:
        if (u.get("league") or "").lower() != league.lower():
            continue
        matchup = (u.get("matchup") or "").strip()
        if not matchup:
            continue

        mask = (
            (df["date"].astype(str) == u.get("date", ""))
            & (df["league"].astype(str).str.lower() == league.lower())
            & (df["matchup"].astype(str) == matchup)
        )
        if not mask.any():
            new_row = {c: "" for c in df.columns}
            # Keep daily-ledger schema invariant: probability-like columns
            # should never be blank (use the `NR` sentinel instead).
            for prob_col in ("v2c", "grok", "gemini", "gpt", "kalshi_mid", "market_proxy", "moneypuck"):
                if prob_col in new_row:
                    new_row[prob_col] = "NR"
            new_row.update(
                {
                    "date": u.get("date", ""),
                    "league": league.lower(),
                    "matchup": matchup,
                }
            )
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            added_rows += 1
            mask = (
                (df["date"].astype(str) == u.get("date", ""))
                & (df["league"].astype(str).str.lower() == league.lower())
                & (df["matchup"].astype(str) == matchup)
            )

        idx = df.index[mask][0]

        for field in ("v2c", "kalshi_mid"):
            new_val = (u.get(field) or "").strip()
            if not new_val:
                continue
            existing = df.at[idx, field] if field in df.columns else ""
            if not _cell_blank(existing):
                continue
            df.at[idx, field] = new_val
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
            blank_sentinels={"NR"},
        )
    except LedgerGuardError as exc:
        raise SystemExit(f"[error] append-only guard failed: {exc}") from exc

    df.to_csv(ledger_path, index=False)
    return added_rows, filled_cells


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill daily ledger v2c + kalshi_mid from v2c planner output (append-only).")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD.")
    ap.add_argument("--config", default="chimera_v2c/config/defaults.yaml", help="League config yaml (default: NBA defaults).")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run).")
    ap.add_argument("--force", action="store_true", help="Allow edits to locked ledgers (still fills blanks only).")
    args = ap.parse_args()

    date_iso = datetime.strptime(args.date, "%Y-%m-%d").date().isoformat()
    cfg = V2CConfig.load(args.config)

    ensure_daily_ledger(date_iso)
    ledger_path = _ledger_path(date_iso)

    updates = build_updates(cfg, date_iso)
    added, filled = apply_updates(
        ledger_path=ledger_path,
        league=cfg.league.lower(),
        updates=updates,
        apply=bool(args.apply),
        force=bool(args.force),
    )

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] {ledger_path} rows_added={added} cells_filled={filled}")


if __name__ == "__main__":
    main()
