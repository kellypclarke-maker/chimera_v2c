"""
Write one append-safe ledger per game day with a simplified schema.

Schema (one row per game):
  - date
  - league
  - matchup (AWAY@HOME)
  - v2c (home win prob; calibrated if calibration enabled at plan time)
  - gemini (home win prob)
  - grok (home win prob)
  - gpt (home win prob)
  - kalshi_mid (home-implied prob from the mid)
  - market_proxy (optional no-vig sportsbook implied prob)
  - moneypuck (optional NHL baseline win prob)
  - actual_outcome (final score if available, otherwise blank)

Defaults:
  - Source: reports/specialist_performance/game_level_ml_master.csv (read-only)
  - Output: reports/daily_ledgers/YYYYMMDD_daily_game_ledger.csv
  - Refuses to overwrite existing daily files unless --overwrite is passed.

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/build_daily_game_ledgers.py --date YYYY-MM-DD
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from chimera_v2c.src.ledger.formatting import DAILY_LEDGER_COLUMNS, MISSING_SENTINEL, format_prob_cell
from chimera_v2c.src.ledger.outcomes import format_final_score


DEFAULT_SOURCE = Path("reports/specialist_performance/game_level_ml_master.csv")
DEFAULT_OUT_DIR = Path("reports/daily_ledgers")

OUT_COLUMNS = DAILY_LEDGER_COLUMNS

REQUIRED_SOURCE_COLS = {
    "date",
    "league",
    "game_id",
    "p_home_v2c",
    "p_home_gemini",
    "p_home_grok",
    "p_home_gpt",
    "p_home_market",
}


def parse_float(val: object) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_home_win(val: object) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none"}:
        return None
    lower = s.lower()
    if lower in {"true", "t", "yes", "y"}:
        return 1.0
    if lower in {"false", "f", "no", "n"}:
        return 0.0
    try:
        return float(s)
    except Exception:
        return None


def clean_prob(val: object) -> str:
    out = format_prob_cell(val, decimals=2, drop_leading_zero=True)
    return out or MISSING_SENTINEL


def format_outcome(row: Dict[str, str]) -> str:
    home_team = (row.get("home_team") or "").strip()
    away_team = (row.get("away_team") or "").strip()
    home_score = parse_float(row.get("home_score"))
    away_score = parse_float(row.get("away_score"))
    if home_score is None or away_score is None:
        return ""
    if not home_team or not away_team:
        return ""
    return format_final_score(away_team, home_team, away_score, home_score)


def to_daily_row(row: Dict[str, str]) -> Dict[str, str]:
    league = (row.get("league") or "").strip().lower()
    return {
        "date": (row.get("date") or "").strip(),
        "league": league,
        "matchup": (row.get("game_id") or "").strip(),
        "v2c": clean_prob(row.get("p_home_v2c")),
        "gemini": clean_prob(row.get("p_home_gemini")),
        "grok": clean_prob(row.get("p_home_grok")),
        "gpt": clean_prob(row.get("p_home_gpt")),
        "kalshi_mid": clean_prob(row.get("p_home_market")),
        "market_proxy": MISSING_SENTINEL,
        "moneypuck": MISSING_SENTINEL,
        "actual_outcome": format_outcome(row),
    }


def build_daily_ledgers(
    source: Path,
    out_dir: Path,
    dates: Optional[Set[str]] = None,
    overwrite: bool = False,
    allow_empty: bool = False,
    force: bool = False,
    snapshot_dir: Optional[Path] = Path("reports/specialist_performance/snapshots"),
) -> None:
    if not source.exists():
        raise SystemExit(f"[error] source ledger not found: {source}")

    with source.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_SOURCE_COLS - fieldnames)
        if missing:
            raise SystemExit(f"[error] missing required columns in source: {missing}")

        grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in reader:
            date = (row.get("date") or "").strip()
            if not date:
                continue
            if dates and date not in dates:
                continue
            grouped[date].append(to_daily_row(row))

    target_dates: Set[str] = set(grouped.keys())
    if dates:
        target_dates = set(dates)
        missing_dates = target_dates - set(grouped.keys())
        if missing_dates and not allow_empty:
            raise SystemExit(f"[error] no rows found for dates: {sorted(missing_dates)}")
        for d in missing_dates:
            grouped[d] = []
    elif not grouped and not allow_empty:
        raise SystemExit("[error] no rows found in source; nothing to write")

    out_dir.mkdir(parents=True, exist_ok=True)
    for date in sorted(target_dates):
        rows = grouped.get(date, [])
        ymd = date.replace("-", "")
        out_path = out_dir / f"{ymd}_daily_game_ledger.csv"
        lock_path = out_path.with_suffix(out_path.suffix + ".lock")
        if lock_path.exists():
            raise SystemExit(f"[error] lock present for {out_path}; another process may be writing. Delete the lock to proceed intentionally.")
        if out_path.exists() and not overwrite:
            print(f"[skip] {out_path} already exists (use --overwrite --force to replace)")
            continue
        if out_path.exists() and overwrite and not force:
            raise SystemExit(f"[error] refusing to overwrite {out_path} without --force")

        # Acquire lock
        lock_path.write_text("locked", encoding="utf-8")

        # Snapshot existing file before overwrite
        if out_path.exists() and snapshot_dir:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            snap_path = snapshot_dir / f"{out_path.name}.{ts}.bak"
            snap_path.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")

        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[ok] wrote {len(rows)} rows -> {out_path}")
        try:
            lock_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


def parse_dates(date_values: Iterable[str]) -> Set[str]:
    return {d.strip() for d in date_values if d and d.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Write per-day ledgers with a simplified schema to avoid overwriting historical rows."
    )
    ap.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Source ledger (default: reports/specialist_performance/game_level_ml_master.csv)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for daily ledgers",
    )
    ap.add_argument(
        "--date",
        action="append",
        dest="dates",
        help="Limit to one or more specific dates (YYYY-MM-DD). If omitted, all dates are emitted.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing daily file (default: refuse and skip).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Must be set with --overwrite to actually replace an existing file.",
    )
    ap.add_argument(
        "--snapshot-dir",
        default="reports/specialist_performance/snapshots",
        help="Directory to save snapshots before overwrite (default: reports/specialist_performance/snapshots). Empty to disable.",
    )
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write header-only files if a requested date has no rows in the source.",
    )
    args = ap.parse_args()

    dates = parse_dates(args.dates or [])
    build_daily_ledgers(
        source=Path(args.source),
        out_dir=Path(args.out_dir),
        dates=dates or None,
        overwrite=args.overwrite,
        allow_empty=args.allow_empty,
        force=args.force,
        snapshot_dir=Path(args.snapshot_dir) if args.snapshot_dir else None,
    )


if __name__ == "__main__":
    main()
