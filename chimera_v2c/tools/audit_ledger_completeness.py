#!/usr/bin/env python
"""
Audit daily-ledger completeness (blanks) and optionally mark confirmed-missing cells.

This tool is meant to help operators find which games/columns are still blank in:
  reports/daily_ledgers/*_daily_game_ledger.csv

It can also (optionally) mark *confirmed* missing values with a sentinel like "NR"
so future audits don't treat them as "unreviewed blanks".

Safety:
- Never overwrites non-blank cells.
- Respects lockfiles under reports/daily_ledgers/locked unless --force.
- Uses ledger guard append-only checks before writing.

Suggested workflow:
1) Run audit to generate missing-cell CSVs.
2) Manually verify whether a missing specialist report truly doesn't exist.
3) Create a "mark list" CSV (subset of the missing-cells file) and apply.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from chimera_v2c.src.ledger.guard import compute_append_only_diff, load_locked_dates


DAILY_DIR = Path("reports/daily_ledgers")
LOCK_DIR = DAILY_DIR / "locked"
DEFAULT_OUT_DIR = Path("reports/thesis_summaries")

DEFAULT_FIELDS = ("v2c", "gemini", "grok", "gpt", "kalshi_mid", "market_proxy", "moneypuck")


def _parse_date(date_str: str) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(date_str)
    except ValueError:
        return None


def _parse_ledger_date_from_filename(path: Path) -> Optional[dt.date]:
    name = path.name
    if not name.endswith("_daily_game_ledger.csv"):
        return None
    token = name.split("_", 1)[0]
    if len(token) != 8 or not token.isdigit():
        return None
    try:
        return dt.datetime.strptime(token, "%Y%m%d").date()
    except ValueError:
        return None


def _clean_cell(value: object) -> str:
    s = str(value).strip() if value is not None else ""
    if not s:
        return ""
    if s.lower() in {"nan", "none", "na", "n/a"}:
        return ""
    return s


def _within(d: dt.date, start: Optional[dt.date], end: Optional[dt.date]) -> bool:
    if start and d < start:
        return False
    if end and d > end:
        return False
    return True


def load_daily_frames(
    start: Optional[dt.date],
    end: Optional[dt.date],
    fields: Sequence[str],
) -> List[Tuple[Path, dt.date, pd.DataFrame]]:
    if not DAILY_DIR.exists():
        raise SystemExit(f"[error] missing daily ledger dir: {DAILY_DIR}")

    frames: List[Tuple[Path, dt.date, pd.DataFrame]] = []
    for path in sorted(DAILY_DIR.glob("*_daily_game_ledger.csv")):
        file_date = _parse_ledger_date_from_filename(path)
        if file_date is None or not _within(file_date, start, end):
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        for col in ("date", "league", "matchup"):
            if col not in df.columns:
                raise SystemExit(f"[error] ledger missing column '{col}': {path}")
        for field in fields:
            if field not in df.columns:
                df[field] = ""
        frames.append((path, file_date, df))
    return frames


def compute_blanks(
    frames: Iterable[Tuple[Path, dt.date, pd.DataFrame]],
    fields: Sequence[str],
    sentinel: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sentinel_norm = sentinel.strip().lower()

    all_games_rows: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []
    for _, file_date, df in frames:
        for _, r in df.iterrows():
            date = _clean_cell(r.get("date", "")) or file_date.isoformat()
            league = _clean_cell(r.get("league", "")).lower()
            matchup = _clean_cell(r.get("matchup", "")).upper()
            if not date or not league or not matchup:
                continue
            rec: Dict[str, object] = {"date": date, "league": league, "matchup": matchup}
            for f in fields:
                rec[f] = _clean_cell(r.get(f, ""))
            all_games_rows.append(rec)
            missing: List[str] = []
            for f in fields:
                val = _clean_cell(r.get(f, ""))
                if not val:
                    missing.append(f)
                    continue
                if sentinel_norm and val.strip().lower() == sentinel_norm:
                    # Explicitly reviewed missing; do not count as a blank.
                    continue
            if missing:
                rows.append(
                    {
                        "date": date,
                        "league": league,
                        "matchup": matchup,
                        "missing_fields": ",".join(missing),
                        "missing_count": len(missing),
                    }
                )

    blanks_games = pd.DataFrame(rows).sort_values(["date", "league", "matchup"]) if rows else pd.DataFrame(
        columns=["date", "league", "matchup", "missing_fields", "missing_count"]
    )

    # Long-form missing-cell list (one row per missing field)
    cells: List[Dict[str, str]] = []
    for _, row in blanks_games.iterrows():
        for f in str(row["missing_fields"]).split(","):
            if not f:
                continue
            cells.append({"date": row["date"], "league": row["league"], "matchup": row["matchup"], "field": f})
    blanks_cells = pd.DataFrame(cells).sort_values(["date", "league", "matchup", "field"]) if cells else pd.DataFrame(
        columns=["date", "league", "matchup", "field"]
    )

    # Summary counts per field
    total_rows = 0
    blank_counts = {f: 0 for f in fields}
    marked_counts = {f: 0 for f in fields}
    for _, _, df in frames:
        total_rows += len(df)
        for f in fields:
            series = df[f].apply(_clean_cell)
            blank_counts[f] += int((series == "").sum())
            if sentinel_norm:
                marked_counts[f] += int((series.str.lower() == sentinel_norm).sum())
    summary_rows = []
    for f in fields:
        blank = blank_counts[f]
        marked = marked_counts[f]
        filled = max(0, total_rows - blank - marked)
        summary_rows.append(
            {
                "field": f,
                "total_rows": total_rows,
                "filled": filled,
                "blank": blank,
                "marked": marked,
                "sentinel": sentinel,
            }
        )
    summary = pd.DataFrame(summary_rows)
    all_games = (
        pd.DataFrame(all_games_rows).sort_values(["date", "league", "matchup"])
        if all_games_rows
        else pd.DataFrame(columns=["date", "league", "matchup", *fields])
    )
    return summary, blanks_games, blanks_cells, all_games


def apply_marks(
    marks: pd.DataFrame,
    *,
    sentinel: str,
    force: bool,
) -> Tuple[int, List[Path]]:
    locked = load_locked_dates(LOCK_DIR)
    updated_cells = 0
    touched: List[Path] = []

    required_cols = {"date", "league", "matchup", "field"}
    if not required_cols.issubset(set(marks.columns)):
        raise SystemExit(f"[error] mark file missing columns: {sorted(required_cols - set(marks.columns))}")

    for (date, _), group in marks.groupby(["date", "matchup"], sort=True):
        date_str = str(date).strip()
        d = _parse_date(date_str)
        if d is None:
            continue
        ymd = d.strftime("%Y%m%d")
        ledger_path = DAILY_DIR / f"{ymd}_daily_game_ledger.csv"
        if not ledger_path.exists():
            continue
        if ymd in locked and not force:
            continue

        df = pd.read_csv(ledger_path, dtype=str).fillna("")
        for col in ("date", "league", "matchup"):
            if col not in df.columns:
                raise SystemExit(f"[error] ledger missing column '{col}': {ledger_path}")

        # We validate append-only by diffing full rows as strings.
        old_rows = df.fillna("").astype(str).to_dict("records")

        for _, mark in group.iterrows():
            league = str(mark["league"]).strip().lower()
            matchup = str(mark["matchup"]).strip().upper()
            field = str(mark["field"]).strip()
            if field not in df.columns:
                continue

            mask = (
                (df["date"].astype(str).str.strip() == date_str)
                & (df["league"].astype(str).str.strip().str.lower() == league)
                & (df["matchup"].astype(str).str.strip().str.upper() == matchup)
            )
            if not mask.any():
                continue
            idx = df.index[mask][0]
            existing = _clean_cell(df.at[idx, field])
            if existing:
                continue
            df.at[idx, field] = sentinel
            updated_cells += 1

        new_rows = df.fillna("").astype(str).to_dict("records")
        compute_append_only_diff(
            old_rows=old_rows,
            new_rows=new_rows,
            key_fields=["date", "league", "matchup"],
            value_fields=[c for c in df.columns if c not in ("date", "league", "matchup")],
        )
        df.to_csv(ledger_path, index=False)
        touched.append(ledger_path)

    return updated_cells, touched


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit daily-ledger blanks and optionally mark confirmed missing cells.")
    ap.add_argument("--start-date", help="ISO start date filter (YYYY-MM-DD).")
    ap.add_argument("--end-date", help="ISO end date filter (YYYY-MM-DD).")
    ap.add_argument(
        "--fields",
        default=",".join(DEFAULT_FIELDS),
        help=f"Comma-separated fields to audit (default: {','.join(DEFAULT_FIELDS)}).",
    )
    ap.add_argument(
        "--sentinel",
        default="NR",
        help="Sentinel value for 'reviewed missing' (default: NR). Avoid NA/NaN since they parse as blank.",
    )
    ap.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory for reports (default: {DEFAULT_OUT_DIR}).",
    )
    ap.add_argument(
        "--mark-from",
        help="CSV path with columns date,league,matchup,field to mark as sentinel (use a subset of blanks_cells CSV).",
    )
    ap.add_argument("--apply", action="store_true", help="Apply marks to ledgers (default: dry-run).")
    ap.add_argument("--force", action="store_true", help="Allow edits to locked daily ledgers.")
    args = ap.parse_args()

    start = _parse_date(args.start_date) if args.start_date else None
    end = _parse_date(args.end_date) if args.end_date else None
    fields = [f.strip() for f in str(args.fields).split(",") if f.strip()]
    out_dir = Path(args.out_dir)

    frames = load_daily_frames(start, end, fields)
    if not frames:
        raise SystemExit("[error] no daily ledgers found in the requested date range.")

    # Determine report token range from filenames.
    file_dates = [d for _, d, _ in frames]
    token_start = (start or min(file_dates)).strftime("%Y%m%d")
    token_end = (end or max(file_dates)).strftime("%Y%m%d")

    summary, blanks_games, blanks_cells, all_games = compute_blanks(frames, fields, args.sentinel)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"ledger_blanks_summary_{token_start}_{token_end}.csv"
    games_path = out_dir / f"ledger_blanks_games_{token_start}_{token_end}.csv"
    cells_path = out_dir / f"ledger_blanks_cells_{token_start}_{token_end}.csv"
    all_games_path = out_dir / f"ledger_games_{token_start}_{token_end}.csv"
    summary.to_csv(summary_path, index=False)
    blanks_games.to_csv(games_path, index=False)
    blanks_cells.to_csv(cells_path, index=False)
    all_games.to_csv(all_games_path, index=False)

    print(f"[ok] wrote {summary_path}")
    print(f"[ok] wrote {games_path}")
    print(f"[ok] wrote {cells_path}")
    print(f"[ok] wrote {all_games_path}")

    total_games = int(summary["total_rows"].iloc[0]) if not summary.empty else 0
    print(f"\n=== Completeness ({token_start}..{token_end}) ===")
    print(f"games={total_games} games_with_any_blank={len(blanks_games)}")
    for _, row in summary.iterrows():
        print(
            f"  {row['field']}: blank={int(row['blank'])} marked={int(row['marked'])} filled={int(row['filled'])}"
        )

    if args.mark_from:
        mark_path = Path(args.mark_from)
        marks = pd.read_csv(mark_path, dtype=str).fillna("")
        if args.apply:
            updated, touched = apply_marks(marks, sentinel=args.sentinel, force=args.force)
            print(f"\n[apply] updated_cells={updated} ledgers_touched={len(touched)}")
            for p in touched:
                print(f"  - {p}")
        else:
            print("\n[dry-run] --mark-from provided; add --apply to write sentinel marks.")


if __name__ == "__main__":
    main()
