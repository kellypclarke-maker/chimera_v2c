#!/usr/bin/env python
"""
Append Kalshi fills (exported CSV) into a durable Rule-A execution log (read-only wrt ledgers).

Why:
  - Operators often trade Rule A manually.
  - `export_kalshi_fills.py` exports account fills for a date window.
  - This tool consolidates those fills into a single append-only log so graders can
    reconcile plans without passing per-day fills CSV paths around.

Inputs:
  - A CSV produced by `export_kalshi_fills.py` (ideally with date_local + tz columns).

Output (default):
  - reports/execution_logs/rule_a_execution_log.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_OUT = Path("reports/execution_logs/rule_a_execution_log.csv")


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("[error] no rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _dedupe_key(r: Dict[str, str]) -> Tuple[str, str, str, str, str, str, str]:
    trade_id = str(r.get("trade_id") or "").strip()
    if trade_id:
        return (trade_id, "", "", "", "", "", "")
    return (
        "",
        str(r.get("created_time_utc") or "").strip(),
        str(r.get("ticker") or "").strip().upper(),
        str(r.get("action") or "").strip().lower(),
        str(r.get("side") or "").strip().lower(),
        str(r.get("count") or "").strip(),
        str(r.get("price_cents") or "").strip(),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Append Rule-A execution log rows from an export_kalshi_fills.py CSV.")
    ap.add_argument("--fills-csv", required=True, help="Path to a kalshi_fills_*.csv file from export_kalshi_fills.py.")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help=f"Output CSV path (default: {DEFAULT_OUT}).")
    ap.add_argument("--date", default="", help="Optional YYYY-MM-DD filter (matches date_local if present else date_utc).")
    ap.add_argument("--series-prefix", default="", help="Optional ticker prefix filter (e.g. KXNHLGAME).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    fills_path = Path(str(args.fills_csv))
    if not fills_path.exists():
        raise SystemExit(f"[error] missing fills csv: {fills_path}")

    out_path = Path(str(args.out))
    date_filter = str(args.date or "").strip()
    series_prefix = str(args.series_prefix or "").strip().upper()

    fills = _read_rows(fills_path)
    if not fills:
        raise SystemExit("[error] fills csv is empty")

    to_add: List[Dict[str, object]] = []
    for r in fills:
        date_key = str(r.get("date_local") or r.get("date_utc") or "").strip()
        if date_filter and date_key and date_key != date_filter:
            continue
        ticker = str(r.get("ticker") or "").strip().upper()
        if series_prefix and not ticker.startswith(series_prefix):
            continue
        to_add.append({**r, "source_fills_csv": str(fills_path)})

    if not to_add:
        raise SystemExit("[error] no rows matched filters")

    existing: List[Dict[str, object]] = []
    existing_keys: set[Tuple[str, str, str, str, str, str, str]] = set()
    index_by_key: Dict[Tuple[str, str, str, str, str, str, str], int] = {}
    if out_path.exists():
        existing = _read_rows(out_path)  # type: ignore[assignment]
        for i, r in enumerate(existing):  # type: ignore[arg-type]
            k = _dedupe_key(r)  # type: ignore[arg-type]
            existing_keys.add(k)
            index_by_key[k] = i

    added = 0
    for r in to_add:
        k = _dedupe_key({k: str(v) for k, v in r.items()})
        if k in existing_keys:
            # Upsert: fill missing fields if the new row has them.
            idx = index_by_key.get(k)
            if idx is not None:
                cur = existing[idx]
                changed = False
                for field, val in r.items():
                    cur_val = str(cur.get(field) or "").strip()
                    new_val = str(val or "").strip()
                    if not cur_val and new_val:
                        cur[field] = val
                        changed = True
                if changed:
                    existing[idx] = cur
            continue
        existing.append(r)
        existing_keys.add(k)
        index_by_key[k] = len(existing) - 1
        added += 1

    # Stable sort for audits.
    existing.sort(key=lambda r: (str(r.get("created_time_utc") or ""), str(r.get("ticker") or "")))

    _write_rows(out_path, existing)
    print(f"[ok] appended {added} fills -> {out_path} (total rows={len(existing)})")


if __name__ == "__main__":
    main()
