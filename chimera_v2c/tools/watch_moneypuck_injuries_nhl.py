#!/usr/bin/env python
"""
Watch MoneyPuck NHL injuries for changes (log-only notify).

Behavior:
- Polls MoneyPuck's public `current_injuries.csv` on an interval.
- Updates canonical files under `chimera_v2c/data/` via update_moneypuck_injuries_nhl.py.
- When changes are detected, writes a diff JSON under:
    reports/alerts/moneypuck_injuries/
  and prints a short alert to stdout.

This tool does NOT auto-run the planner or place trades (log-only).
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from chimera_v2c.tools.update_moneypuck_injuries_nhl import update_moneypuck_injuries_nhl


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch MoneyPuck NHL injuries for changes (log-only).")
    ap.add_argument("--date", help="Optional YYYY-MM-DD; if set, writes a slate-filtered digest for the date.")
    ap.add_argument("--interval-minutes", type=int, default=30, help="Polling interval in minutes (default: 30).")
    ap.add_argument("--once", action="store_true", help="Run once and exit (no loop).")
    args = ap.parse_args()

    interval_s = max(10, int(args.interval_minutes * 60))
    while True:
        try:
            res = update_moneypuck_injuries_nhl(date_iso=args.date, write_digest=bool(args.date), force=False)
        except Exception as exc:
            print(f"[warn] {utc_now()} MoneyPuck injuries poll failed: {exc}")
            res = None

        if res and bool(res.get("changed")):
            msg = f"[alert] {utc_now()} MoneyPuck injuries changed (sha={res.get('new_sha256')})"
            if res.get("diff_path"):
                msg += f" diff={res.get('diff_path')}"
            print(msg)
        else:
            print(f"[info] {utc_now()} MoneyPuck injuries unchanged")

        if args.once:
            return
        time.sleep(interval_s)


if __name__ == "__main__":
    main()

