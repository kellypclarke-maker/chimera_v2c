#!/usr/bin/env python
"""
Derive and evaluate a mid-bucket filter for Rule A from the bid/ask suite outputs (read-only).

Given `ruleA_bidask_suite_mid_buckets_*.csv`, this tool:
  - Chooses "allowed" buckets based on OVERALL net_pnl >= 0 for a chosen anchor/slippage
  - Produces a policy summary comparing ALL buckets vs FILTERED buckets, by league and overall

It does NOT modify any ledgers.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/build_rule_a_bucket_filter.py \
    --suite-mid-buckets reports/thesis_summaries/ruleA_bidask_suite_mid_buckets_20251119_20251215.csv \
    --anchor-minutes 30 --slippage-cents 1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


def _write_rows(out_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _roi(net_pnl: float, risked: float) -> float:
    return 0.0 if float(risked) <= 0 else float(net_pnl) / float(risked)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a Rule A mid-bucket filter policy from suite outputs (read-only).")
    ap.add_argument("--suite-mid-buckets", required=True, help="Path to ruleA_bidask_suite_mid_buckets_*.csv")
    ap.add_argument("--anchor-minutes", type=int, default=30, help="Anchor minutes to use (default: 30).")
    ap.add_argument("--slippage-cents", type=int, default=1, help="Slippage cents to use (default: 1).")
    ap.add_argument("--out", help="Optional output CSV path (default under reports/thesis_summaries).")
    args = ap.parse_args()

    df = pd.read_csv(args.suite_mid_buckets)
    need = {"window_start", "window_end", "anchor_minutes", "slippage_cents", "league", "mid_bucket", "bets", "units", "risked", "fees", "net_pnl"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise SystemExit(f"[error] suite file missing columns: {missing}")

    df = df[(df["anchor_minutes"] == int(args.anchor_minutes)) & (df["slippage_cents"] == int(args.slippage_cents))].copy()
    if df.empty:
        raise SystemExit("[error] no rows for requested anchor/slippage")

    # Allowed buckets are chosen by overall (net_pnl >= 0). This yields a single global filter.
    ov = df[df["league"] == "overall"].copy()
    if ov.empty:
        raise SystemExit("[error] suite file has no league='overall' rows; rerun validate_rule_a_bidask_suite.py")
    allowed = sorted(ov[ov["net_pnl"] >= 0]["mid_bucket"].astype(str).unique().tolist())
    blocked = sorted(ov[ov["net_pnl"] < 0]["mid_bucket"].astype(str).unique().tolist())

    leagues = sorted(df["league"].astype(str).unique().tolist())
    if "overall" not in leagues:
        leagues = ["overall"] + leagues

    rows: List[Dict[str, object]] = []
    for league in leagues:
        sub = df[df["league"] == league]
        if sub.empty:
            continue
        all_tot = sub[["bets", "units", "risked", "fees", "net_pnl"]].sum(numeric_only=True)
        keep = sub[sub["mid_bucket"].astype(str).isin(allowed)][["bets", "units", "risked", "fees", "net_pnl"]].sum(numeric_only=True)

        for label, tot in (("all_buckets", all_tot), ("filtered_buckets", keep)):
            rows.append(
                {
                    "window_start": str(sub["window_start"].iloc[0]),
                    "window_end": str(sub["window_end"].iloc[0]),
                    "anchor_minutes": int(args.anchor_minutes),
                    "slippage_cents": int(args.slippage_cents),
                    "league": league,
                    "policy": label,
                    "allowed_buckets": "|".join(allowed),
                    "blocked_buckets": "|".join(blocked),
                    "bets": int(tot["bets"]),
                    "units": int(tot["units"]),
                    "risked": round(float(tot["risked"]), 6),
                    "fees": round(float(tot["fees"]), 6),
                    "net_pnl": round(float(tot["net_pnl"]), 6),
                    "roi_net": round(_roi(float(tot["net_pnl"]), float(tot["risked"])), 6),
                }
            )

        removed_bets = int(all_tot["bets"] - keep["bets"])
        removed_units = int(all_tot["units"] - keep["units"])
        rows.append(
            {
                "window_start": str(sub["window_start"].iloc[0]),
                "window_end": str(sub["window_end"].iloc[0]),
                "anchor_minutes": int(args.anchor_minutes),
                "slippage_cents": int(args.slippage_cents),
                "league": league,
                "policy": "removed_fraction",
                "allowed_buckets": "|".join(allowed),
                "blocked_buckets": "|".join(blocked),
                "bets": removed_bets,
                "units": removed_units,
                "risked": "",
                "fees": "",
                "net_pnl": "",
                "roi_net": "",
            }
        )

    out_path = Path(args.out) if args.out else Path(
        f"reports/thesis_summaries/ruleA_bucket_filter_policy_{str(rows[0]['window_start']).replace('-', '')}_{str(rows[0]['window_end']).replace('-', '')}_m{int(args.anchor_minutes)}_s{int(args.slippage_cents)}.csv"
    )
    _write_rows(out_path, rows)
    print(f"[ok] wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

