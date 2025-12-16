"""
EV vs Kalshi mid and calibration from daily ledgers (non-destructive).

Usage examples (from repo root):
  # Last 30 ledger days, all leagues
  PYTHONPATH=. python chimera_v2c/tools/analyze_ev_vs_kalshi.py --days 30

  # Explicit date range and league filter
  PYTHONPATH=. python chimera_v2c/tools/analyze_ev_vs_kalshi.py \\
      --start-date 2025-12-04 --end-date 2025-12-09 --league nhl

Reads `reports/daily_ledgers/*_daily_game_ledger.csv` and computes:
  - Realized EV vs Kalshi mid for each model (1 unit per game).
  - Brier scores vs actual outcomes.
  - Bucketed EV by absolute edge |p_model - p_kalshi|.

This tool is read-only and must never write to the ledgers.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from chimera_v2c.src.ledger_analysis import (
    LEDGER_DIR,
    compute_brier,
    compute_bucketed_ev_vs_kalshi,
    compute_ev_vs_kalshi,
    load_games,
)


DEFAULT_MODELS = ["v2c", "gemini", "grok", "gpt"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Analyze EV vs Kalshi mid and Brier scores from daily_ledgers (read-only)."
    )
    ap.add_argument(
        "--days",
        type=int,
        default=30,
        help="Include only the most recent N ledger days by filename date (default: 30). "
        "Use 0 or a negative value to include all days.",
    )
    ap.add_argument(
        "--start-date",
        help="Optional start date (YYYY-MM-DD). If provided, overrides --days.",
    )
    ap.add_argument(
        "--end-date",
        help="Optional end date (YYYY-MM-DD). If provided, overrides --days.",
    )
    ap.add_argument(
        "--league",
        help="Optional league filter (nba|nhl|nfl). Case-insensitive.",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model columns to analyze (default: v2c gemini grok gpt). "
        "kalshi_mid is always used as the market baseline.",
    )
    ap.add_argument(
        "--bucket-width",
        type=float,
        default=0.025,
        help="Absolute edge bucket width for EV breakdown (default: 0.025).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not LEDGER_DIR.exists():
        raise SystemExit(f"[error] daily ledger directory not found: {LEDGER_DIR}")

    # Use start/end-date if provided; otherwise fall back to days-based window.
    days = None if (args.start_date or args.end_date) else args.days
    if days is not None and days <= 0:
        days = None

    models: List[str] = list(args.models)

    games = load_games(
        daily_dir=LEDGER_DIR,
        days=days,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=args.league,
        models=models + ["kalshi_mid"],
    )
    if not games:
        raise SystemExit("[error] no games found for given filters")

    print(
        f"[info] loaded {len(games)} game rows from {LEDGER_DIR} "
        f"(league={args.league or 'all'}, models={','.join(models)}, "
        f"window={'custom' if (args.start_date or args.end_date) else (days or 'all')} days)"
    )

    # Overall EV vs Kalshi mid per model.
    ev_stats = compute_ev_vs_kalshi(games, models=models)
    brier_stats = compute_brier(games, models=models + ["kalshi_mid"])

    print("\n=== Overall EV vs Kalshi mid (1 unit per game) ===")
    print(f"{'model':10s} {'bets':>6s} {'total_pnl':>12s} {'avg_pnl_per_bet':>17s}")
    for m in models:
        s = ev_stats.get(m)
        if s is None:
            continue
        print(f"{m:10s} {s.bets:6d} {s.total_pnl:12.3f} {s.avg_pnl:17.4f}")

    print("\n=== Brier vs outcomes (lower is better) ===")
    print(f"{'model':10s} {'brier':>8s} {'n_games':>10s}")
    for m in models + ["kalshi_mid"]:
        s = brier_stats.get(m)
        if s is None or s.mean_brier is None:
            continue
        print(f"{m:10s} {s.mean_brier:8.3f} {s.n:10d}")

    # Bucketed EV vs Kalshi mid by absolute edge.
    bucketed = compute_bucketed_ev_vs_kalshi(games, models=models, bucket_width=args.bucket_width)

    print("\n=== Bucketed EV vs Kalshi mid by |p_model - p_kalshi| ===")
    print(f"(bucket width = {args.bucket_width:.3f})")
    for m in models:
        model_buckets = bucketed.get(m) or {}
        if not model_buckets:
            continue
        print(f"\n-- {m} --")
        print(f"{'edge_bucket':15s} {'bets':>6s} {'total_pnl':>12s} {'avg_pnl_per_bet':>17s}")
        for bucket in sorted(model_buckets.keys()):
            s = model_buckets[bucket]
            print(f"{bucket:15s} {s.bets:6d} {s.total_pnl:12.3f} {s.avg_pnl:17.4f}")


if __name__ == "__main__":
    main()

