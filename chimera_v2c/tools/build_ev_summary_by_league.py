"""
Build EV + Brier summary tables from daily ledgers (read-only on ledgers).

Writes a single CSV with per-(league,model) summary stats:
  - Realized EV vs Kalshi mid (1 unit per game) for each model.
  - Brier score vs outcomes for each model and for kalshi_mid itself.

This is intended as a durable "EV file" artifact under reports/, regenerated
whenever daily ledgers are repaired/updated.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, compute_brier, compute_ev_vs_kalshi, load_games


DEFAULT_MODELS = ["v2c", "gemini", "grok", "gpt", "market_proxy", "moneypuck"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build EV vs Kalshi + Brier summary by league from daily ledgers (read-only on ledgers)."
    )
    ap.add_argument(
        "--days",
        type=int,
        default=30,
        help="Include only the most recent N ledger days by filename date (default: 30). "
        "Use 0 or a negative value to include all days.",
    )
    ap.add_argument("--start-date", help="Optional start date (YYYY-MM-DD). If set, overrides --days.")
    ap.add_argument("--end-date", help="Optional end date (YYYY-MM-DD). If set, overrides --days.")
    ap.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model columns to include (default: v2c gemini grok gpt market_proxy moneypuck).",
    )
    ap.add_argument(
        "--out",
        default="reports/daily_ledgers/ev_brier_by_league.csv",
        help="Output CSV path (default: reports/daily_ledgers/ev_brier_by_league.csv).",
    )
    return ap.parse_args()


def _fmt(x: Optional[float], digits: int = 6) -> str:
    if x is None:
        return ""
    return f"{x:.{digits}f}"


def main() -> None:
    args = parse_args()
    if not LEDGER_DIR.exists():
        raise SystemExit(f"[error] daily ledger directory not found: {LEDGER_DIR}")

    days: Optional[int]
    if args.start_date or args.end_date:
        days = None
    else:
        days = args.days
        if days <= 0:
            days = None

    models: List[str] = list(args.models)
    games = load_games(
        daily_dir=LEDGER_DIR,
        days=days,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=None,
        models=models + ["kalshi_mid"],
    )
    if not games:
        raise SystemExit("[error] no games loaded from daily ledgers")

    leagues = sorted({g.league for g in games})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "league",
        "model",
        "window_start",
        "window_end",
        "games_in_league",
        "bets",
        "total_pnl",
        "avg_pnl_per_bet",
        "brier",
        "brier_n",
    ]

    rows = []
    for league in leagues + ["overall"]:
        subset = games if league == "overall" else [g for g in games if g.league == league]
        if not subset:
            continue

        ev_stats = compute_ev_vs_kalshi(subset, models=models)
        brier_stats = compute_brier(subset, models=models + ["kalshi_mid"])

        for model in models:
            ev = ev_stats.get(model)
            br = brier_stats.get(model)
            rows.append(
                {
                    "league": league,
                    "model": model,
                    "window_start": args.start_date or "",
                    "window_end": args.end_date or "",
                    "games_in_league": len(subset),
                    "bets": "" if ev is None else ev.bets,
                    "total_pnl": "" if ev is None else _fmt(ev.total_pnl, 6),
                    "avg_pnl_per_bet": "" if ev is None else _fmt(ev.avg_pnl, 6),
                    "brier": "" if br is None else _fmt(br.mean_brier, 6),
                    "brier_n": "" if br is None else br.n,
                }
            )

        # Always include kalshi_mid Brier as a baseline row.
        br_mid = brier_stats.get("kalshi_mid")
        rows.append(
            {
                "league": league,
                "model": "kalshi_mid",
                "window_start": args.start_date or "",
                "window_end": args.end_date or "",
                "games_in_league": len(subset),
                "bets": "",
                "total_pnl": "",
                "avg_pnl_per_bet": "",
                "brier": "" if br_mid is None else _fmt(br_mid.mean_brier, 6),
                "brier_n": "" if br_mid is None else br_mid.n,
            }
        )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[info] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
