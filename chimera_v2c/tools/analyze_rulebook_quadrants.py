"""
Rulebook quadrant analysis (read-only): symmetric edge buckets vs Kalshi mid.

This tool generalizes Scheme D by evaluating 4 market regimes (home/away-fav)
and 2 trade directions (fade/follow), with optional sub-buckets based on
whether the model itself crosses 0.5 (prefers home vs away).

All probabilities are *home* win probabilities:
  - Market baseline: kalshi_mid = p_home(mid)
  - Model: p_model = p_home(model)

Edge threshold t (default 0.05):
  - "home rich"  vs market: p_model <= p_mid - t  (model less bullish on home)
  - "home cheap" vs market: p_model >= p_mid + t  (model more bullish on home)

Buckets (letters kept stable for ops):
  Market home-fav (p_mid >= 0.5)
    A: Fade home (bet away) where home rich (I + J)
      I: p_model < 0.5
      J: p_model >= 0.5
    B: Follow home (bet home) where home cheap (M + N)
      M: p_model >= 0.5
      N: p_model < 0.5

  Market away-fav (p_mid < 0.5)
    C: Follow away (bet away) where home rich (O + P)
      O: p_model < 0.5
      P: p_model >= 0.5
    D: Fade away (bet home) where home cheap (K + L)
      K: p_model >= 0.5
      L: p_model < 0.5

PnL convention: 1 unit, trade at the Kalshi mid price for the side bought.

Writes derived CSV artifacts under `reports/ev_rulebooks/` and never modifies
daily ledgers.

This tool also supports edge-threshold sweeps and can emit a machine-readable
"selected rulebook" (per league/model/bucket) based on the allow-gates
(min_bets + ev_threshold) and a selection mode.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, load_games
from chimera_v2c.src.rulebook_quadrants import (
    DEFAULT_BUCKETS,
    BucketStats,
    compute_bucket_stats,
    is_allowed_bucket,
    select_threshold_for_bucket,
)


DEFAULT_MODELS = ["v2c", "gemini", "grok", "gpt", "market_proxy", "moneypuck"]
DEFAULT_EDGE_THRESHOLDS = [0.02, 0.03, 0.05, 0.07]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Quadrant rulebook analysis from daily ledgers (read-only).",
    )
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--start-date", help="YYYY-MM-DD (inclusive); overrides --days.")
    ap.add_argument("--end-date", help="YYYY-MM-DD (inclusive); overrides --days.")
    ap.add_argument("--league", help="nba|nhl|nfl (optional)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--edge-threshold", type=float, default=0.05, help="Single edge threshold (default: 0.05).")
    ap.add_argument(
        "--edge-thresholds",
        nargs="+",
        type=float,
        help="Optional sweep list (e.g., 0.02 0.03 0.05 0.07). Overrides --edge-threshold.",
    )
    ap.add_argument(
        "--preset-thresholds",
        action="store_true",
        help=f"Shortcut for --edge-thresholds {' '.join(str(x) for x in DEFAULT_EDGE_THRESHOLDS)}.",
    )
    ap.add_argument("--buckets", nargs="+", default=DEFAULT_BUCKETS)
    ap.add_argument("--min-bets", type=int, default=1)
    ap.add_argument("--ev-threshold", type=float, default=0.10)
    ap.add_argument("--out-dir", default="reports/ev_rulebooks")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument(
        "--select-threshold-mode",
        choices=["min_edge", "max_avg_pnl", "max_total_pnl"],
        default="min_edge",
        help="When sweeping thresholds, pick one per (league,model,bucket) from eligible thresholds (default: min_edge).",
    )
    ap.add_argument(
        "--write-selected-rulebook",
        action="store_true",
        help="When sweeping thresholds, write a selected rulebook JSON and sweet-spot CSV under --out-dir.",
    )
    return ap.parse_args()


def print_table(
    stats: Dict[Tuple[str, str, str], BucketStats],
    min_bets: int,
    ev_threshold: float,
) -> None:
    rows = []
    for (league, model, bucket), s in stats.items():
        rows.append((league, model, bucket, s.bets, s.avg_pnl, s.win_rate, s.total_pnl))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    print("\n=== Quadrant bucket stats (trade at mid) ===")
    print(f"{'league':5s} {'model':11s} {'bucket':6s} {'bets':>4s} {'avg_pnl':>8s} {'win_rate':>8s} {'total_pnl':>9s} {'allowed':>7s}")
    for league, model, bucket, bets, avg_pnl, win_rate, total_pnl in rows:
        allowed = bets >= min_bets and avg_pnl >= ev_threshold
        print(
            f"{league:5s} {model:11s} {bucket:6s} {bets:4d} {avg_pnl:8.3f} {win_rate:8.3f} {total_pnl:9.3f} {str(allowed):>7s}"
        )


def write_csv(
    out_path: Path,
    stats: Dict[Tuple[str, str, str], BucketStats],
    start_date: Optional[str],
    end_date: Optional[str],
    edge_threshold: float,
    ev_threshold: float,
    min_bets: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "league",
        "model",
        "bucket",
        "bets",
        "wins",
        "win_rate",
        "total_pnl",
        "avg_pnl",
        "allowed",
        "start_date",
        "end_date",
        "edge_threshold",
        "ev_threshold",
        "min_bets",
    ]
    rows = []
    for (league, model, bucket), s in stats.items():
        rows.append(
            {
                "league": league,
                "model": model,
                "bucket": bucket,
                "bets": s.bets,
                "wins": s.wins,
                "win_rate": f"{s.win_rate:.6f}",
                "total_pnl": f"{s.total_pnl:.6f}",
                "avg_pnl": f"{s.avg_pnl:.6f}",
                "allowed": str(is_allowed_bucket(stats=s, min_bets=min_bets, ev_threshold=ev_threshold)),
                "start_date": start_date or "",
                "end_date": end_date or "",
                "edge_threshold": f"{edge_threshold:.3f}",
                "ev_threshold": f"{ev_threshold:.3f}",
                "min_bets": min_bets,
            }
        )
    rows.sort(key=lambda r: (r["league"], r["model"], r["bucket"]))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sweep_csv(
    out_path: Path,
    stats_grid: Dict[Tuple[float, str, str, str], BucketStats],
    start_date: Optional[str],
    end_date: Optional[str],
    ev_threshold: float,
    min_bets: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "league",
        "model",
        "bucket",
        "edge_threshold",
        "bets",
        "wins",
        "win_rate",
        "total_pnl",
        "avg_pnl",
        "allowed",
        "start_date",
        "end_date",
        "ev_threshold",
        "min_bets",
    ]

    rows = []
    for (edge_threshold, league, model, bucket), s in stats_grid.items():
        rows.append(
            {
                "league": league,
                "model": model,
                "bucket": bucket,
                "edge_threshold": f"{edge_threshold:.3f}",
                "bets": s.bets,
                "wins": s.wins,
                "win_rate": f"{s.win_rate:.6f}",
                "total_pnl": f"{s.total_pnl:.6f}",
                "avg_pnl": f"{s.avg_pnl:.6f}",
                "allowed": str(is_allowed_bucket(stats=s, min_bets=min_bets, ev_threshold=ev_threshold)),
                "start_date": start_date or "",
                "end_date": end_date or "",
                "ev_threshold": f"{ev_threshold:.3f}",
                "min_bets": min_bets,
            }
        )
    rows.sort(key=lambda r: (r["league"], r["model"], r["bucket"], float(r["edge_threshold"])))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sweet_spots_csv(
    out_path: Path,
    selected: Dict[Tuple[str, str, str], Tuple[float, BucketStats]],
    start_date: Optional[str],
    end_date: Optional[str],
    selection_mode: str,
    ev_threshold: float,
    min_bets: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "league",
        "model",
        "bucket",
        "selected_edge_threshold",
        "bets",
        "wins",
        "win_rate",
        "total_pnl",
        "avg_pnl",
        "allowed",
        "start_date",
        "end_date",
        "selection_mode",
        "ev_threshold",
        "min_bets",
    ]
    rows = []
    for (league, model, bucket), (t, s) in selected.items():
        rows.append(
            {
                "league": league,
                "model": model,
                "bucket": bucket,
                "selected_edge_threshold": f"{t:.3f}",
                "bets": s.bets,
                "wins": s.wins,
                "win_rate": f"{s.win_rate:.6f}",
                "total_pnl": f"{s.total_pnl:.6f}",
                "avg_pnl": f"{s.avg_pnl:.6f}",
                "allowed": "True",
                "start_date": start_date or "",
                "end_date": end_date or "",
                "selection_mode": selection_mode,
                "ev_threshold": f"{ev_threshold:.3f}",
                "min_bets": min_bets,
            }
        )
    rows.sort(key=lambda r: (r["league"], r["model"], r["bucket"]))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_selected_rulebook_json(
    out_path: Path,
    selected: Dict[Tuple[str, str, str], Tuple[float, BucketStats]],
    meta: Dict[str, object],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rules = []
    for (league, model, bucket), (t, s) in selected.items():
        rules.append(
            {
                "league": league,
                "model": model,
                "bucket": bucket,
                "edge_threshold": float(t),
                "bets": int(s.bets),
                "wins": int(s.wins),
                "win_rate": float(s.win_rate),
                "total_pnl": float(s.total_pnl),
                "avg_pnl": float(s.avg_pnl),
            }
        )
    rules.sort(key=lambda r: (r["league"], r["model"], r["bucket"]))
    payload = {"meta": meta, "rules": rules}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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

    league_filter = args.league.lower() if args.league else None
    models: List[str] = list(args.models)
    buckets: List[str] = list(args.buckets)

    thresholds: List[float]
    if args.preset_thresholds:
        thresholds = list(DEFAULT_EDGE_THRESHOLDS)
    elif args.edge_thresholds:
        thresholds = list(args.edge_thresholds)
    else:
        thresholds = [float(args.edge_threshold)]
    thresholds = sorted({float(t) for t in thresholds})
    if not thresholds:
        raise SystemExit("[error] no edge thresholds provided")
    for t in thresholds:
        if t <= 0:
            raise SystemExit("[error] edge thresholds must be > 0")

    games = load_games(
        daily_dir=LEDGER_DIR,
        days=days,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=league_filter,
        models=models,
    )
    if not games:
        raise SystemExit("[error] no games found for given filters")

    min_bets = int(args.min_bets)
    ev_threshold = float(args.ev_threshold)

    if len(thresholds) == 1:
        t = thresholds[0]
        stats = compute_bucket_stats(
            games,
            models=models,
            edge_threshold=t,
            buckets=buckets,
        )
        print_table(stats, min_bets=min_bets, ev_threshold=ev_threshold)

        if args.no_write:
            return

        out_dir = Path(args.out_dir)
        out_path = out_dir / "quadrants_rule_stats.csv"
        write_csv(
            out_path,
            stats,
            start_date=args.start_date,
            end_date=args.end_date,
            edge_threshold=t,
            ev_threshold=ev_threshold,
            min_bets=min_bets,
        )
        print(f"\n[info] wrote {out_path}")
        return

    # Sweep thresholds.
    stats_grid: Dict[Tuple[float, str, str, str], BucketStats] = {}
    for t in thresholds:
        stats_t = compute_bucket_stats(
            games,
            models=models,
            edge_threshold=t,
            buckets=buckets,
        )
        print(f"\n--- edge_threshold={t:.3f} ---")
        print_table(stats_t, min_bets=min_bets, ev_threshold=ev_threshold)
        for (league, model, bucket), s in stats_t.items():
            stats_grid[(t, league, model, bucket)] = s

    # Pick per-(league,model,bucket) sweet spots.
    by_key: Dict[Tuple[str, str, str], Dict[float, BucketStats]] = {}
    for (t, league, model, bucket), s in stats_grid.items():
        by_key.setdefault((league, model, bucket), {})[t] = s

    selected: Dict[Tuple[str, str, str], Tuple[float, BucketStats]] = {}
    for key, per_t in by_key.items():
        chosen = select_threshold_for_bucket(
            stats_by_threshold=per_t,
            min_bets=min_bets,
            ev_threshold=ev_threshold,
            mode=str(args.select_threshold_mode),
        )
        if chosen is None:
            continue
        selected[key] = (chosen, per_t[chosen])

    if args.no_write:
        return

    out_dir = Path(args.out_dir)
    sweep_path = out_dir / "quadrants_rule_stats_sweep.csv"
    write_sweep_csv(
        sweep_path,
        stats_grid,
        start_date=args.start_date,
        end_date=args.end_date,
        ev_threshold=ev_threshold,
        min_bets=min_bets,
    )
    print(f"\n[info] wrote {sweep_path}")

    if args.write_selected_rulebook:
        sweet_path = out_dir / "quadrants_rule_sweet_spots.csv"
        write_sweet_spots_csv(
            sweet_path,
            selected,
            start_date=args.start_date,
            end_date=args.end_date,
            selection_mode=str(args.select_threshold_mode),
            ev_threshold=ev_threshold,
            min_bets=min_bets,
        )
        print(f"[info] wrote {sweet_path}")

        rulebook_path = out_dir / "quadrants_rulebook_selected.json"
        meta = {
            "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "start_date": args.start_date or "",
            "end_date": args.end_date or "",
            "league": league_filter or "",
            "models": models,
            "buckets": buckets,
            "edge_thresholds": thresholds,
            "min_bets": min_bets,
            "ev_threshold": ev_threshold,
            "selection_mode": str(args.select_threshold_mode),
        }
        write_selected_rulebook_json(rulebook_path, selected, meta)
        print(f"[info] wrote {rulebook_path}")


if __name__ == "__main__":
    main()
