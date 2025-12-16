#!/usr/bin/env python
"""
Learn per-(league,model,bucket) edge thresholds vs Kalshi mid from daily ledgers (read-only).

This tool scans an edge-threshold grid `t` and evaluates rulebook quadrant buckets
(A/B/C/D and optional sub-buckets) using `kalshi_mid` as the market baseline.

It can optionally apply additive-bias ("handicap") offset calibration to model
probabilities before computing edges/buckets. For each model, it looks for
`chimera_v2c/data/anchor_offset_calibration_<league>_<model>.json` (or falls
back to `anchor_offset_calibration_all_<model>.json`) and applies the selected
slice's `bias_mean` as: p_cal = clamp(p_raw + bias_mean).

Outputs:
  - Sweep CSV under `reports/ev_rulebooks/`
  - Selected rulebook JSON under `chimera_v2c/data/` (snapshots old files)

Daily ledgers are append-only; this tool does not modify them.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, load_games
from chimera_v2c.src.rulebook_quadrants import BucketStats, DEFAULT_BUCKETS
from chimera_v2c.src.threshold_rulebook import (
    SelectedThreshold,
    apply_offset_biases,
    edge_thresholds,
    select_thresholds,
    sweep_rulebook_stats,
)


DEFAULT_MODELS = ["v2c", "gemini", "grok", "gpt", "market_proxy", "moneypuck"]
DEFAULT_CORE_BUCKETS = ["A", "B", "C", "D"]
SNAPSHOT_DIR = Path("reports/rulebook_snapshots")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Learn per-(model,bucket) edge thresholds t* vs Kalshi mid from daily ledgers (read-only).",
    )
    ap.add_argument("--days", type=int, default=0, help="Rolling ledger window (default: 0 = all).")
    ap.add_argument("--start-date", help="YYYY-MM-DD (inclusive); overrides --days.")
    ap.add_argument("--end-date", help="YYYY-MM-DD (inclusive); overrides --days.")
    ap.add_argument("--league", help="Optional league filter (nba|nhl|nfl). Omit for all leagues.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument(
        "--buckets",
        nargs="+",
        help="Buckets to include (default: A B C D). Use --include-subbuckets for A/B/C/D + I/J/K/L/M/N/O/P.",
    )
    ap.add_argument(
        "--include-subbuckets",
        action="store_true",
        help="Include sub-buckets (I/J/K/L/M/N/O/P) in addition to A/B/C/D.",
    )

    ap.add_argument("--min-edge", type=float, default=0.01, help="Min threshold to scan (default: 0.01).")
    ap.add_argument("--max-edge", type=float, default=0.15, help="Max threshold to scan (default: 0.15).")
    ap.add_argument("--step", type=float, default=0.01, help="Threshold step size (default: 0.01).")
    ap.add_argument("--min-bets", type=int, default=10, help="Min bets gate for selecting t* (default: 10).")
    ap.add_argument("--ev-threshold", type=float, default=0.10, help="Min avg_pnl gate for selecting t* (default: 0.10).")
    ap.add_argument(
        "--select-mode",
        choices=["min_edge", "max_avg_pnl", "max_total_pnl"],
        default="min_edge",
        help="How to pick t* among eligible thresholds (default: min_edge).",
    )

    ap.add_argument(
        "--apply-offset-calibration",
        action="store_true",
        help="Apply offset calibration (bias_mean) from anchor_offset_calibration JSON files when present.",
    )
    ap.add_argument(
        "--offset-calibration-dir",
        default="chimera_v2c/data",
        help="Directory containing anchor_offset_calibration_* JSON files (default: chimera_v2c/data).",
    )
    ap.add_argument(
        "--offset-calibration-slice",
        default="ALL",
        help="Which slice bias to apply from anchor_offset_calibration JSON (default: ALL).",
    )

    ap.add_argument(
        "--out-json",
        default="chimera_v2c/data/threshold_rulebook_kalshi_mid.json",
        help="Output JSON path for selected thresholds (default: chimera_v2c/data/threshold_rulebook_kalshi_mid.json).",
    )
    ap.add_argument(
        "--out-dir",
        default="reports/ev_rulebooks",
        help="Directory for derived CSV artifacts (default: reports/ev_rulebooks).",
    )
    ap.add_argument("--no-write", action="store_true", help="Compute + print only; do not write CSV/JSON.")
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="If no graded samples found, write empty outputs instead of exiting non-zero.",
    )
    return ap.parse_args()


def _snapshot_if_exists(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap_path = SNAPSHOT_DIR / f"{path.name}.{ts}.bak"
    try:
        snap_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        return snap_path
    except Exception:
        return None


def _load_bias_mean(
    *,
    calibration_dir: Path,
    league: str,
    model: str,
    slice_name: str,
) -> Optional[float]:
    candidates = [
        calibration_dir / f"anchor_offset_calibration_{league.lower()}_{model.lower()}.json",
        calibration_dir / f"anchor_offset_calibration_all_{model.lower()}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        slices = payload.get("slices") or {}
        s = slices.get(slice_name)
        if not isinstance(s, dict):
            continue
        try:
            return float(s.get("bias_mean", 0.0))
        except Exception:
            continue
    return None


def _write_sweep_csv(
    out_path: Path,
    *,
    stats_grid: Dict[tuple[float, str, str, str], BucketStats],
    thresholds: List[float],
    leagues: List[str],
    models: List[str],
    buckets: List[str],
    meta: Dict[str, object],
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
    ] + list(meta.keys())

    rows: List[Dict[str, object]] = []
    for t in thresholds:
        for league in leagues:
            for model in models:
                for bucket in buckets:
                    s = stats_grid.get((float(t), league, model, bucket), BucketStats())
                    rows.append(
                        {
                            "league": league,
                            "model": model,
                            "bucket": bucket,
                            "edge_threshold": f"{float(t):.3f}",
                            "bets": s.bets,
                            "wins": s.wins,
                            "win_rate": f"{s.win_rate:.6f}",
                            "total_pnl": f"{s.total_pnl:.6f}",
                            "avg_pnl": f"{s.avg_pnl:.6f}",
                            **meta,
                        }
                    )
    rows.sort(key=lambda r: (r["league"], r["model"], r["bucket"], float(r["edge_threshold"])))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_selected_csv(out_path: Path, *, selected: List[SelectedThreshold], meta: Dict[str, object]) -> None:
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
    ] + list(meta.keys())

    rows: List[Dict[str, object]] = []
    for s in selected:
        rows.append(
            {
                "league": s.league,
                "model": s.model,
                "bucket": s.bucket,
                "edge_threshold": f"{s.edge_threshold:.3f}",
                "bets": s.stats.bets,
                "wins": s.stats.wins,
                "win_rate": f"{s.stats.win_rate:.6f}",
                "total_pnl": f"{s.stats.total_pnl:.6f}",
                "avg_pnl": f"{s.stats.avg_pnl:.6f}",
                **meta,
            }
        )
    rows.sort(key=lambda r: (r["league"], r["model"], r["bucket"]))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _print_selected(selected: List[SelectedThreshold]) -> None:
    print("\n=== Selected edge thresholds (t*) vs Kalshi mid ===")
    print(f"{'league':5s} {'model':11s} {'bucket':6s} {'t*':>6s} {'bets':>5s} {'avg_pnl':>8s} {'total_pnl':>9s}")
    for s in sorted(selected, key=lambda x: (x.league, x.model, x.bucket)):
        print(
            f"{s.league:5s} {s.model:11s} {s.bucket:6s} {s.edge_threshold:6.3f} {s.stats.bets:5d} {s.stats.avg_pnl:8.3f} {s.stats.total_pnl:9.3f}"
        )


def main() -> None:
    args = parse_args()
    if not LEDGER_DIR.exists():
        raise SystemExit(f"[error] daily ledger directory missing: {LEDGER_DIR}")

    buckets: List[str]
    if args.buckets:
        buckets = list(args.buckets)
    else:
        buckets = list(DEFAULT_CORE_BUCKETS)
        if args.include_subbuckets:
            buckets = list(DEFAULT_BUCKETS)

    thresholds = edge_thresholds(min_edge=args.min_edge, max_edge=args.max_edge, step=args.step)

    games = load_games(
        daily_dir=LEDGER_DIR,
        days=args.days if not (args.start_date or args.end_date) else None,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=args.league,
        models=list(args.models),
    )

    graded = [g for g in games if g.home_win in (0.0, 1.0) and g.kalshi_mid is not None]
    if not graded and not args.allow_empty:
        raise SystemExit("[error] no graded games with kalshi_mid + actual_outcome found in the selected window.")

    bias_by_model: Dict[str, float] = {}
    bias_sources: Dict[str, str] = {}
    if args.apply_offset_calibration:
        cal_dir = Path(args.offset_calibration_dir)
        league_for_file = (args.league or "all").lower()
        for model in args.models:
            bias = _load_bias_mean(
                calibration_dir=cal_dir,
                league=league_for_file,
                model=model,
                slice_name=args.offset_calibration_slice,
            )
            if bias is None:
                continue
            bias_by_model[model] = float(bias)
            # Best-effort record for metadata (not guaranteed exact path).
            bias_sources[model] = f"anchor_offset_calibration_{league_for_file}_{model}.json|all"

    games_eff = apply_offset_biases(graded, bias_by_model=bias_by_model) if bias_by_model else graded

    stats_grid = sweep_rulebook_stats(
        games_eff,
        thresholds=thresholds,
        models=list(args.models),
        buckets=buckets,
    )
    selected = select_thresholds(
        stats_grid,
        thresholds=thresholds,
        min_bets=args.min_bets,
        ev_threshold=args.ev_threshold,
        mode=args.select_mode,
    )

    # Build metadata fields for CSV/JSON.
    start_used = min((g.date for g in graded), default=None)
    end_used = max((g.date for g in graded), default=None)
    start_used_s = start_used.strftime("%Y-%m-%d") if start_used else ""
    end_used_s = end_used.strftime("%Y-%m-%d") if end_used else ""
    leagues_present = sorted({g.league for g in graded})

    meta = {
        "window_start": start_used_s,
        "window_end": end_used_s,
        "league_filter": (args.league or ""),
        "min_bets": args.min_bets,
        "ev_threshold": f"{float(args.ev_threshold):.3f}",
        "select_mode": args.select_mode,
        "apply_offset_calibration": str(bool(args.apply_offset_calibration)),
        "offset_calibration_slice": args.offset_calibration_slice if args.apply_offset_calibration else "",
    }

    _print_selected(selected)

    if args.no_write:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = out_dir / "threshold_rulebook_kalshi_mid_sweep.csv"
    selected_csv = out_dir / "threshold_rulebook_kalshi_mid_selected.csv"

    _write_sweep_csv(
        sweep_csv,
        stats_grid=stats_grid,
        thresholds=thresholds,
        leagues=leagues_present,
        models=list(args.models),
        buckets=buckets,
        meta=meta,
    )
    _write_selected_csv(selected_csv, selected=selected, meta=meta)
    print(f"[ok] wrote sweep CSV -> {sweep_csv}")
    print(f"[ok] wrote selected CSV -> {selected_csv}")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    snap = _snapshot_if_exists(out_json)
    if snap:
        print(f"[ok] snapshotted prior rulebook -> {snap}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "market_baseline": "kalshi_mid",
        "window_start": start_used_s,
        "window_end": end_used_s,
        "league_filter": (args.league or "all").lower(),
        "models": list(args.models),
        "buckets": buckets,
        "threshold_grid": {
            "min_edge": float(args.min_edge),
            "max_edge": float(args.max_edge),
            "step": float(args.step),
            "thresholds": thresholds,
        },
        "selection": {
            "min_bets": int(args.min_bets),
            "ev_threshold": float(args.ev_threshold),
            "mode": args.select_mode,
        },
        "offset_calibration": {
            "enabled": bool(args.apply_offset_calibration),
            "dir": str(args.offset_calibration_dir),
            "slice": args.offset_calibration_slice if args.apply_offset_calibration else None,
            "bias_by_model": bias_by_model,
            "bias_sources": bias_sources,
        },
        "selected": [
            {
                "league": s.league,
                "model": s.model,
                "bucket": s.bucket,
                "edge_threshold": s.edge_threshold,
                "bets": s.stats.bets,
                "wins": s.stats.wins,
                "win_rate": s.stats.win_rate,
                "total_pnl": s.stats.total_pnl,
                "avg_pnl": s.stats.avg_pnl,
            }
            for s in sorted(selected, key=lambda x: (x.league, x.model, x.bucket))
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ok] wrote selected rulebook JSON -> {out_json}")


if __name__ == "__main__":
    main()

