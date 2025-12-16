#!/usr/bin/env python
"""
Fit additive-bias ("handicap") calibration stats for a probability source vs outcomes.

This is a read-only analysis tool on daily ledgers: it only reads
`reports/daily_ledgers/*_daily_game_ledger.csv` and writes a JSON summary under
`chimera_v2c/data/` (snapshots prior outputs under `reports/calibration_snapshots/`).

We compute:
  residual r = y - p_raw
  bias_mean b = mean(r)
  p_cal = clamp(p_raw + b)

and report a 95% CI half-width for b as 1.96 * stdev(r) / sqrt(n).

We also break down by rulebook quadrants (A/B/C/D + optional sub-buckets) using
`kalshi_mid` as the market baseline and `edge_threshold` as the quadrant gate.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.offset_calibration import OffsetCalibrationStats, compute_offset_calibration
from chimera_v2c.src.rulebook_quadrants import DEFAULT_BUCKETS, bucket_letters


SNAPSHOT_DIR = Path("reports/calibration_snapshots")
DEFAULT_CORE_BUCKETS = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fit additive-bias calibration stats from daily ledgers (overall + rulebook buckets).",
    )
    ap.add_argument("--days", type=int, default=0, help="Rolling ledger window (default: 0 = all).")
    ap.add_argument("--start-date", help="YYYY-MM-DD (inclusive); overrides --days.")
    ap.add_argument("--end-date", help="YYYY-MM-DD (inclusive); overrides --days.")
    ap.add_argument("--league", help="Optional league filter (nba|nhl|nfl). Omit for all leagues.")
    ap.add_argument("--model-col", default="grok", help="Probability column to calibrate (default: grok).")
    ap.add_argument("--edge-threshold", type=float, default=0.05, help="Quadrant edge threshold (default: 0.05).")
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
    ap.add_argument("--out", help="Output JSON path. Defaults under chimera_v2c/data/ based on league/model.")
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="If no samples found, write a JSON with n=0 slices instead of exiting non-zero.",
    )
    return ap.parse_args()


def iter_pairs_for_model(games: Iterable[GameRow], model_col: str) -> Iterable[Tuple[GameRow, float, int]]:
    for g in games:
        if g.home_win not in (0.0, 1.0):
            continue
        if g.kalshi_mid is None:
            continue
        p = g.probs.get(model_col)
        if p is None:
            continue
        yield g, float(p), int(g.home_win)


def _default_out_path(*, league: Optional[str], model_col: str) -> Path:
    league_norm = (league or "all").lower()
    model_norm = model_col.strip().lower()
    return Path(f"chimera_v2c/data/anchor_offset_calibration_{league_norm}_{model_norm}.json")


def snapshot_if_exists(path: Path) -> None:
    if not path.exists():
        return
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap_path = SNAPSHOT_DIR / f"{path.name}.{ts}.bak"
    try:
        snap_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[ok] snapshotted existing params -> {snap_path}")
    except Exception:
        print(f"[warn] failed to snapshot existing output at {path}; proceeding to overwrite.")


def print_summary(
    *,
    slices: Dict[str, OffsetCalibrationStats],
    order: List[str],
) -> None:
    print("\n=== Anchor offset calibration (bias = mean(y - p_raw)) ===")
    print(f"{'slice':8s} {'n':>5s} {'bias':>9s} {'±ci95':>9s} {'brier_raw':>11s} {'brier_cal':>11s}")
    for name in order:
        s = slices.get(name)
        if s is None:
            continue
        br = "" if s.brier_raw is None else f"{s.brier_raw:.4f}"
        bc = "" if s.brier_calibrated is None else f"{s.brier_calibrated:.4f}"
        print(f"{name:8s} {s.n:5d} {s.bias_mean:9.4f} {s.bias_ci95_half_width:9.4f} {br:>11s} {bc:>11s}")


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

    model_col = args.model_col.strip()
    games = load_games(
        daily_dir=LEDGER_DIR,
        days=args.days if not (args.start_date or args.end_date) else None,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=args.league,
        models=[model_col],
    )
    triples = list(iter_pairs_for_model(games, model_col))

    if not triples and not args.allow_empty:
        raise SystemExit(f"[error] no samples found for model_col={model_col} (league={args.league or 'all'})")

    all_pairs: List[Tuple[float, int]] = [(p, y) for _, p, y in triples]
    slices: Dict[str, OffsetCalibrationStats] = {"ALL": compute_offset_calibration(all_pairs)}

    bucket_set = set(buckets)
    bucket_pairs: Dict[str, List[Tuple[float, int]]] = {b: [] for b in buckets}
    for g, p, y in triples:
        letters = bucket_letters(p_mid=float(g.kalshi_mid), p_model=p, edge_threshold=float(args.edge_threshold))
        for b in letters:
            if b in bucket_set:
                bucket_pairs[b].append((p, y))

    for b in buckets:
        slices[b] = compute_offset_calibration(bucket_pairs.get(b, []))

    # Resolve actual included date range (by ledger filename date) for metadata.
    date_min = min((g.date for g, _, _ in triples), default=None)
    date_max = max((g.date for g, _, _ in triples), default=None)
    start_used = date_min.strftime("%Y-%m-%d") if date_min else ""
    end_used = date_max.strftime("%Y-%m-%d") if date_max else ""

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "league": (args.league or "all").lower(),
        "model_col": model_col,
        "edge_threshold": float(args.edge_threshold),
        "start_date": start_used,
        "end_date": end_used,
        "buckets": buckets,
        "slices": {
            k: {
                "n": v.n,
                "bias_mean": v.bias_mean,
                "bias_stdev": v.bias_stdev,
                "bias_ci95_half_width": v.bias_ci95_half_width,
                "brier_raw": v.brier_raw,
                "brier_calibrated": v.brier_calibrated,
            }
            for k, v in slices.items()
        },
    }

    out_path = Path(args.out) if args.out else _default_out_path(league=args.league, model_col=model_col)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_if_exists(out_path)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[ok] wrote anchor offset calibration -> {out_path}")
    order = ["ALL"] + [b for b in buckets if b in slices]
    print_summary(slices=slices, order=order)


if __name__ == "__main__":
    main()

