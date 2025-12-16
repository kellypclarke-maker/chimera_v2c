#!/usr/bin/env python
"""
Learn a Grok↔Kalshi-mid shrinkage hybrid from daily ledgers (read-only).

We fit two things:
  1) Platt calibration for Grok: p_grok_cal = sigmoid(a*logit(p_grok_raw) + b)
  2) A single shrinkage scalar alpha in [0,1]:
       p_hybrid = clamp(p_mid + alpha * (p_grok_cal - p_mid))

Alpha is chosen by leave-one-ledger-date-out CV to minimize mean Brier score
vs actual outcomes. This is intentionally "small-N safe": if Grok is not better
than market in a regime, CV will push alpha toward 0 (market-following).

Outputs:
  - JSON params under `chimera_v2c/data/` (snapshots prior files)
  - CSV curve under `reports/thesis_summaries/` (alpha vs CV Brier)

Daily ledgers are append-only; this tool does not modify them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from chimera_v2c.src.calibration import PlattScaler, fit_platt
from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.offset_calibration import clamp_prob


SNAPSHOT_DIR = Path("reports/calibration_snapshots")
THESIS_DIR = Path("reports/thesis_summaries")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Learn a Platt-calibrated Grok↔Kalshi-mid shrinkage hybrid (read-only).",
    )
    ap.add_argument("--days", type=int, default=0, help="Rolling ledger window (default: 0 = all).")
    ap.add_argument("--start-date", help="YYYY-MM-DD (inclusive); overrides --days.")
    ap.add_argument("--end-date", help="YYYY-MM-DD (inclusive); overrides --days.")
    ap.add_argument("--league", help="Optional league filter (nba|nhl|nfl). Omit for all leagues.")
    ap.add_argument("--model-col", default="grok", help="Model probability column to use (default: grok).")
    ap.add_argument("--min-samples", type=int, default=30, help="Min samples to fit Platt in a fold (default: 30).")
    ap.add_argument("--alpha-step", type=float, default=0.01, help="Alpha grid step in [0,1] (default: 0.01).")
    ap.add_argument(
        "--out",
        help="Output JSON path. Default: chimera_v2c/data/grok_mid_hybrid_<league>.json",
    )
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="If no samples found, write identity params with n=0 instead of exiting non-zero.",
    )
    ap.add_argument(
        "--no-write",
        action="store_true",
        help="Compute and print only; do not write JSON/CSV.",
    )
    return ap.parse_args()


def _default_out_path(*, league: Optional[str]) -> Path:
    league_norm = (league or "all").lower()
    return Path(f"chimera_v2c/data/grok_mid_hybrid_{league_norm}.json")


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


def _alpha_grid(step: float) -> List[float]:
    if step <= 0 or step > 1:
        raise SystemExit("[error] --alpha-step must be in (0,1].")
    out = []
    a = 0.0
    while a <= 1.0 + 1e-9:
        out.append(round(a, 3))
        a += step
    # Ensure 1.0 is included.
    if out[-1] != 1.0:
        out.append(1.0)
    return out


def iter_samples(games: Iterable[GameRow], model_col: str) -> Iterable[Tuple[str, float, float, int]]:
    for g in games:
        if g.home_win not in (0.0, 1.0):
            continue
        if g.kalshi_mid is None:
            continue
        p = g.probs.get(model_col)
        if p is None:
            continue
        date_key = g.date.strftime("%Y-%m-%d")
        yield date_key, clamp_prob(float(p)), clamp_prob(float(g.kalshi_mid)), int(g.home_win)


def brier(p: float, y: int) -> float:
    return (float(p) - float(y)) ** 2


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def cv_alpha_brier(
    samples: List[Tuple[str, float, float, int]],
    *,
    alpha_grid: List[float],
    min_samples: int,
) -> Tuple[Dict[float, float], Dict[str, float]]:
    """
    Leave-one-ledger-date-out CV for alpha.

    Returns:
      - cv_brier_by_alpha: mean brier across all holdouts for each alpha
      - baselines: {'kalshi_mid': ..., 'grok_raw': ..., 'grok_cal': ...}
    """
    by_date: dict[str, List[Tuple[float, float, int]]] = defaultdict(list)
    for d, p_grok, p_mid, y in samples:
        by_date[d].append((p_grok, p_mid, y))
    dates = sorted(by_date.keys())
    if len(dates) < 2:
        raise SystemExit("[error] need at least 2 distinct ledger dates for CV; use a wider window.")

    sq_by_alpha: Dict[float, float] = {a: 0.0 for a in alpha_grid}
    n_by_alpha: Dict[float, int] = {a: 0 for a in alpha_grid}

    base_sq = {"kalshi_mid": 0.0, "grok_raw": 0.0, "grok_cal": 0.0}
    base_n = 0

    for holdout in dates:
        train_pairs = [(p, y) for d, p, _, y in samples if d != holdout]
        if len(train_pairs) < min_samples:
            scaler = PlattScaler(a=1.0, b=0.0)
        else:
            scaler = fit_platt(train_pairs)

        for p_grok, p_mid, y in by_date[holdout]:
            yv = int(y)
            p_mid_f = clamp_prob(p_mid)
            p_grok_raw = clamp_prob(p_grok)
            p_grok_cal = clamp_prob(scaler.predict(p_grok_raw))

            base_sq["kalshi_mid"] += brier(p_mid_f, yv)
            base_sq["grok_raw"] += brier(p_grok_raw, yv)
            base_sq["grok_cal"] += brier(p_grok_cal, yv)
            base_n += 1

            for alpha in alpha_grid:
                p_hybrid = clamp_prob(p_mid_f + float(alpha) * (p_grok_cal - p_mid_f))
                sq_by_alpha[alpha] += brier(p_hybrid, yv)
                n_by_alpha[alpha] += 1

    cv_brier_by_alpha = {a: (sq_by_alpha[a] / n_by_alpha[a] if n_by_alpha[a] else float("inf")) for a in alpha_grid}
    baselines = {k: (base_sq[k] / base_n if base_n else float("inf")) for k in base_sq}
    return cv_brier_by_alpha, baselines


def fit_full_platt(samples: List[Tuple[str, float, float, int]], *, min_samples: int) -> PlattScaler:
    pairs = [(p, y) for _, p, _, y in samples]
    if len(pairs) < min_samples:
        return PlattScaler(a=1.0, b=0.0)
    return fit_platt(pairs)


def in_sample_metrics(samples: List[Tuple[str, float, float, int]], *, scaler: PlattScaler, alpha: float) -> Dict[str, float]:
    sq = {"kalshi_mid": 0.0, "grok_raw": 0.0, "grok_cal": 0.0, "hybrid": 0.0}
    n = 0
    for _, p_grok, p_mid, y in samples:
        yv = int(y)
        p_mid_f = clamp_prob(p_mid)
        p_grok_raw = clamp_prob(p_grok)
        p_grok_cal = clamp_prob(scaler.predict(p_grok_raw))
        p_hybrid = clamp_prob(p_mid_f + float(alpha) * (p_grok_cal - p_mid_f))
        sq["kalshi_mid"] += brier(p_mid_f, yv)
        sq["grok_raw"] += brier(p_grok_raw, yv)
        sq["grok_cal"] += brier(p_grok_cal, yv)
        sq["hybrid"] += brier(p_hybrid, yv)
        n += 1
    if n == 0:
        return {k: float("inf") for k in sq}
    return {k: sq[k] / n for k in sq}


def main() -> None:
    args = parse_args()
    if not LEDGER_DIR.exists():
        raise SystemExit(f"[error] daily ledger directory missing: {LEDGER_DIR}")

    model_col = args.model_col.strip()
    games = load_games(
        daily_dir=LEDGER_DIR,
        days=args.days if not (args.start_date or args.end_date) else None,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=args.league,
        models=[model_col],
    )
    samples = list(iter_samples(games, model_col=model_col))
    if not samples and not args.allow_empty:
        raise SystemExit(f"[error] no samples found for model_col={model_col} (league={args.league or 'all'})")

    if not samples and args.allow_empty:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "league": (args.league or "all").lower(),
            "model_col": model_col,
            "n": 0,
            "a": 1.0,
            "b": 0.0,
            "alpha": 0.0,
            "cv": {},
        }
        out_path = Path(args.out) if args.out else _default_out_path(league=args.league)
        if not args.no_write:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            snap = _snapshot_if_exists(out_path)
            if snap:
                print(f"[ok] snapshotted -> {snap}")
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ok] wrote -> {out_path}")
        else:
            print(json.dumps(payload, indent=2))
        return

    # Determine actual included date range for metadata.
    date_keys = sorted({d for d, _, _, _ in samples})
    window_start = date_keys[0] if date_keys else ""
    window_end = date_keys[-1] if date_keys else ""

    alphas = _alpha_grid(args.alpha_step)
    cv_curve, cv_baselines = cv_alpha_brier(samples, alpha_grid=alphas, min_samples=args.min_samples)
    best_alpha = min(cv_curve.items(), key=lambda kv: kv[1])[0]

    scaler = fit_full_platt(samples, min_samples=args.min_samples)
    ins = in_sample_metrics(samples, scaler=scaler, alpha=best_alpha)

    print("\n=== Grok↔Kalshi-mid hybrid (Platt + alpha) ===")
    print(f"league={args.league or 'all'} model_col={model_col} window={window_start}..{window_end} n={len(samples)}")
    print(f"Platt: a={scaler.a:.4f} b={scaler.b:.4f} (min_samples={args.min_samples})")
    print(f"Best alpha (CV Brier): {best_alpha:.3f}  CV_brier={cv_curve[best_alpha]:.6f}")
    print(f"CV baselines: kalshi_mid={cv_baselines['kalshi_mid']:.6f} grok_raw={cv_baselines['grok_raw']:.6f} grok_cal={cv_baselines['grok_cal']:.6f}")
    print(f"In-sample: kalshi_mid={ins['kalshi_mid']:.6f} grok_raw={ins['grok_raw']:.6f} grok_cal={ins['grok_cal']:.6f} hybrid={ins['hybrid']:.6f}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "league": (args.league or "all").lower(),
        "model_col": model_col,
        "window_start": window_start,
        "window_end": window_end,
        "n": len(samples),
        "a": scaler.a,
        "b": scaler.b,
        "alpha": float(best_alpha),
        "cv": {
            "objective": "brier",
            "alpha_step": float(args.alpha_step),
            "cv_brier_by_alpha": [{"alpha": float(a), "cv_brier": float(cv_curve[a])} for a in alphas],
            "baselines": cv_baselines,
        },
        "in_sample_brier": ins,
    }

    out_path = Path(args.out) if args.out else _default_out_path(league=args.league)
    curve_path = THESIS_DIR / f"grok_mid_hybrid_curve_{(args.league or 'all').lower()}_{window_start.replace('-','')}_{window_end.replace('-','')}.csv"

    if args.no_write:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    snap = _snapshot_if_exists(out_path)
    if snap:
        print(f"[ok] snapshotted -> {snap}")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ok] wrote -> {out_path}")

    THESIS_DIR.mkdir(parents=True, exist_ok=True)
    with curve_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "cv_brier"])
        w.writeheader()
        for a in alphas:
            w.writerow({"alpha": f"{float(a):.3f}", "cv_brier": f"{float(cv_curve[a]):.9f}"})
    print(f"[ok] wrote -> {curve_path}")


if __name__ == "__main__":
    main()

