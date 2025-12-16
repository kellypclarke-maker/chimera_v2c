#!/usr/bin/env python
"""
Walk-forward (train→test) evaluation for the Grok↔Kalshi-mid shrinkage hybrid.

This tool answers: "Does the Grok-mid hybrid beat Kalshi mid out-of-sample?"

Method (per league):
  For each ledger date D in [start,end]:
    - Train on games from dates < D (optionally last N train days).
    - Pick alpha in [0,1] via leave-one-train-date-out CV on the TRAIN set.
      (Within each inner fold, fit Platt on the remaining train dates.)
    - Fit Platt on the full TRAIN set.
    - Evaluate on TEST date D:
        grok_raw, grok_platt, grok_mid_hybrid, and kalshi_mid baseline.

Outputs (CSV under reports/thesis_summaries by default):
  - Daily fold results with chosen (a,b,alpha)
  - Summary results (mean Brier + delta vs kalshi_mid with CI)

Safety: read-only on daily ledgers.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chimera_v2c.src.calibration import PlattScaler, fit_platt
from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.offset_calibration import clamp_prob


DEFAULT_OUT_DIR = Path("reports/thesis_summaries")


@dataclass(frozen=True)
class Sample:
    d: Date
    p_grok: float
    p_mid: float
    y: int


@dataclass(frozen=True)
class FoldResult:
    league: str
    test_date: Date
    trained: bool
    train_dates: int
    train_samples: int
    platt_a: float
    platt_b: float
    alpha: float
    test_samples: int
    brier_kalshi_mid: Optional[float]
    brier_grok_raw: Optional[float]
    brier_grok_platt: Optional[float]
    brier_grok_mid_hybrid: Optional[float]


def _alpha_grid(step: float) -> List[float]:
    if step <= 0 or step > 1:
        raise ValueError("alpha_step must be in (0,1].")
    out: List[float] = []
    a = 0.0
    while a <= 1.0 + 1e-9:
        out.append(round(a, 3))
        a += step
    if out[-1] != 1.0:
        out.append(1.0)
    return out


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _sample_stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = float(sum(values) / len(values))
    var = float(sum((x - mu) ** 2 for x in values) / (len(values) - 1))
    return math.sqrt(var)


def _brier(p: float, y: int) -> float:
    return (float(p) - float(y)) ** 2


def _cv_select_alpha(
    samples: Sequence[Sample],
    *,
    alpha_grid: Sequence[float],
    min_platt_samples: int,
) -> float:
    """
    Choose alpha by leave-one-train-date-out CV (objective: mean Brier).

    This is nested inside the outer walk-forward fold to avoid test leakage.
    """
    by_date: Dict[Date, List[Sample]] = {}
    for s in samples:
        by_date.setdefault(s.d, []).append(s)
    dates = sorted(by_date.keys())
    if len(dates) < 2:
        return 0.0

    sq_by_alpha = {float(a): 0.0 for a in alpha_grid}
    n_by_alpha = {float(a): 0 for a in alpha_grid}

    for holdout in dates:
        train_pairs = [(ss.p_grok, ss.y) for d in dates if d != holdout for ss in by_date[d]]
        if len(train_pairs) < int(min_platt_samples):
            scaler = PlattScaler(a=1.0, b=0.0)
        else:
            scaler = fit_platt(train_pairs)

        for s in by_date[holdout]:
            p_mid = clamp_prob(s.p_mid)
            p_grok_raw = clamp_prob(s.p_grok)
            p_grok_cal = clamp_prob(scaler.predict(p_grok_raw))
            for a in alpha_grid:
                alpha = float(a)
                p_hybrid = clamp_prob(p_mid + alpha * (p_grok_cal - p_mid))
                sq_by_alpha[alpha] += _brier(p_hybrid, s.y)
                n_by_alpha[alpha] += 1

    best_alpha = 0.0
    best = float("inf")
    for alpha in alpha_grid:
        a = float(alpha)
        n = n_by_alpha[a]
        if n <= 0:
            continue
        m = sq_by_alpha[a] / n
        if m < best:
            best = m
            best_alpha = a
    return float(best_alpha)


def _fit_platt_or_identity(pairs: Sequence[Tuple[float, int]], *, min_platt_samples: int) -> PlattScaler:
    if len(pairs) < int(min_platt_samples):
        return PlattScaler(a=1.0, b=0.0)
    return fit_platt(pairs)


def _predict_hybrid(*, scaler: PlattScaler, alpha: float, p_grok_raw: float, p_mid: float) -> float:
    p_mid_f = clamp_prob(float(p_mid))
    p_grok_cal = clamp_prob(scaler.predict(clamp_prob(float(p_grok_raw))))
    alpha_f = float(max(0.0, min(1.0, float(alpha))))
    return clamp_prob(p_mid_f + alpha_f * (p_grok_cal - p_mid_f))


def walkforward_eval(
    samples: Sequence[Sample],
    *,
    league: str,
    alpha_step: float = 0.01,
    min_platt_samples: int = 30,
    train_days: int = 0,
    skip_untrained: bool = False,
) -> List[FoldResult]:
    """
    Walk-forward evaluation over the provided samples (single league).

    samples must already be restricted to the desired date window and have
    known outcomes and kalshi mid, with grok present.
    """
    if not samples:
        return []

    alpha_grid = _alpha_grid(alpha_step)
    by_date: Dict[Date, List[Sample]] = {}
    for s in samples:
        by_date.setdefault(s.d, []).append(s)
    dates = sorted(by_date.keys())

    results: List[FoldResult] = []
    for test_date in dates:
        train_dates = [d for d in dates if d < test_date]
        if train_days and train_days > 0:
            cutoff = test_date.toordinal() - int(train_days)
            train_dates = [d for d in train_dates if d.toordinal() >= cutoff]
        train_samples = [s for d in train_dates for s in by_date.get(d, [])]
        test_samples = list(by_date.get(test_date, []))

        trained = len(train_dates) >= 2 and len(train_samples) >= int(min_platt_samples)
        if skip_untrained and not trained:
            continue

        if trained:
            alpha = _cv_select_alpha(train_samples, alpha_grid=alpha_grid, min_platt_samples=min_platt_samples)
            scaler = _fit_platt_or_identity([(s.p_grok, s.y) for s in train_samples], min_platt_samples=min_platt_samples)
        else:
            alpha = 0.0
            scaler = PlattScaler(a=1.0, b=0.0)

        se_mid: List[float] = []
        se_grok_raw: List[float] = []
        se_grok_platt: List[float] = []
        se_hybrid: List[float] = []
        for s in test_samples:
            p_mid = clamp_prob(s.p_mid)
            p_grok_raw = clamp_prob(s.p_grok)
            p_grok_platt = clamp_prob(scaler.predict(p_grok_raw))
            p_hybrid = _predict_hybrid(scaler=scaler, alpha=alpha, p_grok_raw=p_grok_raw, p_mid=p_mid)

            se_mid.append(_brier(p_mid, s.y))
            se_grok_raw.append(_brier(p_grok_raw, s.y))
            se_grok_platt.append(_brier(p_grok_platt, s.y))
            se_hybrid.append(_brier(p_hybrid, s.y))

        results.append(
            FoldResult(
                league=league,
                test_date=test_date,
                trained=trained,
                train_dates=len(train_dates),
                train_samples=len(train_samples),
                platt_a=float(scaler.a),
                platt_b=float(scaler.b),
                alpha=float(alpha),
                test_samples=len(test_samples),
                brier_kalshi_mid=_mean(se_mid),
                brier_grok_raw=_mean(se_grok_raw),
                brier_grok_platt=_mean(se_grok_platt),
                brier_grok_mid_hybrid=_mean(se_hybrid),
            )
        )

    return results


def _samples_from_games(games: Iterable[GameRow]) -> List[Sample]:
    out: List[Sample] = []
    for g in games:
        if g.home_win not in (0.0, 1.0):
            continue
        if g.kalshi_mid is None:
            continue
        p_grok = g.probs.get("grok")
        if p_grok is None:
            continue
        out.append(
            Sample(
                d=g.date.date(),
                p_grok=clamp_prob(float(p_grok)),
                p_mid=clamp_prob(float(g.kalshi_mid)),
                y=int(g.home_win),
            )
        )
    return out


def _write_daily_csv(path: Path, folds: Sequence[FoldResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "league",
        "test_date",
        "trained",
        "train_dates",
        "train_samples",
        "platt_a",
        "platt_b",
        "alpha",
        "test_samples",
        "brier_kalshi_mid",
        "brier_grok_raw",
        "brier_grok_platt",
        "brier_grok_mid_hybrid",
        "delta_hybrid_vs_kalshi_mid",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for fr in folds:
            base = fr.brier_kalshi_mid
            hybrid = fr.brier_grok_mid_hybrid
            delta = (hybrid - base) if (base is not None and hybrid is not None) else None
            w.writerow(
                {
                    "league": fr.league,
                    "test_date": fr.test_date.isoformat(),
                    "trained": str(bool(fr.trained)),
                    "train_dates": fr.train_dates,
                    "train_samples": fr.train_samples,
                    "platt_a": f"{fr.platt_a:.6f}",
                    "platt_b": f"{fr.platt_b:.6f}",
                    "alpha": f"{fr.alpha:.3f}",
                    "test_samples": fr.test_samples,
                    "brier_kalshi_mid": f"{fr.brier_kalshi_mid:.6f}" if fr.brier_kalshi_mid is not None else "",
                    "brier_grok_raw": f"{fr.brier_grok_raw:.6f}" if fr.brier_grok_raw is not None else "",
                    "brier_grok_platt": f"{fr.brier_grok_platt:.6f}" if fr.brier_grok_platt is not None else "",
                    "brier_grok_mid_hybrid": f"{fr.brier_grok_mid_hybrid:.6f}" if fr.brier_grok_mid_hybrid is not None else "",
                    "delta_hybrid_vs_kalshi_mid": f"{delta:.6f}" if delta is not None else "",
                }
            )


def _summary_rows(folds: Sequence[FoldResult], *, subset: str) -> List[Dict[str, object]]:
    """
    Summarize mean Brier (weighted by #games per test date).

    CI is computed over per-date delta values (each test date is one sample),
    so it reflects stability across days rather than across individual games.
    """
    usable = [
        fr
        for fr in folds
        if fr.test_samples > 0
        and fr.brier_kalshi_mid is not None
        and fr.brier_grok_raw is not None
        and fr.brier_grok_platt is not None
        and fr.brier_grok_mid_hybrid is not None
    ]
    n_dates = len(usable)
    n_games = sum(fr.test_samples for fr in usable)

    if n_dates == 0 or n_games == 0:
        return [
            {
                "subset": subset,
                "model": m,
                "n_dates": 0,
                "n_games": 0,
                "mean_brier": "",
                "delta_vs_kalshi_mid": "",
                "delta_ci95_half_width": "",
            }
            for m in ("kalshi_mid", "grok_raw", "grok_platt", "grok_mid_hybrid")
        ]

    totals = {
        "kalshi_mid": sum(float(fr.brier_kalshi_mid) * fr.test_samples for fr in usable),
        "grok_raw": sum(float(fr.brier_grok_raw) * fr.test_samples for fr in usable),
        "grok_platt": sum(float(fr.brier_grok_platt) * fr.test_samples for fr in usable),
        "grok_mid_hybrid": sum(float(fr.brier_grok_mid_hybrid) * fr.test_samples for fr in usable),
    }
    means = {k: totals[k] / n_games for k in totals}

    # Per-date deltas for CI (date-level stability).
    deltas_by_model: Dict[str, List[float]] = {
        "grok_raw": [],
        "grok_platt": [],
        "grok_mid_hybrid": [],
    }
    for fr in usable:
        base = float(fr.brier_kalshi_mid)
        deltas_by_model["grok_raw"].append(float(fr.brier_grok_raw) - base)
        deltas_by_model["grok_platt"].append(float(fr.brier_grok_platt) - base)
        deltas_by_model["grok_mid_hybrid"].append(float(fr.brier_grok_mid_hybrid) - base)

    def ci_half_width_date(deltas: List[float]) -> float:
        if len(deltas) < 2:
            return 0.0
        st = _sample_stdev(deltas)
        return 1.96 * (st / math.sqrt(len(deltas)))

    rows: List[Dict[str, object]] = []
    rows.append(
        {
            "subset": subset,
            "model": "kalshi_mid",
            "n_dates": n_dates,
            "n_games": n_games,
            "mean_brier": f"{means['kalshi_mid']:.6f}",
            "delta_vs_kalshi_mid": f"{0.0:.6f}",
            "delta_ci95_half_width": f"{0.0:.6f}",
        }
    )
    for model in ("grok_raw", "grok_platt", "grok_mid_hybrid"):
        rows.append(
            {
                "subset": subset,
                "model": model,
                "n_dates": n_dates,
                "n_games": n_games,
                "mean_brier": f"{means[model]:.6f}",
                "delta_vs_kalshi_mid": f"{(means[model] - means['kalshi_mid']):.6f}",
                "delta_ci95_half_width": f"{ci_half_width_date(deltas_by_model[model]):.6f}",
            }
        )
    return rows


def _write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subset",
        "model",
        "n_dates",
        "n_games",
        "mean_brier",
        "delta_vs_kalshi_mid",
        "delta_ci95_half_width",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward evaluation for the Grok↔Kalshi-mid hybrid (read-only).")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD (inclusive).")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive).")
    ap.add_argument("--league", default="all", help="nba|nhl|nfl|all (default: all).")
    ap.add_argument("--train-days", type=int, default=0, help="Use only the last N train days (default: 0 = all prior).")
    ap.add_argument("--min-platt-samples", type=int, default=30, help="Min samples to fit Platt (default: 30).")
    ap.add_argument("--alpha-step", type=float, default=0.01, help="Alpha grid step in [0,1] (default: 0.01).")
    ap.add_argument(
        "--skip-untrained",
        action="store_true",
        help="Skip test dates where the training set is too small to fit Platt and alpha (default: include with alpha=0).",
    )
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory (default: reports/thesis_summaries/).")
    ap.add_argument("--no-write", action="store_true", help="Compute and print only; do not write CSVs.")
    args = ap.parse_args()

    league = args.league.strip().lower()
    league_filter = None if league in {"all", ""} else league
    games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=league_filter,
        models=["grok"],
    )
    if not games:
        raise SystemExit("[error] no games found for the given window/league.")

    # Group by league (even when --league all) for per-league reporting.
    by_league: Dict[str, List[GameRow]] = {}
    for g in games:
        by_league.setdefault(g.league, []).append(g)

    out_dir = Path(args.out_dir)
    start = args.start_date.replace("-", "")
    end = args.end_date.replace("-", "")

    all_summary_rows: List[Dict[str, object]] = []
    for league_key, league_games in sorted(by_league.items()):
        samples = _samples_from_games(league_games)
        folds = walkforward_eval(
            samples,
            league=league_key,
            alpha_step=float(args.alpha_step),
            min_platt_samples=int(args.min_platt_samples),
            train_days=int(args.train_days),
            skip_untrained=bool(args.skip_untrained),
        )
        if not folds:
            continue

        trained_folds = [f for f in folds if f.trained]
        summary_rows = []
        summary_rows.extend(_summary_rows(folds, subset="all_folds"))
        summary_rows.extend(_summary_rows(trained_folds, subset="trained_folds"))
        for r in summary_rows:
            r["league"] = league_key
        all_summary_rows.extend(summary_rows)

        if not args.no_write:
            daily_path = out_dir / f"grok_mid_hybrid_walkforward_daily_{league_key}_{start}_{end}.csv"
            summary_path = out_dir / f"grok_mid_hybrid_walkforward_summary_{league_key}_{start}_{end}.csv"
            _write_daily_csv(daily_path, folds)
            _write_summary_csv(summary_path, [{k: v for k, v in row.items() if k != "league"} for row in summary_rows])
            print(f"[ok] wrote {daily_path}")
            print(f"[ok] wrote {summary_path}")

        # Print a compact stdout summary (trained folds if available; else all).
        pick = trained_folds if trained_folds else folds
        best_subset = "trained_folds" if trained_folds else "all_folds"
        rows = _summary_rows(pick, subset=best_subset)
        print(f"\n=== {league_key.upper()} walk-forward ({best_subset}) ===")
        for row in rows:
            print(
                f"{row['model']:14s} n={row['n_games']:4d} brier={row['mean_brier']} "
                f"delta_vs_mid={row['delta_vs_kalshi_mid']} ±{row['delta_ci95_half_width']}"
            )

    if not all_summary_rows:
        raise SystemExit("[error] no fold results produced (insufficient samples?).")

    # Overall combined view (across leagues) is written as a single summary CSV.
    if not args.no_write:
        overall_path = out_dir / f"grok_mid_hybrid_walkforward_summary_all_{start}_{end}.csv"
        fieldnames = [
            "league",
            "subset",
            "model",
            "n_dates",
            "n_games",
            "mean_brier",
            "delta_vs_kalshi_mid",
            "delta_ci95_half_width",
        ]
        overall_path.parent.mkdir(parents=True, exist_ok=True)
        with overall_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_summary_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\n[ok] wrote {overall_path}")


if __name__ == "__main__":
    main()
