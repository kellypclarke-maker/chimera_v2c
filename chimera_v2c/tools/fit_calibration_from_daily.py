"""
Fit Platt calibration parameters from daily ledgers (per league, per model).

Usage examples (from repo root):
  # NBA v2c-only calibration from daily ledgers
  PYTHONPATH=. python chimera_v2c/tools/fit_calibration_from_daily.py \\
      --league nba --model-col v2c --out chimera_v2c/data/calibration_params_nba.json

  # Gemini calibration
  PYTHONPATH=. python chimera_v2c/tools/fit_calibration_from_daily.py \\
      --league nba --model-col gemini --out chimera_v2c/data/calibration_params_nba_gemini.json

This reads `reports/daily_ledgers/*_daily_game_ledger.csv` and extracts
`(p_model, y_true)` pairs for the requested league/model, where:
  - `p_model` comes from the specified column (home win probability), and
  - `y_true` is derived from the `actual_outcome` string:
      * 1 for home win
      * 0 for home loss (away win)
      * rows with pushes or missing outcomes are skipped.

It then fits a Platt scaler using `chimera_v2c.src.calibration.fit_platt`
and writes JSON with fields:
  {"a": <float>, "b": <float>, "n": <int>}

This tool is read-only with respect to the daily ledgers; it only writes
to the specified calibration params path under `chimera_v2c/data/`.

Important note:
  - When the league config has `calibration.enabled: true`, the planner’s `v2c`
    output is already calibrated at plan time. For refreshing the pipeline
    calibrator, prefer an uncalibrated probability source (e.g., a saved plan JSON)
    rather than calibrating the already-calibrated `v2c` daily-ledger column.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from chimera_v2c.src.calibration import PlattScaler, fit_platt
from chimera_v2c.src.ledger.outcomes import parse_home_win


LEDGER_DIR = Path("reports/daily_ledgers")
DEFAULT_MODEL_COL = "v2c"


def _parse_home_win(actual_outcome: str) -> float | None:
    return parse_home_win(actual_outcome)


def _parse_ledger_date_from_filename(path: Path) -> Optional[datetime]:
    m = re.match(r"(\d{8})_daily_game_ledger\.csv", path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d")
    except ValueError:
        return None


def _iter_pairs_from_daily(
    *,
    league: str,
    model_col: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Tuple[str, float, int]]:
    if not LEDGER_DIR.exists():
        raise SystemExit(f"[error] daily ledger directory missing: {LEDGER_DIR}")

    league_norm = league.lower()
    pairs: List[Tuple[str, float, int]] = []

    paths = sorted(LEDGER_DIR.glob("*_daily_game_ledger.csv"))
    if not paths:
        raise SystemExit(f"[error] no *_daily_game_ledger.csv files found in {LEDGER_DIR}")

    for path in paths:
        ledger_date = _parse_ledger_date_from_filename(path)
        if ledger_date is None:
            continue
        if start_date:
            try:
                if ledger_date < datetime.strptime(start_date, "%Y-%m-%d"):
                    continue
            except ValueError:
                pass
        if end_date:
            try:
                if ledger_date > datetime.strptime(end_date, "%Y-%m-%d"):
                    continue
            except ValueError:
                pass

        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("league") or "").strip().lower() != league_norm:
                    continue
                outcome = row.get("actual_outcome") or ""
                hw = _parse_home_win(outcome)
                if hw not in (0.0, 1.0):
                    continue
                v_str = row.get(model_col)
                if v_str is None or str(v_str).strip() == "":
                    continue
                try:
                    p = float(v_str)
                except ValueError:
                    continue
                pairs.append((ledger_date.strftime("%Y-%m-%d"), p, int(hw)))
    return pairs


def _mean_brier(pairs: List[Tuple[float, int]], *, scaler: Optional[PlattScaler] = None) -> Optional[float]:
    if not pairs:
        return None
    s = 0.0
    for p, y in pairs:
        q = scaler.predict(p) if scaler else p
        s += (q - float(y)) ** 2
    return s / len(pairs)


def _mean_logloss(pairs: List[Tuple[float, int]], *, scaler: Optional[PlattScaler] = None) -> Optional[float]:
    if not pairs:
        return None
    s = 0.0
    for p, y in pairs:
        q = scaler.predict(p) if scaler else p
        q = max(1e-6, min(1 - 1e-6, q))
        yv = float(y)
        s += -(yv * math.log(q) + (1.0 - yv) * math.log(1.0 - q))
    return s / len(pairs)


def _cv_by_date(
    triples: List[Tuple[str, float, int]],
    *,
    min_samples: int,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Leave-one-ledger-date-out CV.
    Returns: (raw_brier, cal_brier, raw_logloss, cal_logloss)
    """
    if not triples:
        return None, None, None, None
    by_date: dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for d, p, y in triples:
        by_date[d].append((p, y))
    dates = sorted(by_date.keys())
    if len(dates) < 2:
        return None, None, None, None

    raw_sq = 0.0
    cal_sq = 0.0
    raw_ll = 0.0
    cal_ll = 0.0
    n_total = 0

    for holdout in dates:
        test_pairs = by_date[holdout]
        train_pairs = [(p, y) for d, p, y in triples if d != holdout]
        if len(train_pairs) < min_samples:
            scaler = PlattScaler(a=1.0, b=0.0)
        else:
            scaler = fit_platt(train_pairs)

        for p, y in test_pairs:
            yv = float(y)
            raw_sq += (p - yv) ** 2
            q = scaler.predict(p)
            cal_sq += (q - yv) ** 2

            p_ll = max(1e-6, min(1 - 1e-6, p))
            q_ll = max(1e-6, min(1 - 1e-6, q))
            raw_ll += -(yv * math.log(p_ll) + (1.0 - yv) * math.log(1.0 - p_ll))
            cal_ll += -(yv * math.log(q_ll) + (1.0 - yv) * math.log(1.0 - q_ll))
            n_total += 1

    if n_total == 0:
        return None, None, None, None
    return raw_sq / n_total, cal_sq / n_total, raw_ll / n_total, cal_ll / n_total


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit Platt calibration params for v2c from daily ledgers (per league)."
    )
    ap.add_argument("--league", default="nba", help="League filter (default: nba).")
    ap.add_argument(
        "--model-col",
        default=DEFAULT_MODEL_COL,
        help="Which daily-ledger probability column to calibrate (default: v2c).",
    )
    ap.add_argument(
        "--out",
        help="Output JSON path. If omitted, writes chimera_v2c/data/calibration_params_<league>_<model>.json.",
    )
    ap.add_argument("--start-date", help="Optional start date YYYY-MM-DD (filters ledger files by filename date).")
    ap.add_argument("--end-date", help="Optional end date YYYY-MM-DD (filters ledger files by filename date).")
    ap.add_argument(
        "--cv",
        action="store_true",
        help="Also compute leave-one-day-out CV Brier/logloss before/after calibration (read-only; prints only).",
    )
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="If no samples are found, write identity calibration with n=0 instead of exiting non-zero.",
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum samples required to fit calibration (default: 30). "
        "Below this, writes identity calibration.",
    )
    args = ap.parse_args()

    triples = _iter_pairs_from_daily(
        league=args.league,
        model_col=args.model_col,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if not triples:
        if not args.allow_empty:
            raise SystemExit(f"[error] no calibration samples found for league={args.league}")
        print(f"[warn] no calibration samples found for league={args.league}; writing identity calibration (n=0).")
        pairs = []
        n = 0
        scaler = PlattScaler(a=1.0, b=0.0)
        payload = {"a": 1.0, "b": 0.0, "n": 0}

        default_out = f"chimera_v2c/data/calibration_params_{args.league}_{args.model_col}.json"
        out_path = Path(args.out) if args.out else Path(default_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            snap_dir = Path("reports/calibration_snapshots")
            snap_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            snap_path = snap_dir / f"{out_path.name}.{ts}.bak"
            try:
                snap_path.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"[ok] snapshotted existing params -> {snap_path}")
            except Exception:
                print(f"[warn] failed to snapshot existing params at {out_path}; proceeding to overwrite.")

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[ok] wrote calibration params to {out_path}")
        return

    pairs: List[Tuple[float, int]] = [(p, y) for _, p, y in triples]
    n = len(pairs)

    if args.model_col.strip().lower() == "v2c":
        print(
            "[warn] calibrating `v2c` from daily ledgers; if calibration was enabled at plan time, "
            "v2c is already calibrated. Prefer calibrating from an uncalibrated plan source instead."
        )

    if n < args.min_samples:
        print(f"[warn] only {n} samples; writing identity calibration.")
        scaler = PlattScaler(a=1.0, b=0.0)
        payload = {"a": 1.0, "b": 0.0, "n": n}
    else:
        scaler = fit_platt(pairs)
        payload = {"a": scaler.a, "b": scaler.b, "n": n}
        print(f"[ok] fit calibration on n={n}: a={scaler.a:.4f}, b={scaler.b:.4f}")

    raw_brier = _mean_brier(pairs)
    cal_brier = _mean_brier(pairs, scaler=scaler)
    raw_ll = _mean_logloss(pairs)
    cal_ll = _mean_logloss(pairs, scaler=scaler)
    if raw_brier is not None and cal_brier is not None:
        print(f"[ok] in-sample brier: raw={raw_brier:.6f} calibrated={cal_brier:.6f}")
    if raw_ll is not None and cal_ll is not None:
        print(f"[ok] in-sample logloss: raw={raw_ll:.6f} calibrated={cal_ll:.6f}")

    if args.cv:
        cv_raw_brier, cv_cal_brier, cv_raw_ll, cv_cal_ll = _cv_by_date(triples, min_samples=args.min_samples)
        if cv_raw_brier is None:
            print("[warn] CV skipped (need at least 2 ledger dates with samples).")
        else:
            print(f"[ok] LODO-CV brier: raw={cv_raw_brier:.6f} calibrated={cv_cal_brier:.6f}")
            print(f"[ok] LODO-CV logloss: raw={cv_raw_ll:.6f} calibrated={cv_cal_ll:.6f}")

    default_out = f"chimera_v2c/data/calibration_params_{args.league}_{args.model_col}.json"
    out_path = Path(args.out) if args.out else Path(default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        snap_dir = Path("reports/calibration_snapshots")
        snap_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap_path = snap_dir / f"{out_path.name}.{ts}.bak"
        try:
            snap_path.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[ok] snapshotted existing params -> {snap_path}")
        except Exception:
            print(f"[warn] failed to snapshot existing params at {out_path}; proceeding to overwrite.")

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ok] wrote calibration params to {out_path}")


if __name__ == "__main__":
    main()
