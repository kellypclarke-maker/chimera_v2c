#!/usr/bin/env python
"""
Learn league-specific hybrid weights across calibrated models (v2c/grok/gemini/gpt).

Read-only on ledgers: consumes master_game_ledger.csv and writes weights/summaries.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from chimera_v2c.src.calibration import PlattScaler
from chimera_v2c.src.ledger.outcomes import parse_home_win


MASTER_PATH = Path("reports/master_ledger/master_game_ledger.csv")
THESIS_DIR = Path("reports/thesis_summaries")
CALIBRATION_DIR = Path("chimera_v2c/data")
MODEL_COLS = ["v2c", "grok", "gemini", "gpt"]


def parse_outcome(outcome: str) -> Optional[int]:
    hw = parse_home_win(outcome)
    if hw == 1.0:
        return 1
    if hw == 0.0:
        return 0
    return None


def clamp(p: float) -> float:
    return float(min(1 - 1e-6, max(1e-6, p)))


def load_calibrator(league: str, model: str) -> PlattScaler:
    candidates = [
        CALIBRATION_DIR / f"calibration_params_{league}_{model}.json",
    ]
    if model == "v2c":
        candidates.append(CALIBRATION_DIR / f"calibration_params_{league}.json")
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            return PlattScaler(a=float(data.get("a", 1.0)), b=float(data.get("b", 0.0)))
    return PlattScaler(a=1.0, b=0.0)


@dataclass
class GameRow:
    date: str
    y: int
    calibrated: Dict[str, float]
    available: List[str]


def load_dataset(league: str) -> Tuple[List[GameRow], str, str]:
    if not MASTER_PATH.exists():
        raise SystemExit(f"[error] missing master ledger: {MASTER_PATH}")
    df = pd.read_csv(MASTER_PATH)
    df = df[df["league"].str.lower() == league.lower()]
    df = df.dropna(subset=["actual_outcome"])
    calibrators = {m: load_calibrator(league.lower(), m) for m in MODEL_COLS}

    rows: List[GameRow] = []
    for _, row in df.iterrows():
        y = parse_outcome(row.get("actual_outcome"))
        if y is None:
            continue
        calibrated: Dict[str, float] = {}
        available: List[str] = []
        for col in MODEL_COLS:
            val = row.get(col)
            if pd.isna(val) or val == "" or val is None:
                continue
            try:
                p = float(val)
            except (TypeError, ValueError):
                continue
            calibrated[col] = calibrators[col].predict(clamp(p))
            available.append(col)
        if len(available) < 2:
            continue
        rows.append(GameRow(date=str(row.get("date")), y=int(y), calibrated=calibrated, available=available))

    if not rows:
        return [], "", ""
    start = min(r.date for r in rows)
    end = max(r.date for r in rows)
    return rows, start.replace("-", ""), end.replace("-", "")


def generate_simplex(models: Sequence[str], step: float) -> List[Dict[str, float]]:
    ticks = int(round(1 / step))
    weights: List[Dict[str, float]] = []
    for combo in itertools.product(range(ticks + 1), repeat=len(models)):
        if sum(combo) != ticks:
            continue
        w = {m: combo[i] * step for i, m in enumerate(models)}
        weights.append(w)
    return weights


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def evaluate_weights(rows: List[GameRow], weights: Dict[str, float], splits: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    preds_all = []
    y_all = []
    for train_idx, test_idx in splits:
        fold_preds = []
        fold_y = []
        for idx in test_idx:
            row = rows[idx]
            avail = [m for m in row.available if weights.get(m, 0) > 0]
            if not avail:
                continue
            total = sum(weights[m] for m in avail)
            if total <= 0:
                continue
            p = sum(weights[m] * row.calibrated[m] for m in avail) / total
            fold_preds.append(p)
            fold_y.append(row.y)
        if not fold_preds:
            continue
        preds_all.extend(fold_preds)
        y_all.extend(fold_y)
    if not preds_all:
        return float("inf")
    return brier(np.array(preds_all, dtype=float), np.array(y_all, dtype=float))


def build_splits(rows: List[GameRow], max_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = np.array([r.date for r in rows])
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return [(np.arange(len(rows)), np.arange(len(rows)))]
    n_splits = min(max_splits, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    splits = [(train, test) for train, test in gkf.split(np.zeros(len(rows)), np.zeros(len(rows)), groups=groups)]
    return splits


def single_model_brier(rows: List[GameRow], model: str, splits: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    preds = []
    ys = []
    for _, test_idx in splits:
        for idx in test_idx:
            row = rows[idx]
            if model not in row.calibrated:
                continue
            preds.append(row.calibrated[model])
            ys.append(row.y)
    if not preds:
        return float("inf")
    return brier(np.array(preds, dtype=float), np.array(ys, dtype=float))


def main() -> None:
    ap = argparse.ArgumentParser(description="Learn hybrid weights across calibrated models (per league).")
    ap.add_argument("--league", default="nba", help="League to evaluate (nba|nhl|nfl). Default: nba.")
    ap.add_argument("--step", type=float, default=0.05, help="Simplex step size for weights (default: 0.05).")
    args = ap.parse_args()

    rows, start, end = load_dataset(args.league)
    if not rows:
        raise SystemExit(f"[error] no rows with outcomes and >=2 model probs for league={args.league}")

    splits = build_splits(rows)
    weight_grid = generate_simplex(MODEL_COLS, step=args.step)

    best_weight: Optional[Dict[str, float]] = None
    best_brier = float("inf")
    for w in weight_grid:
        score = evaluate_weights(rows, w, splits)
        if score < best_brier:
            best_brier = score
            best_weight = w

    baseline = {m: single_model_brier(rows, m, splits) for m in MODEL_COLS}

    # Persist best weights
    THESIS_DIR.mkdir(parents=True, exist_ok=True)
    out_weights = {
        "league": args.league,
        "weights": best_weight,
        "cv_brier": best_brier,
        "baseline_brier": baseline,
        "n_games": len(rows),
        "n_dates": len(set(r.date for r in rows)),
        "start_date": start,
        "end_date": end,
        "step": args.step,
    }
    weights_path = CALIBRATION_DIR / f"hybrid_weights_{args.league}.json"
    weights_path.write_text(json.dumps(out_weights, indent=2), encoding="utf-8")

    summary_json = THESIS_DIR / f"hybrid_weights_{args.league}_{start}_{end}.json"
    summary_csv = THESIS_DIR / f"hybrid_weights_{args.league}_{start}_{end}.csv"
    summary_json.write_text(json.dumps(out_weights, indent=2), encoding="utf-8")
    rows_csv = [{"metric": "cv_brier", "value": best_brier}]
    for model, val in baseline.items():
        rows_csv.append({"metric": f"{model}_brier", "value": val})
    for model, val in (best_weight or {}).items():
        rows_csv.append({"metric": f"weight_{model}", "value": val})
    pd.DataFrame(rows_csv).to_csv(summary_csv, index=False)

    print(f"Hybrid weights learned for {args.league.upper()} ({start}..{end}, n={len(rows)})")
    print(f"Best CV Brier: {best_brier:.4f}")
    print("Weights:")
    for m, w in (best_weight or {}).items():
        print(f"  {m}: {w:.2f}")
    print("Baselines (per-model Brier):")
    for m, val in baseline.items():
        if val == float("inf"):
            print(f"  {m}: unavailable")
        else:
            print(f"  {m}: {val:.4f}")
    print(f"Saved weights to {weights_path}")
    print(f"Summary -> {summary_json}, {summary_csv}")


if __name__ == "__main__":
    main()
