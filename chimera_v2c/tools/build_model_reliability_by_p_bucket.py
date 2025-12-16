from __future__ import annotations

"""
Build reliability / Brier / EV-by-bucket summary from daily ledgers.

Reads:
  reports/daily_ledgers/*_daily_game_ledger.csv

Writes:
  reports/daily_ledgers/model_reliability_by_p_bucket.csv

For each model (v2c, gemini, grok, gpt, kalshi_mid, market_proxy, moneypuck) and each predicted
home-win probability bucket, this tool computes:
  - n: number of graded (non-push) games in the bucket
  - avg_p: mean predicted probability in the bucket
  - actual_rate: empirical home win rate in the bucket
  - brier: mean squared error (Brier score) in the bucket
  - bets: number of games where the model implied a trade vs Kalshi mid
  - total_pnl: total realized PnL vs Kalshi mid for those bets
  - avg_pnl: average PnL per bet vs Kalshi mid

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/build_model_reliability_by_p_bucket.py

You can override bucket width or minimum bucket size, for example:
  PYTHONPATH=. python chimera_v2c/tools/build_model_reliability_by_p_bucket.py \\
      --bucket-width 0.1 --min-n 10
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games


MODELS = ["v2c", "gemini", "grok", "gpt", "kalshi_mid", "market_proxy", "moneypuck"]


def prob_bucket(p: float, bucket_width: float = 0.1) -> str:
    """
    Map a probability p in [0,1] to a bucket label like [0.2,0.3).
    The top bucket is closed at 1.0: [0.9,1.0].
    """
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    # Put 1.0 exactly into the last bucket.
    if p >= 1.0 - 1e-9:
        hi = 1.0
        lo = hi - bucket_width
        if lo < 0.0:
            lo = 0.0
        return f"[{lo:.1f},{hi:.1f}]"
    idx = int(p // bucket_width)
    lo = idx * bucket_width
    hi = lo + bucket_width
    if hi > 1.0:
        hi = 1.0
    return f"[{lo:.1f},{hi:.1f})"


def compute_reliability_by_bucket(
    games: Iterable[GameRow],
    models: List[str],
    bucket_width: float = 0.1,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute per-model, per-bucket reliability and EV vs Kalshi mid.

    Returns:
      stats[model][bucket] = {
        "n": count,
        "sum_p": sum of probabilities,
        "sum_y": sum of home_win outcomes,
        "sum_sq": sum of squared errors,
        "bets": number of EV bets vs kalshi_mid,
        "pnl_sum": total pnl vs kalshi_mid,
      }
    """
    stats: Dict[str, Dict[str, Dict[str, float]]] = {
        m: defaultdict(lambda: {"n": 0, "sum_p": 0.0, "sum_y": 0.0, "sum_sq": 0.0, "bets": 0, "pnl_sum": 0.0})
        for m in models
    }

    for g in games:
        y = g.home_win
        if y is None or y == 0.5:
            # Skip unresolved or push outcomes.
            continue

        # Access probabilities from GameRow.probs; kalshi_mid is kept under that
        # key when load_games is called with models including "kalshi_mid".
        kalshi = g.probs.get("kalshi_mid")

        for m in models:
            p = g.probs.get(m)
            if p is None:
                continue
            b = prob_bucket(p, bucket_width=bucket_width)
            s = stats[m][b]
            s["n"] += 1
            s["sum_p"] += p
            s["sum_y"] += y
            s["sum_sq"] += (p - y) ** 2

            # EV vs Kalshi mid (skip for kalshi_mid itself or missing market).
            if m != "kalshi_mid" and kalshi is not None and p != kalshi:
                price = float(kalshi)
                price = max(0.01, min(0.99, price))
                if p > price:
                    pnl = (1.0 - price) if y == 1.0 else -price
                else:
                    pnl = price if y == 0.0 else -(1.0 - price)
                s["bets"] += 1
                s["pnl_sum"] += pnl

    return stats


def write_reliability_csv(
    out_path: Path,
    stats: Dict[str, Dict[str, Dict[str, float]]],
    models: List[str],
    min_n: int,
) -> List[Dict[str, object]]:
    """
    Write a long-form CSV with one row per (bucket, model).

    Columns:
      bucket,model,n,avg_p,actual_rate,brier,bets,total_pnl,avg_pnl
    """
    buckets = sorted({b for m in models for b in stats.get(m, {}).keys()})
    rows: List[Dict[str, object]] = []
    for b in buckets:
        for m in models:
            s = stats[m].get(b)
            if not s or s["n"] < min_n:
                continue
            n = int(s["n"])
            avg_p = s["sum_p"] / n if n else None
            actual_rate = s["sum_y"] / n if n else None
            brier = s["sum_sq"] / n if n else None
            bets = int(s["bets"])
            total_pnl = s["pnl_sum"]
            avg_pnl = total_pnl / bets if bets else None
            rows.append(
                {
                    "bucket": b,
                    "model": m,
                    "n": n,
                    "avg_p": avg_p,
                    "actual_rate": actual_rate,
                    "brier": brier,
                    "bets": bets,
                    "total_pnl": total_pnl,
                    "avg_pnl": avg_pnl,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bucket",
        "model",
        "n",
        "avg_p",
        "actual_rate",
        "brier",
        "bets",
        "total_pnl",
        "avg_pnl",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def print_reliability_table(
    rows: List[Dict[str, object]],
    models: List[str],
) -> None:
    """
    Print a compact box table showing, per bucket and model:
      - n (games)
      - avg_p (mean predicted probability)
      - actual_rate (empirical win rate)
    """
    # Reorganize rows as bucket -> model -> stats.
    by_bucket: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for r in rows:
        b = str(r["bucket"])
        m = str(r["model"])
        by_bucket[b][m] = r

    buckets = sorted(by_bucket.keys())

    header: List[str] = ["Bucket"]
    for m in models:
        header.extend([f"{m}:n", f"{m}:p", f"{m}:act"])

    # Compute column widths.
    col_widths = [max(len(h), 8) for h in header]

    def update_widths(values: List[object]) -> None:
        for i, v in enumerate(values):
            col_widths[i] = max(col_widths[i], len(str(v)))

    update_widths(header)
    for b in buckets:
        row_vals: List[object] = [b]
        data = by_bucket[b]
        for m in models:
            r = data.get(m)
            if not r:
                row_vals.extend([0, "", ""])
                continue
            row_vals.append(r["n"])
            ap = r["avg_p"]
            ar = r["actual_rate"]
            row_vals.append("" if ap is None else f"{ap:.3f}")
            row_vals.append("" if ar is None else f"{ar:.3f}")
        update_widths(row_vals)

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def fmt_row(values: List[object]) -> str:
        cells = []
        for v, w in zip(values, col_widths):
            cells.append(" " + str(v).ljust(w) + " ")
        return "|" + "|".join(cells) + "|"

    print("\nReliability by predicted p bucket (all leagues, all days):")
    print(sep)
    print(fmt_row(header))
    print(sep)
    for b in buckets:
        values: List[object] = [b]
        data = by_bucket[b]
        for m in models:
            r = data.get(m)
            if not r:
                values.extend([0, "", ""])
                continue
            values.append(r["n"])
            ap = r["avg_p"]
            ar = r["actual_rate"]
            values.append("" if ap is None else f"{ap:.3f}")
            values.append("" if ar is None else f"{ar:.3f}")
        print(fmt_row(values))
    print(sep)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build reliability/Brier/EV-by-bucket summary from daily ledgers (read-only on ledgers)."
    )
    ap.add_argument(
        "--daily-dir",
        default=str(LEDGER_DIR),
        help="Directory containing *_daily_game_ledger.csv files "
        f"(default: {LEDGER_DIR})",
    )
    ap.add_argument(
        "--bucket-width",
        type=float,
        default=0.1,
        help="Probability bucket width for reliability curves (default: 0.1).",
    )
    ap.add_argument(
        "--min-n",
        type=int,
        default=10,
        help="Minimum graded games per (model,bucket) to include in CSV/table (default: 10).",
    )
    ap.add_argument(
        "--out",
        default="reports/daily_ledgers/model_reliability_by_p_bucket.csv",
        help="Output CSV path (default: reports/daily_ledgers/model_reliability_by_p_bucket.csv).",
    )
    args = ap.parse_args()

    daily_dir = Path(args.daily_dir)
    if not daily_dir.exists():
        raise SystemExit(f"[error] daily ledger directory not found: {daily_dir}")

    games = load_games(daily_dir=daily_dir, models=MODELS)
    if not games:
        raise SystemExit("[error] no games loaded from daily ledgers")

    stats = compute_reliability_by_bucket(games, models=MODELS, bucket_width=args.bucket_width)
    rows = write_reliability_csv(Path(args.out), stats, models=MODELS, min_n=args.min_n)
    if not rows:
        print("[info] no buckets met the min-n threshold; CSV is header-only.")
        return
    print(f"[info] wrote {len(rows)} rows -> {args.out}")
    print_reliability_table(rows, models=MODELS)


if __name__ == "__main__":
    main()
