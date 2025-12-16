"""
Find per-model "home-rich fade" sweet spots vs Kalshi.

We restrict to games where Kalshi favors HOME:
  - p_mid = kalshi_mid >= 0.5

For a given model m with probability p_m, define edge:
  edge_m = p_m - p_mid

"Home rich" at threshold t means:
  edge_m <= -t

We then backtest the strategy "fade home" (bet away) at price p_mid, and
report realized EV vs Kalshi mid and Brier for the model on the same subset.

This tool is read-only on daily ledgers; it writes derived CSV artifacts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games


DEFAULT_MODELS = ["v2c", "gemini", "grok", "gpt", "market_proxy"]
COMPARE_MODELS = ["v2c", "gemini", "grok", "gpt", "market_proxy", "kalshi_mid"]


@dataclass
class SpotStats:
    bets: int = 0
    total_pnl: float = 0.0
    brier_n: int = 0
    brier_sum: float = 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.bets if self.bets else 0.0

    @property
    def brier(self) -> Optional[float]:
        return self.brier_sum / self.brier_n if self.brier_n else None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Analyze home-rich fade sweet spots vs Kalshi by model (read-only)."
    )
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD (inclusive).")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive).")
    ap.add_argument("--league", help="Optional league filter (nba|nhl|nfl).")
    ap.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model columns to include (default: v2c gemini grok gpt market_proxy).",
    )
    ap.add_argument(
        "--min-edge",
        type=float,
        default=0.01,
        help="Minimum edge threshold to scan (default: 0.01).",
    )
    ap.add_argument(
        "--max-edge",
        type=float,
        default=0.10,
        help="Maximum edge threshold to scan (default: 0.10).",
    )
    ap.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Edge threshold step size (default: 0.01).",
    )
    ap.add_argument(
        "--min-bets",
        type=int,
        default=10,
        help="Minimum bets required to consider a threshold as a 'sweet spot' (default: 10).",
    )
    ap.add_argument(
        "--out-dir",
        default="reports/thesis_summaries",
        help="Directory to write derived CSVs (default: reports/thesis_summaries).",
    )
    return ap.parse_args()


def _pnl_fade_home(p_mid: float, home_win: float) -> float:
    """
    PnL (1 unit) for betting AWAY at price p_mid.
      - If home loses: +p_mid
      - If home wins:  -(1 - p_mid)
    """
    p_mid = max(0.01, min(0.99, float(p_mid)))
    return p_mid if home_win == 0.0 else -(1.0 - p_mid)


def iter_graded_with_mid(games: Iterable[GameRow]) -> Iterable[GameRow]:
    for g in games:
        if g.home_win is None or g.home_win == 0.5:
            continue
        if g.kalshi_mid is None:
            continue
        yield g


def edge_thresholds(min_edge: float, max_edge: float, step: float) -> List[float]:
    if step <= 0:
        raise SystemExit("[error] --step must be > 0")
    if max_edge < min_edge:
        raise SystemExit("[error] --max-edge must be >= --min-edge")
    out = []
    cur = min_edge
    # Avoid float drift; round to 3 decimals (edges are percent-ish).
    while cur <= max_edge + 1e-9:
        out.append(round(cur, 3))
        cur += step
    return out


def compute_curve(
    games: Iterable[GameRow],
    model: str,
    thresholds: List[float],
) -> Dict[float, SpotStats]:
    stats: Dict[float, SpotStats] = {t: SpotStats() for t in thresholds}
    for g in iter_graded_with_mid(games):
        p_mid = float(g.kalshi_mid)
        if p_mid < 0.5:
            continue
        p_model = g.probs.get(model)
        if p_model is None:
            continue
        edge = p_model - p_mid
        for t in thresholds:
            if edge <= -t:
                s = stats[t]
                s.bets += 1
                s.total_pnl += _pnl_fade_home(p_mid=p_mid, home_win=float(g.home_win))
                s.brier_n += 1
                s.brier_sum += (p_model - float(g.home_win)) ** 2
    return stats


def subset_for_threshold(
    games: Iterable[GameRow],
    bucket_model: str,
    threshold: float,
) -> List[GameRow]:
    subset: List[GameRow] = []
    for g in iter_graded_with_mid(games):
        p_mid = float(g.kalshi_mid)
        if p_mid < 0.5:
            continue
        p_bucket = g.probs.get(bucket_model)
        if p_bucket is None:
            continue
        if (p_bucket - p_mid) <= -threshold:
            subset.append(g)
    return subset


def brier_on_subset(
    subset: Iterable[GameRow],
    model: str,
) -> Tuple[Optional[float], int]:
    """
    Compute Brier for a model over a subset.
    - For kalshi_mid: uses GameRow.kalshi_mid (home probability)
    - For others: uses GameRow.probs[model]
    Returns (mean_brier, n).
    """
    n = 0
    sum_sq = 0.0
    for g in subset:
        if g.home_win is None or g.home_win == 0.5:
            continue
        if model == "kalshi_mid":
            p = g.kalshi_mid
            if p is None:
                continue
        else:
            p = g.probs.get(model)
            if p is None:
                continue
        y = float(g.home_win)
        n += 1
        sum_sq += (float(p) - y) ** 2
    if n == 0:
        return None, 0
    return sum_sq / n, n


def write_brier_compare_csv(
    out_path: Path,
    bucket_models: List[str],
    sweet: Dict[str, Tuple[float, SpotStats]],
    games: List[GameRow],
    meta: Dict[str, str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "bucket_model",
        "sweet_edge_threshold",
        "subset_games",
        "fade_away_bets",
        "fade_away_total_pnl",
        "fade_away_avg_pnl",
    ]
    for m in COMPARE_MODELS:
        fieldnames.extend([f"{m}_brier", f"{m}_brier_n"])
    fieldnames.extend(
        [
            "window_start",
            "window_end",
            "league",
            "min_bets",
            "min_edge",
            "max_edge",
            "step",
        ]
    )

    rows: List[Dict[str, object]] = []
    for bucket_model in bucket_models:
        if bucket_model not in sweet:
            continue
        threshold, spot = sweet[bucket_model]
        subset = subset_for_threshold(games, bucket_model=bucket_model, threshold=threshold)
        row: Dict[str, object] = {
            "bucket_model": bucket_model,
            "sweet_edge_threshold": f"{threshold:.3f}",
            "subset_games": len(subset),
            "fade_away_bets": spot.bets,
            "fade_away_total_pnl": f"{spot.total_pnl:.6f}",
            "fade_away_avg_pnl": f"{spot.avg_pnl:.6f}",
        }
        for m in COMPARE_MODELS:
            br, n = brier_on_subset(subset, model=m)
            row[f"{m}_brier"] = "" if br is None else f"{br:.6f}"
            row[f"{m}_brier_n"] = n
        row.update(meta)
        rows.append(row)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def pick_sweet_spot(curve: Dict[float, SpotStats], min_bets: int) -> Optional[Tuple[float, SpotStats]]:
    candidates = [(t, s) for t, s in curve.items() if s.bets >= min_bets]
    if not candidates:
        return None
    # Maximize avg_pnl; break ties by more bets.
    candidates.sort(key=lambda x: (x[1].avg_pnl, x[1].bets), reverse=True)
    return candidates[0]


def write_curve_csv(out_path: Path, model: str, curve: Dict[float, SpotStats]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "edge_threshold", "bets", "total_pnl", "avg_pnl_per_bet", "brier", "brier_n"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in sorted(curve.keys()):
            s = curve[t]
            w.writerow(
                {
                    "model": model,
                    "edge_threshold": f"{t:.3f}",
                    "bets": s.bets,
                    "total_pnl": f"{s.total_pnl:.6f}",
                    "avg_pnl_per_bet": f"{s.avg_pnl:.6f}",
                    "brier": "" if s.brier is None else f"{s.brier:.6f}",
                    "brier_n": s.brier_n,
                }
            )


def write_sweet_spots_csv(
    out_path: Path,
    models: List[str],
    sweet: Dict[str, Tuple[float, SpotStats]],
    meta: Dict[str, str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "sweet_edge_threshold",
        "bets",
        "total_pnl",
        "avg_pnl_per_bet",
        "brier",
        "brier_n",
        "window_start",
        "window_end",
        "league",
        "min_bets",
        "min_edge",
        "max_edge",
        "step",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in models:
            if m not in sweet:
                continue
            t, s = sweet[m]
            w.writerow(
                {
                    "model": m,
                    "sweet_edge_threshold": f"{t:.3f}",
                    "bets": s.bets,
                    "total_pnl": f"{s.total_pnl:.6f}",
                    "avg_pnl_per_bet": f"{s.avg_pnl:.6f}",
                    "brier": "" if s.brier is None else f"{s.brier:.6f}",
                    "brier_n": s.brier_n,
                    **meta,
                }
            )


def main() -> None:
    args = parse_args()
    league_filter = args.league.lower() if args.league else None
    models: List[str] = list(args.models)

    games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=league_filter,
        models=models + ["kalshi_mid"],
    )
    if not games:
        raise SystemExit("[error] no games found for given filters")

    thresholds = edge_thresholds(args.min_edge, args.max_edge, args.step)
    out_dir = Path(args.out_dir)

    meta = {
        "window_start": args.start_date,
        "window_end": args.end_date,
        "league": league_filter or "all",
        "min_bets": str(args.min_bets),
        "min_edge": f"{args.min_edge:.3f}",
        "max_edge": f"{args.max_edge:.3f}",
        "step": f"{args.step:.3f}",
    }

    sweet: Dict[str, Tuple[float, SpotStats]] = {}
    for m in models:
        curve = compute_curve(games, model=m, thresholds=thresholds)
        pick = pick_sweet_spot(curve, min_bets=args.min_bets)
        if pick is not None:
            sweet[m] = pick

        curve_path = out_dir / (
            f"home_rich_curve_{meta['window_start']}_{meta['window_end']}_{meta['league']}"
            f"_minbets{meta['min_bets']}_{m}.csv"
        )
        write_curve_csv(curve_path, model=m, curve=curve)

    sweet_path = out_dir / (
        f"home_rich_sweet_spots_{meta['window_start']}_{meta['window_end']}_{meta['league']}"
        f"_minbets{meta['min_bets']}.csv"
    )
    write_sweet_spots_csv(sweet_path, models=models, sweet=sweet, meta=meta)

    compare_path = out_dir / (
        f"home_rich_brier_compare_{meta['window_start']}_{meta['window_end']}_{meta['league']}"
        f"_minbets{meta['min_bets']}.csv"
    )
    write_brier_compare_csv(
        compare_path,
        bucket_models=models,
        sweet=sweet,
        games=games,
        meta=meta,
    )

    print(f"[info] wrote sweet spots -> {sweet_path}")
    print(f"[info] wrote brier compare -> {compare_path}")
    for m in models:
        if m not in sweet:
            print(f"{m}: no threshold met min_bets={args.min_bets}")
            continue
        t, s = sweet[m]
        print(
            f"{m}: sweet_edge>={t:.3f} | bets={s.bets} | avg_pnl={s.avg_pnl:.3f} | "
            f"brier={'' if s.brier is None else f'{s.brier:.3f}'}"
        )


if __name__ == "__main__":
    main()
