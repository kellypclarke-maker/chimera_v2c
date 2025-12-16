#!/usr/bin/env python
"""
Online (game-by-game) walkthrough for the Grok↔Kalshi-mid hybrid.

This tool answers: "If we start at n=1 and re-learn after each game, does the
Grok-mid hybrid get better over time?"

Method
------
We build a chronological list of graded games where:
  - actual outcome is final (home_win in {0,1})
  - kalshi_mid is present
  - grok is present (not NR)

For each game i in order:
  - Train on the prior games (0..i-1) only (no leakage).
  - Fit Platt scaling on Grok if there are at least --min-train-samples prior games,
    else use identity calibration.
  - Select alpha in [0,1] (grid) that minimizes mean Brier on the prior games:
      p_hybrid = p_mid + alpha * (Platt(p_grok) - p_mid)
  - Predict on game i and record per-game + cumulative metrics.

Notes
-----
- This is an "online learning curve" diagnostic. The more robust evaluation for
  go/no-go is the per-day walk-forward tools under reports/thesis_summaries/.
- PnL here is "mid-trade" PnL: if p_model != p_mid we buy the side implied by
  the model vs the mid, at the mid price (no fees/spread modeled).

Safety: read-only on daily ledgers.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chimera_v2c.src.calibration import PlattScaler, fit_platt
from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.offset_calibration import clamp_prob
from chimera_v2c.src.rulebook_quadrants import pnl_buy_away, pnl_buy_home


DEFAULT_OUT_DIR = Path("reports/thesis_summaries")


@dataclass(frozen=True)
class Sample:
    date: str
    league: str
    matchup: str
    p_mid: float
    p_grok: float
    y: int


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


def _brier(p: float, y: int) -> float:
    return (float(p) - float(y)) ** 2


def _mid_trade_pnl(*, p_mid: float, p_model: float, y: int) -> Tuple[int, float]:
    """
    PnL for 1 contract if we trade at the mid based on model vs market:
      - if p_model > p_mid: buy HOME YES @ p_mid
      - if p_model < p_mid: buy AWAY YES @ (1 - p_mid)
      - if equal: no bet

    Returns (bets, pnl).
    """
    pm = float(p_mid)
    p = float(p_model)
    if p == pm:
        return 0, 0.0
    if p > pm:
        return 1, float(pnl_buy_home(p_mid=pm, home_win=float(y)))
    return 1, float(pnl_buy_away(p_mid=pm, home_win=float(y)))


def _fit_platt_or_identity(train: Sequence[Sample], *, min_train_samples: int) -> PlattScaler:
    if len(train) < int(min_train_samples):
        return PlattScaler(a=1.0, b=0.0)
    pairs = [(s.p_grok, s.y) for s in train]
    return fit_platt(pairs)


def _select_alpha_in_sample(
    train: Sequence[Sample],
    *,
    scaler: PlattScaler,
    alpha_grid: Sequence[float],
) -> float:
    if not train:
        return 0.0

    # Precompute calibrated Grok on train once.
    cal: List[Tuple[float, float, int]] = []
    for s in train:
        p_mid = float(s.p_mid)
        p_grok_cal = clamp_prob(scaler.predict(float(s.p_grok)))
        cal.append((p_mid, p_grok_cal, int(s.y)))

    best_alpha = 0.0
    best = float("inf")
    for a in alpha_grid:
        alpha = float(a)
        se = 0.0
        for p_mid, p_grok_cal, y in cal:
            p_h = clamp_prob(p_mid + alpha * (p_grok_cal - p_mid))
            se += _brier(p_h, y)
        mse = se / len(cal)
        if mse < best - 1e-12:
            best = mse
            best_alpha = alpha
        elif abs(mse - best) <= 1e-12:
            # Deterministic tie-break: prefer smaller alpha (more conservative).
            best_alpha = min(best_alpha, alpha)
    return float(best_alpha)


def _select_alpha_date_cv(
    train: Sequence[Sample],
    *,
    alpha_grid: Sequence[float],
    min_platt_samples: int,
) -> float:
    """
    Leave-one-train-date-out CV for alpha, fitting Platt only on the inner-train fold.

    This mirrors the alpha selection logic used in the per-day walk-forward tools,
    but applied here on an expanding, game-by-game training set.
    """
    if not train:
        return 0.0

    by_date: Dict[str, List[Sample]] = {}
    for s in train:
        by_date.setdefault(s.date, []).append(s)
    dates = sorted(by_date.keys())
    if len(dates) < 2:
        return 0.0

    sq_by_alpha = {float(a): 0.0 for a in alpha_grid}
    n_by_alpha = {float(a): 0 for a in alpha_grid}

    for holdout in dates:
        inner_pairs = [(s.p_grok, s.y) for s in train if s.date != holdout]
        if len(inner_pairs) < int(min_platt_samples):
            scaler = PlattScaler(a=1.0, b=0.0)
        else:
            scaler = fit_platt(inner_pairs)

        for s in by_date[holdout]:
            p_mid = float(s.p_mid)
            p_grok_cal = clamp_prob(scaler.predict(float(s.p_grok)))
            for a in alpha_grid:
                alpha = float(a)
                p_h = clamp_prob(p_mid + alpha * (p_grok_cal - p_mid))
                sq_by_alpha[alpha] += _brier(p_h, s.y)
                n_by_alpha[alpha] += 1

    best_alpha = 0.0
    best = float("inf")
    for a in alpha_grid:
        alpha = float(a)
        n = n_by_alpha[alpha]
        if n <= 0:
            continue
        mse = sq_by_alpha[alpha] / n
        if mse < best - 1e-12:
            best = mse
            best_alpha = alpha
        elif abs(mse - best) <= 1e-12:
            best_alpha = min(best_alpha, alpha)
    return float(best_alpha)


def iter_samples(games: Iterable[GameRow]) -> List[Sample]:
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
                date=g.date.strftime("%Y-%m-%d"),
                league=g.league,
                matchup=g.matchup,
                p_mid=clamp_prob(float(g.kalshi_mid)),
                p_grok=clamp_prob(float(p_grok)),
                y=int(g.home_win),
            )
        )
    return out


def online_walkthrough(
    samples: Sequence[Sample],
    *,
    min_train_samples: int = 30,
    alpha_step: float = 0.02,
    alpha_mode: str = "insample",
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    Return (rows, summary) for an online, game-by-game learning walkthrough.
    """
    if not samples:
        return [], {"n": 0}

    alpha_grid = _alpha_grid(alpha_step)

    cum_brier = {"kalshi_mid": 0.0, "grok_raw": 0.0, "grok_platt": 0.0, "grok_mid_hybrid": 0.0}
    cum_pnl = {"grok_raw": 0.0, "grok_platt": 0.0, "grok_mid_hybrid": 0.0}
    cum_bets = {"grok_raw": 0, "grok_platt": 0, "grok_mid_hybrid": 0}

    rows: List[Dict[str, object]] = []
    alpha_mode_norm = (alpha_mode or "").strip().lower()
    for i, s in enumerate(samples):
        train = list(samples[:i])
        trained = len(train) >= int(min_train_samples)
        scaler = _fit_platt_or_identity(train, min_train_samples=min_train_samples)
        if alpha_mode_norm == "date_cv":
            alpha = _select_alpha_date_cv(train, alpha_grid=alpha_grid, min_platt_samples=min_train_samples)
        elif alpha_mode_norm == "insample":
            alpha = _select_alpha_in_sample(train, scaler=scaler, alpha_grid=alpha_grid)
        else:
            raise ValueError("alpha_mode must be one of: insample, date_cv")

        p_mid = float(s.p_mid)
        p_grok_raw = float(s.p_grok)
        p_grok_platt = clamp_prob(scaler.predict(p_grok_raw))
        p_hybrid = clamp_prob(p_mid + float(alpha) * (p_grok_platt - p_mid))

        b_mid = _brier(p_mid, s.y)
        b_raw = _brier(p_grok_raw, s.y)
        b_platt = _brier(p_grok_platt, s.y)
        b_hybrid = _brier(p_hybrid, s.y)

        cum_brier["kalshi_mid"] += b_mid
        cum_brier["grok_raw"] += b_raw
        cum_brier["grok_platt"] += b_platt
        cum_brier["grok_mid_hybrid"] += b_hybrid

        bets_raw, pnl_raw = _mid_trade_pnl(p_mid=p_mid, p_model=p_grok_raw, y=s.y)
        bets_platt, pnl_platt = _mid_trade_pnl(p_mid=p_mid, p_model=p_grok_platt, y=s.y)
        bets_hybrid, pnl_hybrid = _mid_trade_pnl(p_mid=p_mid, p_model=p_hybrid, y=s.y)
        cum_bets["grok_raw"] += bets_raw
        cum_bets["grok_platt"] += bets_platt
        cum_bets["grok_mid_hybrid"] += bets_hybrid
        cum_pnl["grok_raw"] += pnl_raw
        cum_pnl["grok_platt"] += pnl_platt
        cum_pnl["grok_mid_hybrid"] += pnl_hybrid

        n_seen = i + 1
        rows.append(
            {
                "n_seen": n_seen,
                "n_train": i,
                "trained": str(bool(trained)),
                "date": s.date,
                "league": s.league,
                "matchup": s.matchup,
                "y_home_win": int(s.y),
                "p_mid": f"{p_mid:.6f}",
                "p_grok_raw": f"{p_grok_raw:.6f}",
                "platt_a": f"{float(scaler.a):.6f}",
                "platt_b": f"{float(scaler.b):.6f}",
                "p_grok_platt": f"{p_grok_platt:.6f}",
                "alpha_mode": alpha_mode_norm,
                "alpha": f"{float(alpha):.3f}",
                "p_hybrid": f"{p_hybrid:.6f}",
                "brier_mid": f"{b_mid:.6f}",
                "brier_grok_raw": f"{b_raw:.6f}",
                "brier_grok_platt": f"{b_platt:.6f}",
                "brier_hybrid": f"{b_hybrid:.6f}",
                "cum_mean_brier_mid": f"{(cum_brier['kalshi_mid'] / n_seen):.6f}",
                "cum_mean_brier_grok_raw": f"{(cum_brier['grok_raw'] / n_seen):.6f}",
                "cum_mean_brier_grok_platt": f"{(cum_brier['grok_platt'] / n_seen):.6f}",
                "cum_mean_brier_hybrid": f"{(cum_brier['grok_mid_hybrid'] / n_seen):.6f}",
                "cum_mean_brier_hybrid_minus_mid": f"{((cum_brier['grok_mid_hybrid'] - cum_brier['kalshi_mid']) / n_seen):.6f}",
                "pnl_mid_trade_grok_raw": f"{pnl_raw:.6f}" if bets_raw else "",
                "pnl_mid_trade_grok_platt": f"{pnl_platt:.6f}" if bets_platt else "",
                "pnl_mid_trade_hybrid": f"{pnl_hybrid:.6f}" if bets_hybrid else "",
                "cum_bets_grok_raw": cum_bets["grok_raw"],
                "cum_bets_grok_platt": cum_bets["grok_platt"],
                "cum_bets_hybrid": cum_bets["grok_mid_hybrid"],
                "cum_total_pnl_grok_raw": f"{cum_pnl['grok_raw']:.6f}",
                "cum_total_pnl_grok_platt": f"{cum_pnl['grok_platt']:.6f}",
                "cum_total_pnl_hybrid": f"{cum_pnl['grok_mid_hybrid']:.6f}",
                "cum_avg_pnl_per_bet_grok_raw": f"{(cum_pnl['grok_raw'] / cum_bets['grok_raw']):.6f}"
                if cum_bets["grok_raw"]
                else "",
                "cum_avg_pnl_per_bet_grok_platt": f"{(cum_pnl['grok_platt'] / cum_bets['grok_platt']):.6f}"
                if cum_bets["grok_platt"]
                else "",
                "cum_avg_pnl_per_bet_hybrid": f"{(cum_pnl['grok_mid_hybrid'] / cum_bets['grok_mid_hybrid']):.6f}"
                if cum_bets["grok_mid_hybrid"]
                else "",
            }
        )

    summary: Dict[str, object] = {
        "n": len(samples),
        "mean_brier_mid": cum_brier["kalshi_mid"] / len(samples),
        "mean_brier_grok_raw": cum_brier["grok_raw"] / len(samples),
        "mean_brier_grok_platt": cum_brier["grok_platt"] / len(samples),
        "mean_brier_hybrid": cum_brier["grok_mid_hybrid"] / len(samples),
        "delta_brier_hybrid_minus_mid": (cum_brier["grok_mid_hybrid"] - cum_brier["kalshi_mid"]) / len(samples),
        "bets_grok_raw": cum_bets["grok_raw"],
        "bets_grok_platt": cum_bets["grok_platt"],
        "bets_hybrid": cum_bets["grok_mid_hybrid"],
        "total_pnl_grok_raw": cum_pnl["grok_raw"],
        "total_pnl_grok_platt": cum_pnl["grok_platt"],
        "total_pnl_hybrid": cum_pnl["grok_mid_hybrid"],
        "avg_pnl_per_bet_hybrid": (cum_pnl["grok_mid_hybrid"] / cum_bets["grok_mid_hybrid"]) if cum_bets["grok_mid_hybrid"] else None,
    }
    return rows, summary


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser(description="Online (game-by-game) walkthrough for Grok-mid hybrid (read-only).")
    ap.add_argument("--start-date", default="", help="YYYY-MM-DD (inclusive). Omit for earliest available.")
    ap.add_argument("--end-date", default="", help="YYYY-MM-DD (inclusive). Omit for latest available.")
    ap.add_argument("--league", default="all", help="nba|nhl|nfl|all (default: all).")
    ap.add_argument("--min-train-samples", type=int, default=30, help="Min prior games to fit Platt (default: 30).")
    ap.add_argument("--alpha-step", type=float, default=0.02, help="Alpha grid step in [0,1] (default: 0.02).")
    ap.add_argument("--alpha-mode", default="insample", help="Alpha selection: insample|date_cv (default: insample).")
    ap.add_argument("--out", default="", help="Optional CSV output path. Default under reports/thesis_summaries/.")
    ap.add_argument("--no-write", action="store_true", help="Compute and print only; do not write CSV.")
    args = ap.parse_args()

    league_arg = (args.league or "").strip().lower()
    league_filter = None if league_arg in {"", "all"} else league_arg
    league_key = league_filter or "all"

    games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        league_filter=league_filter,
        models=["grok"],
    )
    if not games:
        raise SystemExit("[error] no games found for the selected window.")

    samples = iter_samples(games)
    if not samples:
        raise SystemExit("[error] no graded games with grok+kalshi_mid in the selected window.")

    rows, summary = online_walkthrough(
        samples,
        min_train_samples=int(args.min_train_samples),
        alpha_step=float(args.alpha_step),
        alpha_mode=str(args.alpha_mode),
    )

    if args.no_write:
        print(summary)
        return

    if args.out:
        out_path = Path(args.out)
    else:
        start = (samples[0].date if samples else "").replace("-", "")
        end = (samples[-1].date if samples else "").replace("-", "")
        out_path = DEFAULT_OUT_DIR / f"grok_mid_hybrid_online_walkthrough_{league_key}_{start}_{end}.csv"

    _write_csv(out_path, rows)

    print("\n=== Grok-mid hybrid online walkthrough ===")
    print(f"league={league_key} games={summary['n']}")
    print(
        "mean_brier: "
        f"mid={summary['mean_brier_mid']:.6f} "
        f"grok_raw={summary['mean_brier_grok_raw']:.6f} "
        f"grok_platt={summary['mean_brier_grok_platt']:.6f} "
        f"hybrid={summary['mean_brier_hybrid']:.6f} "
        f"delta(hybrid-mid)={summary['delta_brier_hybrid_minus_mid']:.6f}"
    )
    print(
        "mid-trade pnl (no fees): "
        f"hybrid bets={summary['bets_hybrid']} total_pnl={summary['total_pnl_hybrid']:.3f} "
        f"avg_pnl_per_bet={(summary['avg_pnl_per_bet_hybrid'] if summary['avg_pnl_per_bet_hybrid'] is not None else float('nan')):.3f}"
    )
    print(f"[ok] wrote -> {out_path}")


if __name__ == "__main__":
    main()
