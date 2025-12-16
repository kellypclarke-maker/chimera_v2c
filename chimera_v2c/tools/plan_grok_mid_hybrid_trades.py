#!/usr/bin/env python
"""
Paper trade sheet: Grok↔Kalshi-mid hybrid + learned A/B/C/D threshold policy.

Goal
----
Produce trade recommendations for a target ledger date using ONLY information
from dates < target date (no leakage).

Method (single league filter or combined "all")
---------------------------------------------
Training set (dates < target):
  - Requires: known outcome (home_win in {0,1}), kalshi_mid, grok.
  - Fit Platt calibration on Grok (train-only).
  - Choose alpha in [0,1] by leave-one-train-date-out CV on TRAIN:
      p_hybrid = p_mid + alpha * (Platt(p_grok) - p_mid)
    with Brier as objective and Platt fit only on each inner-train fold.
  - Learn per-bucket edge thresholds t* for A/B/C/D on TRAIN by sweeping
    thresholds and applying allow-gates (min_bets + ev_threshold).

Test set (date == target):
  - Requires: kalshi_mid, grok. Outcome may be blank.
  - Apply trained Platt + alpha to compute p_hybrid.
  - Use t* policy to decide A/B/C/D trades for each game.

Safety
------
Daily ledgers in `reports/daily_ledgers/` are canonical + append-only.
This tool is read-only on ledgers and only writes derived CSVs under `reports/`.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chimera_v2c.src.calibration import PlattScaler, fit_platt
from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.offset_calibration import clamp_prob
from chimera_v2c.src.rulebook_policy import TradeDecision, select_trade_decision
from chimera_v2c.src.threshold_rulebook import edge_thresholds, select_thresholds, sweep_rulebook_stats


DEFAULT_OUT_DIR = Path("reports/trade_sheets")
MODEL_KEY = "grok_mid_hybrid"


@dataclass(frozen=True)
class TrainSample:
    date: str
    league: str
    matchup: str
    p_grok: float
    p_mid: float
    y: int


@dataclass(frozen=True)
class TestSample:
    date: str
    league: str
    matchup: str
    p_grok: float
    p_mid: float


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


def _cv_alpha_brier(
    train: Sequence[TrainSample],
    *,
    alphas: Sequence[float],
    min_train_samples: int,
) -> Dict[float, float]:
    """
    Leave-one-train-date-out CV for alpha.

    Nested no-leakage: in each fold we fit Platt only on inner-train samples.
    """
    by_date: dict[str, List[TrainSample]] = defaultdict(list)
    for s in train:
        by_date[s.date].append(s)
    dates = sorted(by_date.keys())
    if len(dates) < 2:
        return {float(a): float("inf") for a in alphas}

    sq_by_alpha: Dict[float, float] = {float(a): 0.0 for a in alphas}
    n_by_alpha: Dict[float, int] = {float(a): 0 for a in alphas}

    for holdout in dates:
        train_pairs = [(s.p_grok, s.y) for s in train if s.date != holdout]
        scaler = fit_platt(train_pairs) if len(train_pairs) >= int(min_train_samples) else PlattScaler(a=1.0, b=0.0)

        for s in by_date[holdout]:
            p_mid = float(s.p_mid)
            p_grok_cal = clamp_prob(scaler.predict(float(s.p_grok)))
            for alpha in alphas:
                a = float(alpha)
                p_hybrid = clamp_prob(p_mid + a * (p_grok_cal - p_mid))
                sq_by_alpha[a] += _brier(p_hybrid, s.y)
                n_by_alpha[a] += 1

    out: Dict[float, float] = {}
    for a in alphas:
        af = float(a)
        out[af] = (sq_by_alpha[af] / n_by_alpha[af]) if n_by_alpha[af] else float("inf")
    return out


def _fit_platt_full(train: Sequence[TrainSample], *, min_train_samples: int) -> PlattScaler:
    pairs = [(s.p_grok, s.y) for s in train]
    if len(pairs) < int(min_train_samples):
        return PlattScaler(a=1.0, b=0.0)
    return fit_platt(pairs)


def _train_window(
    samples: Sequence[TrainSample],
    *,
    target_date: str,
    train_days: int,
) -> List[TrainSample]:
    before = [s for s in samples if s.date < target_date]
    if train_days <= 0:
        return before
    dates = sorted({s.date for s in before})
    keep = set(dates[-int(train_days):])
    return [s for s in before if s.date in keep]


def _predict_hybrid(*, scaler: PlattScaler, alpha: float, p_grok: float, p_mid: float) -> Tuple[float, float]:
    p_mid_f = clamp_prob(float(p_mid))
    p_grok_platt = clamp_prob(scaler.predict(clamp_prob(float(p_grok))))
    alpha_f = float(max(0.0, min(1.0, float(alpha))))
    p_hybrid = clamp_prob(p_mid_f + alpha_f * (p_grok_platt - p_mid_f))
    return p_grok_platt, p_hybrid


def _to_train_game_rows(
    train: Sequence[TrainSample],
    *,
    league_for_stats: str,
    p_hybrid_by_key: Dict[Tuple[str, str, str], float],
) -> List[GameRow]:
    out: List[GameRow] = []
    for s in train:
        p = p_hybrid_by_key.get((s.date, s.league, s.matchup))
        if p is None:
            continue
        out.append(
            GameRow(
                date=datetime.strptime(s.date, "%Y-%m-%d"),
                league=league_for_stats,
                matchup=s.matchup,
                kalshi_mid=float(s.p_mid),
                probs={MODEL_KEY: float(p)},
                home_win=float(s.y),
            )
        )
    return out


def _thresholds_by_bucket(
    *,
    train_games: Sequence[GameRow],
    league_for_stats: str,
    t_min: float,
    t_max: float,
    t_step: float,
    min_bets: int,
    ev_threshold: float,
    select_mode: str,
) -> Dict[str, float]:
    thresholds = edge_thresholds(min_edge=float(t_min), max_edge=float(t_max), step=float(t_step))
    stats_grid = sweep_rulebook_stats(
        train_games,
        thresholds=thresholds,
        models=[MODEL_KEY],
        buckets=["A", "B", "C", "D"],
    )
    selected = select_thresholds(
        stats_grid,
        thresholds=thresholds,
        min_bets=int(min_bets),
        ev_threshold=float(ev_threshold),
        mode=str(select_mode),
    )
    out: Dict[str, float] = {}
    for s in selected:
        if s.league != league_for_stats:
            continue
        if s.model != MODEL_KEY:
            continue
        out[str(s.bucket)] = float(s.edge_threshold)
    return out


def _ev_per_contract(*, decision: TradeDecision, p_hybrid: float, p_mid: float) -> float:
    # Expected profit per contract at the mid is p_true - price.
    edge_home = float(p_hybrid) - float(p_mid)
    return edge_home if decision.side == "home" else -edge_home


def _iter_train_samples(games: Iterable[GameRow]) -> Iterable[TrainSample]:
    for g in games:
        if g.home_win not in (0.0, 1.0):
            continue
        if g.kalshi_mid is None:
            continue
        p_grok = g.probs.get("grok")
        if p_grok is None:
            continue
        yield TrainSample(
            date=g.date.strftime("%Y-%m-%d"),
            league=g.league,
            matchup=g.matchup,
            p_grok=clamp_prob(float(p_grok)),
            p_mid=clamp_prob(float(g.kalshi_mid)),
            y=int(g.home_win),
        )


def _iter_test_samples(games: Iterable[GameRow], *, target_date: str) -> Iterable[TestSample]:
    for g in games:
        d = g.date.strftime("%Y-%m-%d")
        if d != target_date:
            continue
        if g.kalshi_mid is None:
            continue
        p_grok = g.probs.get("grok")
        if p_grok is None:
            continue
        yield TestSample(
            date=d,
            league=g.league,
            matchup=g.matchup,
            p_grok=clamp_prob(float(p_grok)),
            p_mid=clamp_prob(float(g.kalshi_mid)),
        )


def plan_trades(
    games: Sequence[GameRow],
    *,
    target_date: str,
    league_for_stats: str,
    train_days: int,
    min_train_days: int,
    min_train_samples: int,
    alpha_step: float,
    t_min: float,
    t_max: float,
    t_step: float,
    min_bets: int,
    ev_threshold: float,
    select_mode: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    Core planner (pure-ish): takes already-loaded games and returns (rows, meta).
    """
    train_all = list(_iter_train_samples(games))
    test = list(_iter_test_samples(games, target_date=target_date))
    if not test:
        return [], {"reason": "no_test_games"}

    train = _train_window(train_all, target_date=target_date, train_days=train_days)
    train_dates = sorted({s.date for s in train})
    trained = len(train_dates) >= int(min_train_days) and len(train) >= int(min_train_samples)

    alphas = _alpha_grid(alpha_step)
    if trained:
        cv_curve = _cv_alpha_brier(train, alphas=alphas, min_train_samples=min_train_samples)
        best_alpha = min(cv_curve.items(), key=lambda kv: kv[1])[0]
        scaler = _fit_platt_full(train, min_train_samples=min_train_samples)
    else:
        best_alpha = 0.0
        scaler = PlattScaler(a=1.0, b=0.0)

    # Precompute hybrid probs for train set (used for threshold learning).
    p_train: Dict[Tuple[str, str, str], float] = {}
    for s in train:
        p_grok_platt, p_hybrid = _predict_hybrid(scaler=scaler, alpha=best_alpha, p_grok=s.p_grok, p_mid=s.p_mid)
        _ = p_grok_platt
        p_train[(s.date, s.league, s.matchup)] = float(p_hybrid)

    thresholds_by_bucket: Dict[str, float] = {}
    if trained:
        train_games = _to_train_game_rows(train, league_for_stats=league_for_stats, p_hybrid_by_key=p_train)
        thresholds_by_bucket = _thresholds_by_bucket(
            train_games=train_games,
            league_for_stats=league_for_stats,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            min_bets=min_bets,
            ev_threshold=ev_threshold,
            select_mode=select_mode,
        )

    # Build output rows (one per test game).
    train_start = train_dates[0] if train_dates else ""
    train_end = train_dates[-1] if train_dates else ""
    t_a = thresholds_by_bucket.get("A")
    t_b = thresholds_by_bucket.get("B")
    t_c = thresholds_by_bucket.get("C")
    t_d = thresholds_by_bucket.get("D")

    rows: List[Dict[str, object]] = []
    for s in sorted(test, key=lambda x: (x.league, x.matchup)):
        p_grok_platt, p_hybrid = _predict_hybrid(scaler=scaler, alpha=best_alpha, p_grok=s.p_grok, p_mid=s.p_mid)
        edge = float(p_hybrid) - float(s.p_mid)
        decision = select_trade_decision(p_mid=float(s.p_mid), p_model=float(p_hybrid), thresholds_by_bucket=thresholds_by_bucket)
        rows.append(
            {
                "date": s.date,
                "league": s.league,
                "matchup": s.matchup,
                "p_mid": f"{float(s.p_mid):.6f}",
                "p_grok_raw": f"{float(s.p_grok):.6f}",
                "p_grok_platt": f"{float(p_grok_platt):.6f}",
                "p_hybrid": f"{float(p_hybrid):.6f}",
                "edge": f"{edge:.6f}",
                "abs_edge": f"{abs(edge):.6f}",
                "decision_bucket": decision.bucket if decision else "",
                "decision_side": decision.side if decision else "",
                "threshold_used": f"{float(decision.edge_threshold):.6f}" if decision else "",
                "ev_per_contract": f"{_ev_per_contract(decision=decision, p_hybrid=p_hybrid, p_mid=s.p_mid):.6f}"
                if decision
                else "",
                "train_start": train_start,
                "train_end": train_end,
                "train_days": len(train_dates),
                "train_samples": len(train),
                "platt_a": f"{float(scaler.a):.6f}",
                "platt_b": f"{float(scaler.b):.6f}",
                "alpha": f"{float(best_alpha):.3f}",
                "t_A": f"{float(t_a):.3f}" if t_a is not None else "",
                "t_B": f"{float(t_b):.3f}" if t_b is not None else "",
                "t_C": f"{float(t_c):.3f}" if t_c is not None else "",
                "t_D": f"{float(t_d):.3f}" if t_d is not None else "",
            }
        )

    meta = {
        "trained": trained,
        "train_start": train_start,
        "train_end": train_end,
        "train_days": len(train_dates),
        "train_samples": len(train),
        "platt_a": float(scaler.a),
        "platt_b": float(scaler.b),
        "alpha": float(best_alpha),
        "thresholds_by_bucket": dict(thresholds_by_bucket),
    }
    return rows, meta


def _default_out_path(*, league_for_stats: str, target_date: str) -> Path:
    ymd = target_date.replace("-", "")
    return DEFAULT_OUT_DIR / f"grok_mid_hybrid_trade_sheet_{league_for_stats}_{ymd}.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan paper trades using Grok-mid hybrid + A/B/C/D threshold policy (read-only).")
    ap.add_argument("--date", required=True, help="Target ledger date YYYY-MM-DD.")
    ap.add_argument("--league", default="all", help="nba|nhl|nfl|all (default: all).")
    ap.add_argument("--train-days", type=int, default=0, help="Rolling train window in ledger days (0 = all prior).")
    ap.add_argument("--min-train-days", type=int, default=5, help="Minimum distinct train days required (default: 5).")
    ap.add_argument("--min-train-samples", type=int, default=30, help="Minimum train samples for Platt fit (default: 30).")
    ap.add_argument("--alpha-step", type=float, default=0.02, help="Alpha grid step in [0,1] (default: 0.02).")
    ap.add_argument("--t-min", type=float, default=0.01, help="Min edge threshold to scan (default: 0.01).")
    ap.add_argument("--t-max", type=float, default=0.15, help="Max edge threshold to scan (default: 0.15).")
    ap.add_argument("--t-step", type=float, default=0.01, help="Edge threshold step (default: 0.01).")
    ap.add_argument("--min-bets", type=int, default=10, help="Min bets gate for selecting t* (default: 10).")
    ap.add_argument("--ev-threshold", type=float, default=0.0, help="Min avg_pnl gate for selecting t* (default: 0.0).")
    ap.add_argument(
        "--select-mode",
        choices=["min_edge", "max_avg_pnl", "max_total_pnl"],
        default="min_edge",
        help="How to pick t* among eligible thresholds (default: min_edge).",
    )
    ap.add_argument("--out", default="", help="Optional output CSV path. Default under reports/trade_sheets/.")
    ap.add_argument("--no-write", action="store_true", help="Print summary only; do not write CSV.")
    args = ap.parse_args()

    if not LEDGER_DIR.exists():
        raise SystemExit(f"[error] daily ledger directory missing: {LEDGER_DIR}")

    league_arg = (args.league or "").strip().lower()
    league_filter = None if league_arg in {"", "all"} else league_arg
    league_for_stats = league_filter or "all"

    games = load_games(
        daily_dir=LEDGER_DIR,
        end_date=args.date,
        league_filter=league_filter,
        models=["grok"],
    )
    if not games:
        raise SystemExit("[error] no games found up to the requested date (check --league/--date).")

    rows, meta = plan_trades(
        games,
        target_date=args.date,
        league_for_stats=league_for_stats,
        train_days=int(args.train_days),
        min_train_days=int(args.min_train_days),
        min_train_samples=int(args.min_train_samples),
        alpha_step=float(args.alpha_step),
        t_min=float(args.t_min),
        t_max=float(args.t_max),
        t_step=float(args.t_step),
        min_bets=int(args.min_bets),
        ev_threshold=float(args.ev_threshold),
        select_mode=str(args.select_mode),
    )
    if not rows:
        raise SystemExit(f"[error] no candidate test games with grok+kalshi_mid for date={args.date} league={league_for_stats}")

    trades = [r for r in rows if r.get("decision_bucket")]
    by_bucket = Counter([str(r.get("decision_bucket")) for r in trades])
    total_ev = 0.0
    for r in trades:
        ev = r.get("ev_per_contract")
        if ev:
            try:
                total_ev += float(ev)
            except Exception:
                continue

    print("\n=== Grok-mid hybrid paper trade sheet ===")
    print(f"date={args.date} league={league_for_stats} candidates={len(rows)} trades={len(trades)} sum_ev={total_ev:.3f}")
    if not meta.get("trained"):
        print(
            f"[warn] insufficient training data (train_days={meta.get('train_days')} train_samples={meta.get('train_samples')}); "
            "alpha=0 and no thresholds may result in zero trades."
        )
    if by_bucket:
        print("trades_by_bucket:", dict(by_bucket))
    if meta.get("thresholds_by_bucket"):
        print("thresholds_by_bucket:", meta["thresholds_by_bucket"])

    if args.no_write:
        return

    out_path = Path(args.out) if args.out else _default_out_path(league_for_stats=league_for_stats, target_date=args.date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote -> {out_path}")


if __name__ == "__main__":
    main()
