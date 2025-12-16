#!/usr/bin/env python
"""
Walk-forward (train->test) evaluation for a Grok↔Kalshi-mid hybrid strategy.

For each test ledger date D in a window:
  - Train on dates < D (optionally rolling `--train-days`)
  - Choose alpha in:
        p_hybrid = p_mid + alpha*(Platt(p_grok) - p_mid)
    via either:
      - `brier_cv` (default): leave-one-train-date-out CV on Brier
      - `pnl_cv`: nested leave-one-train-date-out CV on realized PnL under the
        thresholded quadrant policy (learn thresholds on the inner-train fold)
      - `pnl_train`: in-sample realized PnL on the train set under the
        thresholded quadrant policy (learn thresholds on the full train set)
  - Fit Platt on full train, compute p_hybrid for train and test
  - Learn per-bucket edge thresholds t* on train for buckets A/B/C/D by sweeping
    `edge_threshold` and applying allow-gates (min_bets + ev_threshold)
  - Apply those thresholds to test day and compute realized PnL vs Kalshi mid

This tool is read-only with respect to daily ledgers; it writes derived CSVs
under `reports/thesis_summaries/`.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from chimera_v2c.src.calibration import PlattScaler, fit_platt
from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.offset_calibration import clamp_prob
from chimera_v2c.src.rulebook_policy import TradeDecision, select_trade_decision
from chimera_v2c.src.rulebook_quadrants import pnl_buy_away, pnl_buy_home
from chimera_v2c.src.threshold_rulebook import edge_thresholds, select_thresholds, sweep_rulebook_stats


THESIS_DIR = Path("reports/thesis_summaries")


@dataclass(frozen=True)
class Sample:
    date: str
    league: str
    matchup: str
    p_grok: float
    p_mid: float
    y: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Walk-forward backtest for Grok-mid hybrid vs Kalshi mid (read-only).")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD (inclusive) for test window.")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive) for test window.")
    ap.add_argument("--league", help="League filter: nba|nhl|nfl|all (default: all).")
    ap.add_argument("--model-col", default="grok", help="Source model column to use (default: grok).")
    ap.add_argument("--train-days", type=int, default=0, help="Rolling train window in ledger days (0 = expanding).")
    ap.add_argument("--min-train-days", type=int, default=5, help="Minimum distinct train days required (default: 5).")
    ap.add_argument("--min-train-samples", type=int, default=30, help="Minimum train samples for Platt fit (default: 30).")
    ap.add_argument("--alpha-step", type=float, default=0.02, help="Alpha grid step in [0,1] (default: 0.02).")
    ap.add_argument(
        "--alpha-objective",
        choices=["brier_cv", "pnl_cv", "pnl_train"],
        default="brier_cv",
        help="How to pick alpha on the train set (default: brier_cv).",
    )
    ap.add_argument(
        "--alpha-pnl-metric",
        choices=["avg_pnl_per_bet", "total_pnl"],
        default="avg_pnl_per_bet",
        help="Metric for pnl_* alpha selection (default: avg_pnl_per_bet).",
    )
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
    ap.add_argument("--out", default="", help="Optional output CSV path. Default under reports/thesis_summaries/.")
    return ap.parse_args()


def _alpha_grid(step: float) -> List[float]:
    if step <= 0 or step > 1:
        raise SystemExit("[error] --alpha-step must be in (0,1].")
    out = []
    a = 0.0
    while a <= 1.0 + 1e-9:
        out.append(round(a, 3))
        a += step
    if out[-1] != 1.0:
        out.append(1.0)
    return out


def iter_samples(games: Iterable[GameRow], model_col: str) -> Iterable[Sample]:
    for g in games:
        if g.home_win not in (0.0, 1.0):
            continue
        if g.kalshi_mid is None:
            continue
        p = g.probs.get(model_col)
        if p is None:
            continue
        yield Sample(
            date=g.date.strftime("%Y-%m-%d"),
            league=g.league,
            matchup=g.matchup,
            p_grok=clamp_prob(float(p)),
            p_mid=clamp_prob(float(g.kalshi_mid)),
            y=int(g.home_win),
        )


def brier(p: float, y: int) -> float:
    return (float(p) - float(y)) ** 2


def _select_best_alpha_from_scores(
    scores_by_alpha: Dict[float, float],
    *,
    larger_is_better: bool,
    tie_eps: float = 1e-12,
) -> float:
    """
    Deterministic alpha selector with conservative tie-break (smaller alpha).
    """
    if not scores_by_alpha:
        return 0.0
    best_alpha = 0.0
    best_score = None
    for alpha, score in scores_by_alpha.items():
        a = float(alpha)
        s = float(score)
        if best_score is None:
            best_alpha, best_score = a, s
            continue
        better = (s > best_score + tie_eps) if larger_is_better else (s < best_score - tie_eps)
        tied = abs(s - best_score) <= tie_eps
        if better or (tied and a < best_alpha):
            best_alpha, best_score = a, s
    return float(best_alpha)


def _cv_alpha_brier(
    train: List[Sample],
    *,
    alphas: List[float],
    min_train_samples: int,
) -> Dict[float, float]:
    by_date: dict[str, List[Sample]] = defaultdict(list)
    for s in train:
        by_date[s.date].append(s)
    dates = sorted(by_date.keys())
    if len(dates) < 2:
        return {a: float("inf") for a in alphas}

    sq_by_alpha: Dict[float, float] = {a: 0.0 for a in alphas}
    n_by_alpha: Dict[float, int] = {a: 0 for a in alphas}

    for holdout in dates:
        train_pairs = [(s.p_grok, s.y) for s in train if s.date != holdout]
        if len(train_pairs) < min_train_samples:
            scaler = PlattScaler(a=1.0, b=0.0)
        else:
            scaler = fit_platt(train_pairs)

        for s in by_date[holdout]:
            p_mid = s.p_mid
            p_grok_cal = clamp_prob(scaler.predict(s.p_grok))
            for alpha in alphas:
                p_hybrid = clamp_prob(p_mid + float(alpha) * (p_grok_cal - p_mid))
                sq_by_alpha[alpha] += brier(p_hybrid, s.y)
                n_by_alpha[alpha] += 1

    out: Dict[float, float] = {}
    for a in alphas:
        out[a] = sq_by_alpha[a] / n_by_alpha[a] if n_by_alpha[a] else float("inf")
    return out


def _fit_platt_full(train: List[Sample], min_train_samples: int) -> PlattScaler:
    pairs = [(s.p_grok, s.y) for s in train]
    if len(pairs) < min_train_samples:
        return PlattScaler(a=1.0, b=0.0)
    return fit_platt(pairs)


def _p_model_keys(model_col: str) -> Tuple[str, str, str]:
    """
    Return (raw_key, platt_key, hybrid_key) for a source model column.
    """
    base = model_col.strip().lower() or "grok"
    return f"{base}_raw", f"{base}_platt", f"{base}_mid_hybrid"


def _train_window(
    all_samples: List[Sample],
    *,
    test_date: str,
    train_days: int,
) -> List[Sample]:
    before = [s for s in all_samples if s.date < test_date]
    if train_days <= 0:
        return before
    dates = sorted({s.date for s in before})
    keep = set(dates[-train_days:])
    return [s for s in before if s.date in keep]


def _to_game_rows(
    samples: List[Sample],
    *,
    model_key: str,
    p_model_by_matchup: Dict[Tuple[str, str, str], float],
    league_for_stats: str,
) -> List[GameRow]:
    out: List[GameRow] = []
    for s in samples:
        p = p_model_by_matchup.get((s.date, s.league, s.matchup))
        if p is None:
            continue
        out.append(
            GameRow(
                date=datetime.strptime(s.date, "%Y-%m-%d"),
                league=league_for_stats,
                matchup=s.matchup,
                kalshi_mid=s.p_mid,
                probs={model_key: float(p)},
                home_win=float(s.y),
            )
        )
    return out


def _selected_thresholds_by_bucket(selected: List[object], *, model_key: str, league: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for s in selected:
        if getattr(s, "league", None) != league:
            continue
        if getattr(s, "model", None) != model_key:
            continue
        b = getattr(s, "bucket", None)
        t = getattr(s, "edge_threshold", None)
        if b and t is not None:
            out[str(b)] = float(t)
    return out


def _pnl_for_trade(decision: TradeDecision, *, p_mid: float, y: int) -> float:
    if decision.side == "home":
        return pnl_buy_home(p_mid=p_mid, home_win=float(y))
    return pnl_buy_away(p_mid=p_mid, home_win=float(y))


def _learn_thresholds_for_model(
    train_games: List[GameRow],
    *,
    thresholds: List[float],
    model_key: str,
    league_for_stats: str,
    min_bets: int,
    ev_threshold: float,
    select_mode: str,
) -> Dict[str, float]:
    stats_grid = sweep_rulebook_stats(
        train_games,
        thresholds=thresholds,
        models=[model_key],
        buckets=["A", "B", "C", "D"],
    )
    selected = select_thresholds(
        stats_grid,
        thresholds=thresholds,
        min_bets=min_bets,
        ev_threshold=ev_threshold,
        mode=select_mode,
    )
    return _selected_thresholds_by_bucket(selected, model_key=model_key, league=league_for_stats)


def _evaluate_pnl_for_model(
    samples: List[Sample],
    *,
    p_model_by_matchup: Dict[Tuple[str, str, str], float],
    thresholds_by_bucket: Dict[str, float],
) -> Tuple[int, float]:
    bets = 0
    total_pnl = 0.0
    for s in samples:
        p_model = p_model_by_matchup.get((s.date, s.league, s.matchup))
        if p_model is None:
            continue
        decision = select_trade_decision(p_mid=s.p_mid, p_model=float(p_model), thresholds_by_bucket=thresholds_by_bucket)
        if decision is None:
            continue
        bets += 1
        total_pnl += _pnl_for_trade(decision, p_mid=s.p_mid, y=s.y)
    return bets, float(total_pnl)


def _alpha_score_from_pnl(bets: int, total_pnl: float, *, metric: str, min_bets: int) -> float:
    if bets < int(min_bets):
        return float("-inf")
    metric_norm = (metric or "").strip().lower()
    if metric_norm == "total_pnl":
        return float(total_pnl)
    if metric_norm == "avg_pnl_per_bet":
        return float(total_pnl) / float(bets) if bets else float("-inf")
    raise ValueError("alpha_pnl_metric must be one of: avg_pnl_per_bet, total_pnl")


def _cv_alpha_pnl(
    train: List[Sample],
    *,
    alphas: List[float],
    min_train_samples: int,
    thresholds: List[float],
    league_for_stats: str,
    min_bets: int,
    ev_threshold: float,
    select_mode: str,
    metric: str,
) -> Dict[float, float]:
    """
    Nested leave-one-train-date-out CV for alpha with realized PnL objective.

    For each holdout date H in the TRAIN set:
      - fit Platt on train (excluding H) (or identity if too small)
      - for each alpha:
          * compute p_hybrid on inner train and learn thresholds t* on inner train
          * apply t* to H and accumulate realized PnL

    Returns score_by_alpha where larger is better (avg_pnl_per_bet or total_pnl).
    """
    by_date: dict[str, List[Sample]] = defaultdict(list)
    for s in train:
        by_date[s.date].append(s)
    dates = sorted(by_date.keys())
    if len(dates) < 2:
        return {a: float("-inf") for a in alphas}

    hybrid_key = "hybrid"
    total_pnl_by_alpha: Dict[float, float] = {float(a): 0.0 for a in alphas}
    total_bets_by_alpha: Dict[float, int] = {float(a): 0 for a in alphas}

    for holdout in dates:
        inner = [s for s in train if s.date != holdout]
        scaler = _fit_platt_full(inner, min_train_samples=min_train_samples)

        # Precompute calibrated Grok for inner and holdout once (shared across alphas).
        inner_cal = {(s.date, s.league, s.matchup): clamp_prob(scaler.predict(s.p_grok)) for s in inner}
        hold_cal = {(s.date, s.league, s.matchup): clamp_prob(scaler.predict(s.p_grok)) for s in by_date[holdout]}

        for alpha in alphas:
            a = float(alpha)
            p_inner: Dict[Tuple[str, str, str], float] = {}
            for s in inner:
                p_grok_cal = inner_cal[(s.date, s.league, s.matchup)]
                p_inner[(s.date, s.league, s.matchup)] = clamp_prob(s.p_mid + a * (p_grok_cal - s.p_mid))
            inner_games = _to_game_rows(inner, model_key=hybrid_key, p_model_by_matchup=p_inner, league_for_stats=league_for_stats)
            thresholds_by_bucket = _learn_thresholds_for_model(
                inner_games,
                thresholds=thresholds,
                model_key=hybrid_key,
                league_for_stats=league_for_stats,
                min_bets=min_bets,
                ev_threshold=ev_threshold,
                select_mode=select_mode,
            )

            p_hold: Dict[Tuple[str, str, str], float] = {}
            for s in by_date[holdout]:
                p_grok_cal = hold_cal[(s.date, s.league, s.matchup)]
                p_hold[(s.date, s.league, s.matchup)] = clamp_prob(s.p_mid + a * (p_grok_cal - s.p_mid))
            bets, pnl = _evaluate_pnl_for_model(by_date[holdout], p_model_by_matchup=p_hold, thresholds_by_bucket=thresholds_by_bucket)
            total_bets_by_alpha[a] += int(bets)
            total_pnl_by_alpha[a] += float(pnl)

    return {
        float(a): _alpha_score_from_pnl(
            total_bets_by_alpha[float(a)],
            total_pnl_by_alpha[float(a)],
            metric=metric,
            min_bets=min_bets,
        )
        for a in alphas
    }


def main() -> None:
    args = parse_args()
    if not LEDGER_DIR.exists():
        raise SystemExit(f"[error] daily ledger directory missing: {LEDGER_DIR}")

    league_arg = (args.league or "").strip().lower()
    league_filter = None if league_arg in {"", "all"} else league_arg
    league_for_stats = league_filter or "all"
    games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=league_filter,
        models=[args.model_col],
    )
    samples_all = list(iter_samples(games, model_col=args.model_col.strip()))
    if not samples_all:
        raise SystemExit("[error] no graded samples found in the selected window.")

    test_dates = sorted({s.date for s in samples_all})
    alphas = _alpha_grid(args.alpha_step)
    thresholds = edge_thresholds(min_edge=args.t_min, max_edge=args.t_max, step=args.t_step)

    rows_out: List[Dict[str, object]] = []
    overall: Dict[str, object] = {
        "bets_raw": 0,
        "pnl_raw": 0.0,
        "bets_platt": 0,
        "pnl_platt": 0.0,
        "bets_hybrid": 0,
        "pnl_hybrid": 0.0,
        "brier_mid_sum": 0.0,
        "brier_raw_sum": 0.0,
        "brier_platt_sum": 0.0,
        "brier_hybrid_sum": 0.0,
        "n_games": 0,
    }
    raw_key, platt_key, hybrid_key = _p_model_keys(args.model_col)

    for d in test_dates:
        train = _train_window(samples_all, test_date=d, train_days=args.train_days)
        train_days = sorted({s.date for s in train})
        if len(train_days) < args.min_train_days:
            continue
        test = [s for s in samples_all if s.date == d]
        if not test:
            continue

        alpha_objective = (args.alpha_objective or "brier_cv").strip().lower()
        if alpha_objective == "brier_cv":
            cv_curve = _cv_alpha_brier(train, alphas=alphas, min_train_samples=args.min_train_samples)
            best_alpha = _select_best_alpha_from_scores(cv_curve, larger_is_better=False)
            alpha_score = float(cv_curve.get(float(best_alpha), float("inf")))
            alpha_score_label = "cv_brier"
        elif alpha_objective == "pnl_cv":
            score_by_alpha = _cv_alpha_pnl(
                train,
                alphas=alphas,
                min_train_samples=args.min_train_samples,
                thresholds=thresholds,
                league_for_stats=league_for_stats,
                min_bets=args.min_bets,
                ev_threshold=args.ev_threshold,
                select_mode=args.select_mode,
                metric=args.alpha_pnl_metric,
            )
            best_alpha = _select_best_alpha_from_scores(score_by_alpha, larger_is_better=True)
            alpha_score = float(score_by_alpha.get(float(best_alpha), float("-inf")))
            alpha_score_label = f"cv_{args.alpha_pnl_metric}"
        elif alpha_objective == "pnl_train":
            # Train-only alpha selection: for each alpha, learn thresholds on train and score realized train PnL.
            scaler_tmp = _fit_platt_full(train, min_train_samples=args.min_train_samples)
            p_grok_cal_by_key = {(s.date, s.league, s.matchup): clamp_prob(scaler_tmp.predict(s.p_grok)) for s in train}
            score_by_alpha: Dict[float, float] = {}
            for alpha in alphas:
                a = float(alpha)
                p_train_h: Dict[Tuple[str, str, str], float] = {}
                for s in train:
                    p_grok_cal = p_grok_cal_by_key[(s.date, s.league, s.matchup)]
                    p_train_h[(s.date, s.league, s.matchup)] = clamp_prob(s.p_mid + a * (p_grok_cal - s.p_mid))
                train_games_h = _to_game_rows(train, model_key=hybrid_key, p_model_by_matchup=p_train_h, league_for_stats=league_for_stats)
                thresholds_by_bucket_h = _learn_thresholds_for_model(
                    train_games_h,
                    thresholds=thresholds,
                    model_key=hybrid_key,
                    league_for_stats=league_for_stats,
                    min_bets=args.min_bets,
                    ev_threshold=args.ev_threshold,
                    select_mode=args.select_mode,
                )
                bets_h, pnl_h = _evaluate_pnl_for_model(train, p_model_by_matchup=p_train_h, thresholds_by_bucket=thresholds_by_bucket_h)
                score_by_alpha[a] = _alpha_score_from_pnl(bets_h, pnl_h, metric=args.alpha_pnl_metric, min_bets=args.min_bets)
            best_alpha = _select_best_alpha_from_scores(score_by_alpha, larger_is_better=True)
            alpha_score = float(score_by_alpha.get(float(best_alpha), float("-inf")))
            alpha_score_label = f"train_{args.alpha_pnl_metric}"
        else:
            raise SystemExit(f"[error] unknown --alpha-objective: {args.alpha_objective}")

        scaler = _fit_platt_full(train, min_train_samples=args.min_train_samples)

        # Precompute raw/platt/hybrid probabilities for train and test sets (matchup-keyed).
        p_train_raw: Dict[Tuple[str, str, str], float] = {}
        p_train_platt: Dict[Tuple[str, str, str], float] = {}
        p_train_hybrid: Dict[Tuple[str, str, str], float] = {}
        for s in train:
            p_grok_raw = clamp_prob(float(s.p_grok))
            p_grok_cal = clamp_prob(scaler.predict(p_grok_raw))
            key = (s.date, s.league, s.matchup)
            p_train_raw[key] = p_grok_raw
            p_train_platt[key] = p_grok_cal
            p_train_hybrid[key] = clamp_prob(s.p_mid + float(best_alpha) * (p_grok_cal - s.p_mid))

        p_test_raw: Dict[Tuple[str, str, str], float] = {}
        p_test_platt: Dict[Tuple[str, str, str], float] = {}
        p_test_hybrid: Dict[Tuple[str, str, str], float] = {}
        for s in test:
            p_grok_raw = clamp_prob(float(s.p_grok))
            p_grok_cal = clamp_prob(scaler.predict(p_grok_raw))
            key = (s.date, s.league, s.matchup)
            p_test_raw[key] = p_grok_raw
            p_test_platt[key] = p_grok_cal
            p_test_hybrid[key] = clamp_prob(s.p_mid + float(best_alpha) * (p_grok_cal - s.p_mid))

        # Learn per-bucket thresholds for each model variant on train.
        train_games_all: List[GameRow] = []
        for s in train:
            key = (s.date, s.league, s.matchup)
            probs = {
                raw_key: float(p_train_raw[key]),
                platt_key: float(p_train_platt[key]),
                hybrid_key: float(p_train_hybrid[key]),
            }
            train_games_all.append(
                GameRow(
                    date=datetime.strptime(s.date, "%Y-%m-%d"),
                    league=league_for_stats,
                    matchup=s.matchup,
                    kalshi_mid=s.p_mid,
                    probs=probs,
                    home_win=float(s.y),
                )
            )

        thresholds_raw = _learn_thresholds_for_model(
            train_games_all,
            thresholds=thresholds,
            model_key=raw_key,
            league_for_stats=league_for_stats,
            min_bets=args.min_bets,
            ev_threshold=args.ev_threshold,
            select_mode=args.select_mode,
        )
        thresholds_platt = _learn_thresholds_for_model(
            train_games_all,
            thresholds=thresholds,
            model_key=platt_key,
            league_for_stats=league_for_stats,
            min_bets=args.min_bets,
            ev_threshold=args.ev_threshold,
            select_mode=args.select_mode,
        )
        thresholds_hybrid = _learn_thresholds_for_model(
            train_games_all,
            thresholds=thresholds,
            model_key=hybrid_key,
            league_for_stats=league_for_stats,
            min_bets=args.min_bets,
            ev_threshold=args.ev_threshold,
            select_mode=args.select_mode,
        )

        # Evaluate on test day.
        bets_raw, pnl_raw = _evaluate_pnl_for_model(test, p_model_by_matchup=p_test_raw, thresholds_by_bucket=thresholds_raw)
        bets_platt, pnl_platt = _evaluate_pnl_for_model(test, p_model_by_matchup=p_test_platt, thresholds_by_bucket=thresholds_platt)
        bets_h, pnl_h = _evaluate_pnl_for_model(test, p_model_by_matchup=p_test_hybrid, thresholds_by_bucket=thresholds_hybrid)

        brier_m = 0.0
        brier_r = 0.0
        brier_p = 0.0
        brier_h = 0.0
        for s in test:
            key = (s.date, s.league, s.matchup)
            brier_m += brier(s.p_mid, s.y)
            brier_r += brier(float(p_test_raw[key]), s.y)
            brier_p += brier(float(p_test_platt[key]), s.y)
            brier_h += brier(float(p_test_hybrid[key]), s.y)

        n_games = len(test)
        if n_games == 0:
            continue
        overall["bets_raw"] = int(overall["bets_raw"]) + int(bets_raw)
        overall["pnl_raw"] = float(overall["pnl_raw"]) + float(pnl_raw)
        overall["bets_platt"] = int(overall["bets_platt"]) + int(bets_platt)
        overall["pnl_platt"] = float(overall["pnl_platt"]) + float(pnl_platt)
        overall["bets_hybrid"] = int(overall["bets_hybrid"]) + int(bets_h)
        overall["pnl_hybrid"] = float(overall["pnl_hybrid"]) + float(pnl_h)
        overall["brier_hybrid_sum"] += brier_h
        overall["brier_mid_sum"] += brier_m
        overall["brier_raw_sum"] += brier_r
        overall["brier_platt_sum"] += brier_p
        overall["n_games"] += n_games

        rows_out.append(
            {
                "test_date": d,
                "league": league_for_stats,
                "train_start": train_days[0],
                "train_end": train_days[-1],
                "train_days": len(train_days),
                "train_samples": len(train),
                "test_games": n_games,
                "platt_a": f"{scaler.a:.6f}",
                "platt_b": f"{scaler.b:.6f}",
                "alpha": f"{float(best_alpha):.3f}",
                "alpha_objective": alpha_objective,
                "alpha_score_label": alpha_score_label,
                "alpha_score": f"{alpha_score:.6f}" if alpha_score == alpha_score else "",
                "t_A_raw": f"{thresholds_raw.get('A', float('nan')):.3f}" if "A" in thresholds_raw else "",
                "t_B_raw": f"{thresholds_raw.get('B', float('nan')):.3f}" if "B" in thresholds_raw else "",
                "t_C_raw": f"{thresholds_raw.get('C', float('nan')):.3f}" if "C" in thresholds_raw else "",
                "t_D_raw": f"{thresholds_raw.get('D', float('nan')):.3f}" if "D" in thresholds_raw else "",
                "t_A_platt": f"{thresholds_platt.get('A', float('nan')):.3f}" if "A" in thresholds_platt else "",
                "t_B_platt": f"{thresholds_platt.get('B', float('nan')):.3f}" if "B" in thresholds_platt else "",
                "t_C_platt": f"{thresholds_platt.get('C', float('nan')):.3f}" if "C" in thresholds_platt else "",
                "t_D_platt": f"{thresholds_platt.get('D', float('nan')):.3f}" if "D" in thresholds_platt else "",
                "t_A_hybrid": f"{thresholds_hybrid.get('A', float('nan')):.3f}" if "A" in thresholds_hybrid else "",
                "t_B_hybrid": f"{thresholds_hybrid.get('B', float('nan')):.3f}" if "B" in thresholds_hybrid else "",
                "t_C_hybrid": f"{thresholds_hybrid.get('C', float('nan')):.3f}" if "C" in thresholds_hybrid else "",
                "t_D_hybrid": f"{thresholds_hybrid.get('D', float('nan')):.3f}" if "D" in thresholds_hybrid else "",
                "bets_raw": bets_raw,
                "total_pnl_raw": f"{pnl_raw:.6f}",
                "avg_pnl_per_bet_raw": f"{(pnl_raw / bets_raw):.6f}" if bets_raw else "",
                "bets_platt": bets_platt,
                "total_pnl_platt": f"{pnl_platt:.6f}",
                "avg_pnl_per_bet_platt": f"{(pnl_platt / bets_platt):.6f}" if bets_platt else "",
                "bets_hybrid": bets_h,
                "total_pnl_hybrid": f"{pnl_h:.6f}",
                "avg_pnl_per_bet_hybrid": f"{(pnl_h / bets_h):.6f}" if bets_h else "",
                "brier_mid": f"{(brier_m / n_games):.6f}",
                "brier_raw": f"{(brier_r / n_games):.6f}",
                "brier_platt": f"{(brier_p / n_games):.6f}",
                "brier_hybrid": f"{(brier_h / n_games):.6f}",
            }
        )

    if not rows_out:
        raise SystemExit("[error] no walk-forward rows produced (check min_train_days/min_bets).")

    out_path = Path(args.out) if args.out else THESIS_DIR / f"walkforward_grok_mid_hybrid_{(args.league or 'all').lower()}_{args.start_date.replace('-','')}_{args.end_date.replace('-','')}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows_out[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    avg_raw = float(overall["pnl_raw"]) / int(overall["bets_raw"]) if int(overall["bets_raw"]) else 0.0
    avg_platt = float(overall["pnl_platt"]) / int(overall["bets_platt"]) if int(overall["bets_platt"]) else 0.0
    avg_h = float(overall["pnl_hybrid"]) / int(overall["bets_hybrid"]) if int(overall["bets_hybrid"]) else 0.0

    brier_mid = float(overall["brier_mid_sum"]) / int(overall["n_games"]) if int(overall["n_games"]) else float("inf")
    brier_raw = float(overall["brier_raw_sum"]) / int(overall["n_games"]) if int(overall["n_games"]) else float("inf")
    brier_platt = float(overall["brier_platt_sum"]) / int(overall["n_games"]) if int(overall["n_games"]) else float("inf")
    brier_h = float(overall["brier_hybrid_sum"]) / int(overall["n_games"]) if int(overall["n_games"]) else float("inf")

    print("\n=== Walk-forward summary ===")
    print(f"league={args.league or 'all'} window={args.start_date}..{args.end_date} rows={len(rows_out)}")
    print(
        f"raw: bets={overall['bets_raw']} total_pnl={float(overall['pnl_raw']):.3f} avg_pnl_per_bet={avg_raw:.3f}"
    )
    print(
        f"platt: bets={overall['bets_platt']} total_pnl={float(overall['pnl_platt']):.3f} avg_pnl_per_bet={avg_platt:.3f}"
    )
    print(
        f"hybrid: bets={overall['bets_hybrid']} total_pnl={float(overall['pnl_hybrid']):.3f} avg_pnl_per_bet={avg_h:.3f}"
    )
    print(
        f"brier_mid={brier_mid:.6f} brier_raw={brier_raw:.6f} brier_platt={brier_platt:.6f} brier_hybrid={brier_h:.6f}"
    )
    print(f"[ok] wrote -> {out_path}")


if __name__ == "__main__":
    main()
