#!/usr/bin/env python
"""
Walk-forward (train->test) backtest for a Grok↔Kalshi-mid hybrid with optional consensus gating/sizing.

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
  - Optional consensus: let the chosen primary model variant determine the
    trade side and require/size based on confirmer agreement at the same
    bucket-specific threshold (e.g., `market_proxy`, `v2c`).

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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    p_confirmers: Dict[str, float]


@dataclass
class Totals:
    bets: int = 0
    total_pnl: float = 0.0

    @property
    def avg_pnl_per_bet(self) -> float:
        return self.total_pnl / self.bets if self.bets else 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Walk-forward Grok-mid hybrid backtest with optional consensus gating (read-only)."
    )
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
    ap.add_argument(
        "--primary",
        choices=["raw", "platt", "hybrid"],
        default="hybrid",
        help="Which Grok variant to use as the primary trade driver for consensus gating (default: hybrid).",
    )
    ap.add_argument(
        "--confirmers",
        nargs="+",
        default=["market_proxy"],
        help="Models that can confirm the primary direction (default: market_proxy).",
    )
    ap.add_argument(
        "--consensus-min-models",
        type=int,
        default=2,
        help="Minimum total models (primary + agreeing confirmers) required to place a trade (default: 2).",
    )
    ap.add_argument(
        "--consensus-sizing",
        nargs=3,
        type=int,
        default=(1, 1, 1),
        metavar=("U1", "U2", "U3"),
        help="Consensus unit sizing for (1 model / 2 models / 3+ models). Default: 1 1 1.",
    )
    ap.add_argument(
        "--confirmer-positive-bucket-only",
        action="store_true",
        help="Only count a confirmer toward consensus if that confirmer's in-sample (train-only) avg_pnl_per_bet "
        "in the same bucket is >= --confirmer-min-avg-pnl (and >= --confirmer-min-bets).",
    )
    ap.add_argument(
        "--confirmer-min-bets",
        type=int,
        default=10,
        help="Minimum train bets for a confirmer bucket to be eligible for confirmer-positive filtering (default: 10).",
    )
    ap.add_argument(
        "--confirmer-min-avg-pnl",
        type=float,
        default=0.0,
        help="Minimum train avg_pnl_per_bet for a confirmer bucket to be considered positive (default: 0.0).",
    )
    ap.add_argument("--out-daily", default="", help="Optional per-day output CSV path.")
    ap.add_argument("--out-trades", default="", help="Optional per-trade audit CSV path.")
    ap.add_argument("--out-summary", default="", help="Optional summary CSV path.")
    ap.add_argument("--no-write", action="store_true", help="Print summary only; do not write CSVs.")
    return ap.parse_args()


def _alpha_grid(step: float) -> List[float]:
    if step <= 0 or step > 1:
        raise SystemExit("[error] --alpha-step must be in (0,1].")
    out: List[float] = []
    a = 0.0
    while a <= 1.0 + 1e-9:
        out.append(round(a, 3))
        a += step
    if out[-1] != 1.0:
        out.append(1.0)
    return out


def iter_samples(games: Iterable[GameRow], model_col: str, confirmers: Sequence[str]) -> Iterable[Sample]:
    for g in games:
        if g.home_win not in (0.0, 1.0):
            continue
        if g.kalshi_mid is None:
            continue
        p = g.probs.get(model_col)
        if p is None:
            continue
        conf: Dict[str, float] = {}
        for m in confirmers:
            pm = g.probs.get(m)
            if pm is None:
                continue
            conf[str(m)] = clamp_prob(float(pm))
        yield Sample(
            date=g.date.strftime("%Y-%m-%d"),
            league=g.league,
            matchup=g.matchup,
            p_grok=clamp_prob(float(p)),
            p_mid=clamp_prob(float(g.kalshi_mid)),
            y=int(g.home_win),
            p_confirmers=conf,
        )


def brier(p: float, y: int) -> float:
    return (float(p) - float(y)) ** 2


def _select_best_alpha_from_scores(
    scores_by_alpha: Dict[float, float],
    *,
    larger_is_better: bool,
    tie_eps: float = 1e-12,
) -> float:
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


def _fit_platt_full(train: List[Sample], min_train_samples: int) -> PlattScaler:
    pairs = [(s.p_grok, s.y) for s in train]
    if len(pairs) < min_train_samples:
        return PlattScaler(a=1.0, b=0.0)
    return fit_platt(pairs)


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
        scaler = fit_platt(train_pairs) if len(train_pairs) >= min_train_samples else PlattScaler(a=1.0, b=0.0)

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


def _p_model_keys(model_col: str) -> Tuple[str, str, str]:
    base = model_col.strip().lower() or "grok"
    return f"{base}_raw", f"{base}_platt", f"{base}_mid_hybrid"


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

        inner_cal = {(s.date, s.league, s.matchup): clamp_prob(scaler.predict(s.p_grok)) for s in inner}
        hold_cal = {(s.date, s.league, s.matchup): clamp_prob(scaler.predict(s.p_grok)) for s in by_date[holdout]}

        for alpha in alphas:
            a = float(alpha)
            p_inner: Dict[Tuple[str, str, str], float] = {}
            for s in inner:
                p_grok_cal = inner_cal[(s.date, s.league, s.matchup)]
                p_inner[(s.date, s.league, s.matchup)] = clamp_prob(s.p_mid + a * (p_grok_cal - s.p_mid))
            inner_games: List[GameRow] = []
            for s in inner:
                key = (s.date, s.league, s.matchup)
                p_model = p_inner.get(key)
                if p_model is None:
                    continue
                inner_games.append(
                    GameRow(
                        date=datetime.strptime(s.date, "%Y-%m-%d"),
                        league=league_for_stats,
                        matchup=s.matchup,
                        kalshi_mid=s.p_mid,
                        probs={hybrid_key: float(p_model)},
                        home_win=float(s.y),
                    )
                )
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


def _units_for_models(n_models: int, sizing: Tuple[int, int, int]) -> int:
    u1, u2, u3 = sizing
    if n_models <= 0:
        return 0
    if n_models == 1:
        return int(u1)
    if n_models == 2:
        return int(u2)
    return int(u3)


def _confirmer_agrees(*, p_mid: float, p_model: float, decision: TradeDecision) -> bool:
    t = float(decision.edge_threshold)
    if decision.side == "home":
        return float(p_model) >= float(p_mid) + t
    return float(p_model) <= float(p_mid) - t


def _compute_confirmer_allowed(
    train: List[Sample],
    *,
    thresholds_by_bucket: Dict[str, float],
    confirmers: Sequence[str],
    min_bets: int,
    min_avg_pnl: float,
) -> Dict[Tuple[str, str], bool]:
    allowed: Dict[Tuple[str, str], bool] = {}
    if not confirmers:
        return allowed

    totals_by_conf_bucket: Dict[Tuple[str, str], Totals] = {}
    for s in train:
        for conf in confirmers:
            p_conf = s.p_confirmers.get(conf)
            if p_conf is None:
                continue
            decision = select_trade_decision(p_mid=s.p_mid, p_model=float(p_conf), thresholds_by_bucket=thresholds_by_bucket)
            if decision is None:
                continue
            tot = totals_by_conf_bucket.setdefault((conf, decision.bucket), Totals())
            tot.bets += 1
            tot.total_pnl += _pnl_for_trade(decision, p_mid=s.p_mid, y=s.y)

    for (conf, bucket), tot in totals_by_conf_bucket.items():
        ok = tot.bets >= int(min_bets) and float(tot.avg_pnl_per_bet) >= float(min_avg_pnl)
        allowed[(conf, bucket)] = ok
    return allowed


def _evaluate_consensus(
    test: List[Sample],
    *,
    p_primary_by_matchup: Dict[Tuple[str, str, str], float],
    thresholds_by_bucket: Dict[str, float],
    confirmers: Sequence[str],
    consensus_min_models: int,
    sizing: Tuple[int, int, int],
    confirmer_positive_bucket_only: bool,
    confirmer_allowed: Dict[Tuple[str, str], bool],
    trade_rows_out: Optional[List[Dict[str, object]]] = None,
) -> Tuple[int, int, float]:
    bets = 0
    units = 0
    total_pnl = 0.0

    for s in test:
        p_primary = p_primary_by_matchup.get((s.date, s.league, s.matchup))
        if p_primary is None:
            continue
        decision = select_trade_decision(p_mid=s.p_mid, p_model=float(p_primary), thresholds_by_bucket=thresholds_by_bucket)
        if decision is None:
            continue

        agreeing: List[str] = []
        for conf in confirmers:
            p_conf = s.p_confirmers.get(conf)
            if p_conf is None:
                continue
            if not _confirmer_agrees(p_mid=s.p_mid, p_model=float(p_conf), decision=decision):
                continue
            if confirmer_positive_bucket_only and not confirmer_allowed.get((conf, decision.bucket), False):
                continue
            agreeing.append(conf)

        n_models = 1 + len(agreeing)
        if n_models < int(consensus_min_models):
            continue

        u = _units_for_models(n_models, sizing)
        if u <= 0:
            continue
        pnl_1u = _pnl_for_trade(decision, p_mid=s.p_mid, y=s.y)

        bets += 1
        units += u
        total_pnl += pnl_1u * float(u)

        if trade_rows_out is not None:
            row: Dict[str, object] = {
                "date": s.date,
                "league": s.league,
                "matchup": s.matchup,
                "p_mid": f"{s.p_mid:.6f}",
                "p_primary": f"{float(p_primary):.6f}",
                "bucket": decision.bucket,
                "side": decision.side,
                "threshold_used": f"{float(decision.edge_threshold):.6f}",
                "n_confirmers_agree": len(agreeing),
                "confirmers_agree": " ".join(agreeing),
                "n_models": n_models,
                "units": u,
                "y": s.y,
                "pnl": f"{(pnl_1u * float(u)):.6f}",
            }
            for conf in confirmers:
                pc = s.p_confirmers.get(conf)
                row[f"p_conf_{conf}"] = f"{float(pc):.6f}" if pc is not None else ""
            trade_rows_out.append(row)

    return bets, units, float(total_pnl)


def main() -> None:
    args = parse_args()
    if not LEDGER_DIR.exists():
        raise SystemExit(f"[error] daily ledger directory missing: {LEDGER_DIR}")

    league_arg = (args.league or "").strip().lower()
    league_filter = None if league_arg in {"", "all"} else league_arg
    league_for_stats = league_filter or "all"

    model_col = args.model_col.strip()
    confirmers = [c.strip() for c in (args.confirmers or []) if c.strip()]
    needed_models = sorted(set([model_col, *confirmers]))

    games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=league_filter,
        models=needed_models,
    )
    samples_all = list(iter_samples(games, model_col=model_col, confirmers=confirmers))
    if not samples_all:
        raise SystemExit("[error] no graded samples found in the selected window.")

    test_dates = sorted({s.date for s in samples_all})
    alphas = _alpha_grid(args.alpha_step)
    thresholds = edge_thresholds(min_edge=args.t_min, max_edge=args.t_max, step=args.t_step)

    rows_out: List[Dict[str, object]] = []
    trade_rows: List[Dict[str, object]] = []

    overall: Dict[str, object] = {
        "bets_raw": 0,
        "pnl_raw": 0.0,
        "bets_platt": 0,
        "pnl_platt": 0.0,
        "bets_hybrid": 0,
        "pnl_hybrid": 0.0,
        "bets_consensus": 0,
        "units_consensus": 0,
        "pnl_consensus": 0.0,
        "brier_mid_sum": 0.0,
        "brier_raw_sum": 0.0,
        "brier_platt_sum": 0.0,
        "brier_hybrid_sum": 0.0,
        "n_games": 0,
    }

    raw_key, platt_key, hybrid_key = _p_model_keys(model_col)
    primary_label = (args.primary or "hybrid").strip().lower()
    if primary_label not in {"raw", "platt", "hybrid"}:
        raise SystemExit(f"[error] invalid --primary: {args.primary}")

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
            scaler_tmp = _fit_platt_full(train, min_train_samples=args.min_train_samples)
            p_grok_cal_by_key = {(s.date, s.league, s.matchup): clamp_prob(scaler_tmp.predict(s.p_grok)) for s in train}
            score_by_alpha: Dict[float, float] = {}
            for alpha in alphas:
                a = float(alpha)
                p_train_h: Dict[Tuple[str, str, str], float] = {}
                for s in train:
                    p_grok_cal = p_grok_cal_by_key[(s.date, s.league, s.matchup)]
                    p_train_h[(s.date, s.league, s.matchup)] = clamp_prob(s.p_mid + a * (p_grok_cal - s.p_mid))
                train_games_h: List[GameRow] = []
                for s in train:
                    key = (s.date, s.league, s.matchup)
                    p_model = p_train_h.get(key)
                    if p_model is None:
                        continue
                    train_games_h.append(
                        GameRow(
                            date=datetime.strptime(s.date, "%Y-%m-%d"),
                            league=league_for_stats,
                            matchup=s.matchup,
                            kalshi_mid=s.p_mid,
                            probs={hybrid_key: float(p_model)},
                            home_win=float(s.y),
                        )
                    )
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

        bets_raw, pnl_raw = _evaluate_pnl_for_model(test, p_model_by_matchup=p_test_raw, thresholds_by_bucket=thresholds_raw)
        bets_platt, pnl_platt = _evaluate_pnl_for_model(test, p_model_by_matchup=p_test_platt, thresholds_by_bucket=thresholds_platt)
        bets_h, pnl_h = _evaluate_pnl_for_model(test, p_model_by_matchup=p_test_hybrid, thresholds_by_bucket=thresholds_hybrid)

        if primary_label == "raw":
            primary_key = raw_key
            thresholds_primary = thresholds_raw
            p_primary = p_test_raw
        elif primary_label == "platt":
            primary_key = platt_key
            thresholds_primary = thresholds_platt
            p_primary = p_test_platt
        else:
            primary_key = hybrid_key
            thresholds_primary = thresholds_hybrid
            p_primary = p_test_hybrid

        confirmer_allowed: Dict[Tuple[str, str], bool] = {}
        if args.confirmer_positive_bucket_only and confirmers:
            confirmer_allowed = _compute_confirmer_allowed(
                train,
                thresholds_by_bucket=thresholds_primary,
                confirmers=confirmers,
                min_bets=args.confirmer_min_bets,
                min_avg_pnl=args.confirmer_min_avg_pnl,
            )

        day_trade_rows: Optional[List[Dict[str, object]]] = trade_rows if (args.out_trades or trade_rows is not None) else None
        bets_c, units_c, pnl_c = _evaluate_consensus(
            test,
            p_primary_by_matchup=p_primary,
            thresholds_by_bucket=thresholds_primary,
            confirmers=confirmers,
            consensus_min_models=args.consensus_min_models,
            sizing=tuple(args.consensus_sizing),
            confirmer_positive_bucket_only=bool(args.confirmer_positive_bucket_only),
            confirmer_allowed=confirmer_allowed,
            trade_rows_out=day_trade_rows,
        )

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
        overall["bets_consensus"] = int(overall["bets_consensus"]) + int(bets_c)
        overall["units_consensus"] = int(overall["units_consensus"]) + int(units_c)
        overall["pnl_consensus"] = float(overall["pnl_consensus"]) + float(pnl_c)

        overall["brier_mid_sum"] += brier_m
        overall["brier_raw_sum"] += brier_r
        overall["brier_platt_sum"] += brier_p
        overall["brier_hybrid_sum"] += brier_h
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
                "primary": primary_key,
                "confirmers": " ".join(confirmers),
                "consensus_min_models": int(args.consensus_min_models),
                "consensus_sizing": " ".join(str(int(x)) for x in args.consensus_sizing),
                "confirmer_positive_bucket_only": bool(args.confirmer_positive_bucket_only),
                "t_A_primary": f"{thresholds_primary.get('A', float('nan')):.3f}" if "A" in thresholds_primary else "",
                "t_B_primary": f"{thresholds_primary.get('B', float('nan')):.3f}" if "B" in thresholds_primary else "",
                "t_C_primary": f"{thresholds_primary.get('C', float('nan')):.3f}" if "C" in thresholds_primary else "",
                "t_D_primary": f"{thresholds_primary.get('D', float('nan')):.3f}" if "D" in thresholds_primary else "",
                "bets_raw": bets_raw,
                "total_pnl_raw": f"{pnl_raw:.6f}",
                "avg_pnl_per_bet_raw": f"{(pnl_raw / bets_raw):.6f}" if bets_raw else "",
                "bets_platt": bets_platt,
                "total_pnl_platt": f"{pnl_platt:.6f}",
                "avg_pnl_per_bet_platt": f"{(pnl_platt / bets_platt):.6f}" if bets_platt else "",
                "bets_hybrid": bets_h,
                "total_pnl_hybrid": f"{pnl_h:.6f}",
                "avg_pnl_per_bet_hybrid": f"{(pnl_h / bets_h):.6f}" if bets_h else "",
                "bets_consensus": bets_c,
                "units_consensus": units_c,
                "total_pnl_consensus": f"{pnl_c:.6f}",
                "avg_pnl_per_bet_consensus": f"{(pnl_c / bets_c):.6f}" if bets_c else "",
                "avg_pnl_per_unit_consensus": f"{(pnl_c / units_c):.6f}" if units_c else "",
                "brier_mid": f"{(brier_m / n_games):.6f}",
                "brier_raw": f"{(brier_r / n_games):.6f}",
                "brier_platt": f"{(brier_p / n_games):.6f}",
                "brier_hybrid": f"{(brier_h / n_games):.6f}",
            }
        )

    if not rows_out:
        raise SystemExit("[error] no walk-forward rows produced (check --min-train-days and window).")

    if args.no_write:
        out_daily = None
        out_trades = None
        out_summary = None
    else:
        def _slug(items: Sequence[str]) -> str:
            cleaned: List[str] = []
            for item in items:
                v = "".join(ch if (ch.isalnum() or ch in "-") else "-" for ch in str(item).strip().lower())
                v = "-".join([p for p in v.split("-") if p])
                if v:
                    cleaned.append(v)
            return "-".join(cleaned) if cleaned else "none"

        base_tag = f"{(args.league or 'all').lower()}_{args.start_date.replace('-','')}_{args.end_date.replace('-','')}"
        conf_tag = _slug(confirmers)
        cfg_tag = (
            f"p-{primary_label}_c-{conf_tag}_m-{int(args.consensus_min_models)}"
            f"_pos-{int(bool(args.confirmer_positive_bucket_only))}_a-{_slug([args.alpha_objective])}"
        )
        tag = f"{base_tag}_{cfg_tag}"
        out_daily = Path(args.out_daily) if args.out_daily else THESIS_DIR / f"walkforward_grok_mid_hybrid_consensus_daily_{tag}.csv"
        out_trades = Path(args.out_trades) if args.out_trades else THESIS_DIR / f"walkforward_grok_mid_hybrid_consensus_trades_{tag}.csv"
        out_summary = Path(args.out_summary) if args.out_summary else THESIS_DIR / f"walkforward_grok_mid_hybrid_consensus_summary_{tag}.csv"
        for p in (out_daily, out_trades, out_summary):
            p.parent.mkdir(parents=True, exist_ok=True)

        with out_daily.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

        if trade_rows:
            fieldnames = list(trade_rows[0].keys())
            with out_trades.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(trade_rows)

        summary_rows: List[Dict[str, object]] = []

        def add_summary(strategy: str, bets: int, units: int, pnl: float) -> None:
            summary_rows.append(
                {
                    "league": league_for_stats,
                    "strategy": strategy,
                    "days": len(rows_out),
                    "bets": bets,
                    "units": units,
                    "total_pnl": f"{pnl:.6f}",
                    "avg_pnl_per_bet": f"{(pnl / bets):.6f}" if bets else "",
                    "avg_pnl_per_unit": f"{(pnl / units):.6f}" if units else "",
                    "primary": primary_label,
                    "confirmers": " ".join(confirmers),
                    "consensus_min_models": int(args.consensus_min_models),
                    "consensus_sizing": " ".join(str(int(x)) for x in args.consensus_sizing),
                    "confirmer_positive_bucket_only": bool(args.confirmer_positive_bucket_only),
                }
            )

        add_summary("raw", int(overall["bets_raw"]), int(overall["bets_raw"]), float(overall["pnl_raw"]))
        add_summary("platt", int(overall["bets_platt"]), int(overall["bets_platt"]), float(overall["pnl_platt"]))
        add_summary("hybrid", int(overall["bets_hybrid"]), int(overall["bets_hybrid"]), float(overall["pnl_hybrid"]))
        add_summary(
            "consensus",
            int(overall["bets_consensus"]),
            int(overall["units_consensus"]),
            float(overall["pnl_consensus"]),
        )

        with out_summary.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    avg_raw = float(overall["pnl_raw"]) / int(overall["bets_raw"]) if int(overall["bets_raw"]) else 0.0
    avg_platt = float(overall["pnl_platt"]) / int(overall["bets_platt"]) if int(overall["bets_platt"]) else 0.0
    avg_h = float(overall["pnl_hybrid"]) / int(overall["bets_hybrid"]) if int(overall["bets_hybrid"]) else 0.0
    avg_cons_bet = float(overall["pnl_consensus"]) / int(overall["bets_consensus"]) if int(overall["bets_consensus"]) else 0.0
    avg_cons_unit = (
        float(overall["pnl_consensus"]) / int(overall["units_consensus"]) if int(overall["units_consensus"]) else 0.0
    )

    brier_mid = float(overall["brier_mid_sum"]) / int(overall["n_games"]) if int(overall["n_games"]) else float("inf")
    brier_raw = float(overall["brier_raw_sum"]) / int(overall["n_games"]) if int(overall["n_games"]) else float("inf")
    brier_platt = float(overall["brier_platt_sum"]) / int(overall["n_games"]) if int(overall["n_games"]) else float("inf")
    brier_h = float(overall["brier_hybrid_sum"]) / int(overall["n_games"]) if int(overall["n_games"]) else float("inf")

    print("\n=== Walk-forward summary ===")
    print(f"league={args.league or 'all'} window={args.start_date}..{args.end_date} rows={len(rows_out)}")
    print(f"raw: bets={overall['bets_raw']} total_pnl={float(overall['pnl_raw']):.3f} avg_pnl_per_bet={avg_raw:.3f}")
    print(
        f"platt: bets={overall['bets_platt']} total_pnl={float(overall['pnl_platt']):.3f} avg_pnl_per_bet={avg_platt:.3f}"
    )
    print(
        f"hybrid: bets={overall['bets_hybrid']} total_pnl={float(overall['pnl_hybrid']):.3f} avg_pnl_per_bet={avg_h:.3f}"
    )
    print(
        f"consensus(primary={primary_label}, min_models={args.consensus_min_models}, confirmers={' '.join(confirmers) or 'none'}): "
        f"bets={overall['bets_consensus']} units={overall['units_consensus']} total_pnl={float(overall['pnl_consensus']):.3f} "
        f"avg_pnl_per_bet={avg_cons_bet:.3f} avg_pnl_per_unit={avg_cons_unit:.3f}"
    )
    print(
        f"brier_mid={brier_mid:.6f} brier_raw={brier_raw:.6f} brier_platt={brier_platt:.6f} brier_hybrid={brier_h:.6f}"
    )

    if not args.no_write:
        print(f"[ok] wrote daily -> {out_daily}" if out_daily else "")
        if trade_rows:
            print(f"[ok] wrote trades -> {out_trades}" if out_trades else "")
        print(f"[ok] wrote summary -> {out_summary}" if out_summary else "")


if __name__ == "__main__":
    main()
