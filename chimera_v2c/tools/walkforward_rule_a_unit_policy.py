#!/usr/bin/env python
"""
Walk-forward (train->test) Rule-A unit sizing policy using Kalshi bid/ask snapshots (taker-only; fee-aware).

Rule A (baseline):
  - Qualifying game: Kalshi favors HOME at the anchor (home YES mid > 0.50)
  - Trade: buy AWAY YES at the AWAY YES ask (+ optional slippage)
  - Baseline size: always 1 contract per qualifying game

Hybrid sizing:
  - For each league and model m, learn a threshold t_m such that:
      edge_net = (1 - p_model_home) - price_away - fee(1 contract @ price_away) >= t_m
    has a positive *conservative* realized mean net PnL per added contract.
  - Confidence is enforced via a bootstrap lower bound on mean net PnL:
      mean_low = quantile_{(1-confidence_level)}(bootstrap_means)
    A model is eligible only if mean_low > 0 and the signal subset has >= min_signals.
  - For a game, score = sum(edge_net_m for triggered models).
  - Extra contracts = floor(score / unit_scale); total contracts = min(cap_units, 1 + extra)
  - unit_scale is selected on train-only data to maximize total net PnL subject to an ROI floor vs blind fade.

This tool is read-only on daily ledgers; it writes derived CSVs under reports/.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.rule_a_unit_policy import (
    PolicyCalibration,
    Totals,
    learn_weak_mid_buckets_for_votes,
    iter_rule_a_games,
    load_bidask_csv,
    mid_bucket,
    net_pnl_taker,
    policy_totals,
    train_rule_a_policy,
    units_for_game,
    votes_agg_totals,
    votes_for_game,
)


DEFAULT_MODELS = ["v2c", "grok", "gemini", "gpt", "market_proxy", "moneypuck"]
DEFAULT_THRESHOLDS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
DEFAULT_UNIT_SCALES = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 1e9]


def _normalize_league_arg(league: str) -> Optional[str]:
    v = (league or "").strip().lower()
    if v in {"", "all"}:
        return None
    if v in {"nba", "nhl", "nfl"}:
        return v
    raise SystemExit("[error] --league must be one of: nba, nhl, nfl, all")


def _ensure_bidask_csv(*, start_date: str, end_date: str, minutes: int, out_path: Path, export_missing: bool) -> None:
    if out_path.exists() or not export_missing:
        return
    cmd = [
        "python",
        "chimera_v2c/tools/export_kalshi_bidask_tminus.py",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--minutes-before-start",
        str(int(minutes)),
        "--out",
        str(out_path),
    ]
    env = dict(**{k: v for k, v in dict(**__import__("os").environ).items()})
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
    subprocess.run(cmd, check=True, env=env)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _dates_sorted(games: Iterable[GameRow]) -> List[str]:
    ds = sorted({g.date.strftime("%Y-%m-%d") for g in games})
    return ds


def _train_window_dates(dates: Sequence[str], *, test_date: str, train_days: int) -> List[str]:
    prior = [d for d in dates if d < test_date]
    if train_days <= 0:
        return prior
    return prior[-int(train_days) :]


def _sum_totals(a: Totals, b: Totals) -> Totals:
    out = Totals()
    out.bets = int(a.bets + b.bets)
    out.contracts = int(a.contracts + b.contracts)
    out.risked = float(a.risked + b.risked)
    out.gross_pnl = float(a.gross_pnl + b.gross_pnl)
    out.fees = float(a.fees + b.fees)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Walk-forward Rule-A unit sizing policy (taker-only; bid/ask at T-minus).")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive (analysis window).")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive (analysis window).")
    ap.add_argument("--league", default="all", help="nba|nhl|nfl|all (default: all).")
    ap.add_argument("--anchor-minutes", type=int, default=30, help="T-minus anchor in minutes (default: 30).")
    ap.add_argument("--bidask-csv", default="", help="Path to kalshi_bidask_tminus_*.csv (default derived from dates).")
    ap.add_argument("--export-missing", action="store_true", help="Export bid/ask snapshot CSV if missing.")
    ap.add_argument("--slippage-cents", type=int, default=0, help="Extra slippage to add on top of away ask (default: 0).")
    ap.add_argument("--cap-units", type=int, default=10, help="Max contracts per game (default: 10).")

    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help=f"Models to consider (default: {' '.join(DEFAULT_MODELS)}).")
    ap.add_argument("--train-days", type=int, default=0, help="Rolling train window in days (0 = expanding).")
    ap.add_argument("--min-train-days", type=int, default=5, help="Minimum distinct train days required (default: 5).")
    ap.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Delta thresholds (mid_home - p_model) to consider (default preset grid).",
    )
    ap.add_argument("--min-signals", type=int, default=10, help="Min signal games per (league,model,threshold) (default: 10).")
    ap.add_argument(
        "--confidence-level",
        type=float,
        default=0.9,
        help="Bootstrap confidence level for mean net > 0 (default: 0.90).",
    )
    ap.add_argument("--bootstrap-sims", type=int, default=1000, help="Bootstrap sims for mean distribution (default: 1000).")
    ap.add_argument(
        "--threshold-select-mode",
        choices=["max_total_mean_low", "max_mean_low", "min_threshold"],
        default="max_total_mean_low",
        help="How to choose t_m on train-only data (default: max_total_mean_low).",
    )
    ap.add_argument(
        "--unit-scales",
        nargs="+",
        type=float,
        default=DEFAULT_UNIT_SCALES,
        help="Candidate unit_scale values (default includes 1e9 ~ baseline-only).",
    )
    ap.add_argument(
        "--roi-floor-mult",
        type=float,
        default=0.9,
        help="Require train ROI >= baseline_roi * roi_floor_mult when selecting unit_scale (default: 0.90).",
    )
    ap.add_argument(
        "--bucket-cap-mode",
        choices=["none", "votes_mean_low"],
        default="votes_mean_low",
        help="Optional risk cap for weak mid-home buckets, learned from train only (default: votes_mean_low).",
    )
    ap.add_argument("--min-bucket-bets", type=int, default=30, help="Min train games per mid bucket (default: 30).")
    ap.add_argument(
        "--weak-bucket-cap",
        type=int,
        default=3,
        help="Max contracts in weak mid buckets (default: 3).",
    )
    ap.add_argument("--seed", type=int, default=1337, help="Base RNG seed for bootstraps (default: 1337).")

    ap.add_argument("--out-dir", default="reports/thesis_summaries", help="Output directory (default: reports/thesis_summaries).")
    ap.add_argument("--no-write", action="store_true", help="Compute and print summary only; do not write CSVs.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    league_filter = _normalize_league_arg(args.league)

    bidask_path = Path(args.bidask_csv) if args.bidask_csv else Path(
        f"reports/market_snapshots/kalshi_bidask_tminus_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}_m{int(args.anchor_minutes)}.csv"
    )
    _ensure_bidask_csv(
        start_date=args.start_date,
        end_date=args.end_date,
        minutes=int(args.anchor_minutes),
        out_path=bidask_path,
        export_missing=bool(args.export_missing),
    )
    if not bidask_path.exists():
        raise SystemExit(f"[error] missing bid/ask csv: {bidask_path} (pass --export-missing or --bidask-csv)")

    bidask = load_bidask_csv(bidask_path)

    ledger_games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=league_filter,
        models=list(args.models),
    )
    if not ledger_games:
        raise SystemExit("[error] no ledger games loaded for window")

    rule_a_games = iter_rule_a_games(
        ledger_games,
        bidask=bidask,
        models=list(args.models),
        slippage_cents=int(args.slippage_cents),
        require_outcome=True,
    )
    if not rule_a_games:
        raise SystemExit("[error] no Rule-A qualifying graded games found for window + snapshot")

    leagues = sorted({g.league for g in rule_a_games})
    dates = sorted({g.date for g in rule_a_games})
    dates_by_league: Dict[str, List[str]] = {lg: sorted({g.date for g in rule_a_games if g.league == lg}) for lg in leagues}

    by_date_league: Dict[Tuple[str, str], List] = defaultdict(list)
    for g in rule_a_games:
        by_date_league[(g.date, g.league)].append(g)

    daily_rows: List[Dict[str, object]] = []
    trade_rows: List[Dict[str, object]] = []
    calib_rows: List[Dict[str, object]] = []

    totals_policy: Dict[str, Totals] = {lg: Totals() for lg in leagues + ["overall"]}
    totals_base: Dict[str, Totals] = {lg: Totals() for lg in leagues + ["overall"]}
    totals_votes: Dict[str, Totals] = {lg: Totals() for lg in leagues + ["overall"]}
    totals_votes_cap: Dict[str, Totals] = {lg: Totals() for lg in leagues + ["overall"]}

    for test_date in dates:
        for league in leagues:
            test = by_date_league.get((test_date, league), [])
            if not test:
                continue

            train_dates = _train_window_dates(
                dates_by_league.get(league, []),
                test_date=test_date,
                train_days=int(args.train_days),
            )
            train = [g for d in train_dates for g in by_date_league.get((d, league), [])]
            trained = len({g.date for g in train}) >= int(args.min_train_days)

            if trained:
                weak_buckets: List[str] = []
                bucket_metrics: Dict[str, object] = {}
                if str(args.bucket_cap_mode) == "votes_mean_low":
                    bm = learn_weak_mid_buckets_for_votes(
                        train,
                        models=list(args.models),
                        cap_units=int(args.cap_units),
                        confidence_level=float(args.confidence_level),
                        bootstrap_sims=int(args.bootstrap_sims),
                        min_bucket_bets=int(args.min_bucket_bets),
                        seed=int(args.seed) ^ 0xA11CE,
                    )
                    bucket_metrics = {
                        b: {
                            "n": m.n,
                            "mean_net": m.mean_net,
                            "mean_low": m.mean_low,
                            "conf_gt0": m.conf_gt0,
                        }
                        for b, m in bm.items()
                    }
                    weak_buckets = sorted([b for b, m in bm.items() if float(m.mean_low) < 0.0])

                policy = train_rule_a_policy(
                    train,
                    league=league,
                    models=list(args.models),
                    thresholds=[float(t) for t in args.thresholds],
                    unit_scales=[float(s) for s in args.unit_scales],
                    cap_units=int(args.cap_units),
                    min_signals=int(args.min_signals),
                    confidence_level=float(args.confidence_level),
                    bootstrap_sims=int(args.bootstrap_sims),
                    roi_floor_mult=float(args.roi_floor_mult),
                    threshold_select_mode=str(args.threshold_select_mode),
                    seed=int(args.seed),
                    weak_buckets=weak_buckets,
                    weak_bucket_cap=int(args.weak_bucket_cap),
                )
            else:
                weak_buckets = []
                bucket_metrics = {}
                policy = PolicyCalibration(league=league, confidence_level=float(args.confidence_level), unit_scale=1e9, models={})

            weak_bucket_set = set(weak_buckets)

            # Record calibration details for this test date.
            for mp in policy.models.values():
                calib_rows.append(
                    {
                        "date": test_date,
                        "league": league,
                        "model": mp.model,
                        "threshold": mp.threshold,
                        "weight_mean_low": mp.weight,
                        "signals_n": mp.metrics.n,
                        "signals_mean_net": mp.metrics.mean_net,
                        "signals_mean_low": mp.metrics.mean_low,
                        "signals_conf_gt0": mp.metrics.conf_gt0,
                        "confidence_level": policy.confidence_level,
                        "unit_scale": policy.unit_scale,
                        "bucket_cap_mode": str(args.bucket_cap_mode),
                        "weak_mid_buckets": ",".join(weak_buckets),
                        "bucket_metrics_json": "",
                    }
                )
            if bucket_metrics:
                calib_rows.append(
                    {
                        "date": test_date,
                        "league": league,
                        "model": "__bucket_caps__",
                        "threshold": "",
                        "weight_mean_low": "",
                        "signals_n": "",
                        "signals_mean_net": "",
                        "signals_mean_low": "",
                        "signals_conf_gt0": "",
                        "confidence_level": policy.confidence_level,
                        "unit_scale": policy.unit_scale,
                        "bucket_cap_mode": str(args.bucket_cap_mode),
                        "weak_mid_buckets": ",".join(weak_buckets),
                        "bucket_metrics_json": str(bucket_metrics),
                    }
                )

            # Evaluate baseline vs policy on this date.
            base_tot = policy_totals(
                test,
                models={},
                unit_scale=1e9,
                cap_units=int(args.cap_units),
                weak_buckets=weak_buckets,
                weak_bucket_cap=int(args.weak_bucket_cap),
            )
            votes_tot = votes_agg_totals(test, models=list(args.models), cap_units=int(args.cap_units))
            votes_cap_tot = votes_agg_totals(
                test,
                models=list(args.models),
                cap_units=int(args.cap_units),
                weak_buckets=weak_buckets,
                weak_bucket_cap=int(args.weak_bucket_cap),
            )
            pol_tot = policy_totals(
                test,
                models=policy.models,
                unit_scale=float(policy.unit_scale),
                cap_units=int(args.cap_units),
                weak_buckets=weak_buckets,
                weak_bucket_cap=int(args.weak_bucket_cap),
            )

            totals_base[league] = _sum_totals(totals_base[league], base_tot)
            totals_policy[league] = _sum_totals(totals_policy[league], pol_tot)
            totals_votes[league] = _sum_totals(totals_votes[league], votes_tot)
            totals_votes_cap[league] = _sum_totals(totals_votes_cap[league], votes_cap_tot)
            totals_base["overall"] = _sum_totals(totals_base["overall"], base_tot)
            totals_policy["overall"] = _sum_totals(totals_policy["overall"], pol_tot)
            totals_votes["overall"] = _sum_totals(totals_votes["overall"], votes_tot)
            totals_votes_cap["overall"] = _sum_totals(totals_votes_cap["overall"], votes_cap_tot)

            daily_rows.append(
                {
                    "date": test_date,
                    "league": league,
                    "trained": bool(trained),
                    "train_days_used": len({g.date for g in train}),
                    "active_models": ",".join(sorted(policy.models.keys())),
                    "confidence_level": policy.confidence_level,
                    "unit_scale": policy.unit_scale,
                    "bucket_cap_mode": str(args.bucket_cap_mode),
                    "weak_mid_buckets": ",".join(weak_buckets),
                    "bets": pol_tot.bets,
                    "contracts": pol_tot.contracts,
                    "risked": round(pol_tot.risked, 6),
                    "fees": round(pol_tot.fees, 6),
                    "net_pnl": round(pol_tot.net_pnl, 6),
                    "roi_net": round(pol_tot.roi_net, 6),
                    "votes_contracts": votes_tot.contracts,
                    "votes_risked": round(votes_tot.risked, 6),
                    "votes_fees": round(votes_tot.fees, 6),
                    "votes_net_pnl": round(votes_tot.net_pnl, 6),
                    "votes_roi_net": round(votes_tot.roi_net, 6),
                    "votes_cap_contracts": votes_cap_tot.contracts,
                    "votes_cap_risked": round(votes_cap_tot.risked, 6),
                    "votes_cap_fees": round(votes_cap_tot.fees, 6),
                    "votes_cap_net_pnl": round(votes_cap_tot.net_pnl, 6),
                    "votes_cap_roi_net": round(votes_cap_tot.roi_net, 6),
                    "baseline_contracts": base_tot.contracts,
                    "baseline_risked": round(base_tot.risked, 6),
                    "baseline_fees": round(base_tot.fees, 6),
                    "baseline_net_pnl": round(base_tot.net_pnl, 6),
                    "baseline_roi_net": round(base_tot.roi_net, 6),
                }
            )

            for g in test:
                units, score, triggering = units_for_game(
                    g,
                    policy=policy,
                    cap_units=int(args.cap_units),
                    weak_buckets=weak_buckets,
                    weak_bucket_cap=int(args.weak_bucket_cap),
                )
                bucket = mid_bucket(g.mid_home)
                is_weak_bucket = bool(weak_bucket_set) and (bucket in weak_bucket_set)
                votes, vote_models = votes_for_game(g, models=list(args.models))
                units_votes = min(int(args.cap_units), 1 + int(votes))
                units_votes_cap = min(int(units_votes), int(args.weak_bucket_cap)) if is_weak_bucket else int(units_votes)
                gross, fees, risked = net_pnl_taker(contracts=units, price_away=g.price_away, home_win=int(g.home_win or 0))
                gross_b, fees_b, risked_b = net_pnl_taker(contracts=1, price_away=g.price_away, home_win=int(g.home_win or 0))
                gross_v, fees_v, risked_v = net_pnl_taker(contracts=units_votes, price_away=g.price_away, home_win=int(g.home_win or 0))
                gross_vc, fees_vc, risked_vc = net_pnl_taker(contracts=units_votes_cap, price_away=g.price_away, home_win=int(g.home_win or 0))
                trade_rows.append(
                    {
                        "date": g.date,
                        "league": g.league,
                        "matchup": g.matchup,
                        "mid_home": round(g.mid_home, 6),
                        "mid_bucket": bucket,
                        "weak_bucket": int(is_weak_bucket),
                        "price_away": round(g.price_away, 6),
                        "contracts": units,
                        "score": round(score, 6),
                        "triggering_models": ",".join(triggering),
                        "gross_pnl": round(gross, 6),
                        "fees": round(fees, 6),
                        "net_pnl": round(gross - fees, 6),
                        "risked": round(risked, 6),
                        "votes": int(votes),
                        "votes_models": ",".join(vote_models),
                        "votes_contracts": int(units_votes),
                        "votes_gross_pnl": round(gross_v, 6),
                        "votes_fees": round(fees_v, 6),
                        "votes_net_pnl": round(gross_v - fees_v, 6),
                        "votes_risked": round(risked_v, 6),
                        "votes_cap_contracts": int(units_votes_cap),
                        "votes_cap_gross_pnl": round(gross_vc, 6),
                        "votes_cap_fees": round(fees_vc, 6),
                        "votes_cap_net_pnl": round(gross_vc - fees_vc, 6),
                        "votes_cap_risked": round(risked_vc, 6),
                        "baseline_gross_pnl": round(gross_b, 6),
                        "baseline_fees": round(fees_b, 6),
                        "baseline_net_pnl": round(gross_b - fees_b, 6),
                        "baseline_risked": round(risked_b, 6),
                    }
                )

    summary_rows: List[Dict[str, object]] = []
    for league in leagues + ["overall"]:
        b = totals_base[league]
        p = totals_policy[league]
        v = totals_votes[league]
        vc = totals_votes_cap[league]
        summary_rows.append(
            {
                "window_start": args.start_date,
                "window_end": args.end_date,
                "league": league,
                "anchor_minutes": int(args.anchor_minutes),
                "slippage_cents": int(args.slippage_cents),
                "cap_units": int(args.cap_units),
                "confidence_level": float(args.confidence_level),
                "threshold_select_mode": str(args.threshold_select_mode),
                "roi_floor_mult": float(args.roi_floor_mult),
                "models": ",".join(list(args.models)),
                "policy_bets": int(p.bets),
                "policy_contracts": int(p.contracts),
                "policy_risked": round(p.risked, 6),
                "policy_fees": round(p.fees, 6),
                "policy_net_pnl": round(p.net_pnl, 6),
                "policy_roi_net": round(p.roi_net, 6),
                "votes_contracts": int(v.contracts),
                "votes_risked": round(v.risked, 6),
                "votes_fees": round(v.fees, 6),
                "votes_net_pnl": round(v.net_pnl, 6),
                "votes_roi_net": round(v.roi_net, 6),
                "votes_cap_contracts": int(vc.contracts),
                "votes_cap_risked": round(vc.risked, 6),
                "votes_cap_fees": round(vc.fees, 6),
                "votes_cap_net_pnl": round(vc.net_pnl, 6),
                "votes_cap_roi_net": round(vc.roi_net, 6),
                "baseline_bets": int(b.bets),
                "baseline_contracts": int(b.contracts),
                "baseline_risked": round(b.risked, 6),
                "baseline_fees": round(b.fees, 6),
                "baseline_net_pnl": round(b.net_pnl, 6),
                "baseline_roi_net": round(b.roi_net, 6),
            }
        )

    if args.no_write:
        for r in summary_rows:
            if r["league"] == "overall":
                print(r)
        return

    out_dir = Path(args.out_dir)
    tag = f"{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}"
    _write_csv(out_dir / f"walkforward_rule_a_unit_policy_summary_{tag}.csv", summary_rows)
    _write_csv(out_dir / f"walkforward_rule_a_unit_policy_daily_{tag}.csv", daily_rows)
    _write_csv(out_dir / f"walkforward_rule_a_unit_policy_modelcal_{tag}.csv", calib_rows)
    _write_csv(out_dir / f"walkforward_rule_a_unit_policy_trades_{tag}.csv", trade_rows)
    print(f"[ok] wrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
