#!/usr/bin/env python
"""
Paper trade sheet: Rule-A unit sizing policy using Kalshi bid/ask snapshots (taker-only; fee-aware).

This planner trains ONLY on dates < target_date (walk-forward; no leakage),
then emits per-game Rule-A recommended contract sizes (1..cap_units) for the
target date, using the away YES ask (+ slippage) as the execution price.

See chimera_v2c/tools/walkforward_rule_a_unit_policy.py for the full policy definition.

Safety: read-only on daily ledgers; writes a CSV under reports/trade_sheets/.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.rule_a_unit_policy import (
    PolicyCalibration,
    expected_edge_net_per_contract,
    learn_weak_mid_buckets_for_votes,
    iter_rule_a_games,
    load_bidask_csv,
    mid_bucket,
    train_rule_a_policy,
    units_for_game,
    votes_for_game,
)


DEFAULT_MODELS = ["v2c", "grok", "gemini", "gpt", "market_proxy", "moneypuck"]
DEFAULT_THRESHOLDS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
DEFAULT_UNIT_SCALES = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 1e9]


DEFAULT_OUT_DIR = Path("reports/trade_sheets")


@dataclass(frozen=True)
class PlanMeta:
    trained: bool
    train_days_used: int
    confidence_level: float
    unit_scale: float
    active_models: str


def _normalize_league(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"nba", "nhl", "nfl"}:
        return v
    raise SystemExit("[error] --league must be one of: nba, nhl, nfl")


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


def plan_trades(
    games: Sequence[GameRow],
    *,
    bidask_csv: Path,
    target_date: str,
    league: str,
    models: Sequence[str],
    thresholds: Sequence[float],
    unit_scales: Sequence[float],
    cap_units: int,
    slippage_cents: int,
    train_days: int,
    min_train_days: int,
    min_signals: int,
    confidence_level: float,
    bootstrap_sims: int,
    roi_floor_mult: float,
    threshold_select_mode: str,
    seed: int,
    bucket_cap_mode: str,
    min_bucket_bets: int,
    weak_bucket_cap: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    Core planner (pure-ish): takes already-loaded ledger games and returns (rows, meta).
    """
    bidask = load_bidask_csv(bidask_csv)
    ra = iter_rule_a_games(
        games,
        bidask=bidask,
        models=models,
        slippage_cents=int(slippage_cents),
        require_outcome=False,
    )
    ra = [g for g in ra if g.league == league]
    if not ra:
        return [], {"reason": "no_rule_a_games"}

    # Train on dates < target_date.
    train_all = [g for g in ra if g.date < target_date and g.home_win is not None]
    train_dates = sorted({g.date for g in train_all})
    if train_days > 0:
        keep = set(train_dates[-int(train_days) :])
        train = [g for g in train_all if g.date in keep]
        train_dates = sorted({g.date for g in train})
    else:
        train = train_all

    trained = len(train_dates) >= int(min_train_days)
    if not trained:
        weak_buckets: List[str] = []
        policy = PolicyCalibration(league=league, confidence_level=float(confidence_level), unit_scale=1e9, models={})
    else:
        weak_buckets = []
        bucket_metrics: Dict[str, object] = {}
        if str(bucket_cap_mode) == "votes_mean_low":
            bm = learn_weak_mid_buckets_for_votes(
                train,
                models=list(models),
                cap_units=int(cap_units),
                confidence_level=float(confidence_level),
                bootstrap_sims=int(bootstrap_sims),
                min_bucket_bets=int(min_bucket_bets),
                seed=int(seed) ^ 0xA11CE,
            )
            bucket_metrics = {
                b: {"n": m.n, "mean_net": m.mean_net, "mean_low": m.mean_low, "conf_gt0": m.conf_gt0} for b, m in bm.items()
            }
            weak_buckets = sorted([b for b, m in bm.items() if float(m.mean_low) < 0.0])

        policy = train_rule_a_policy(
            train,
            league=league,
            models=models,
            thresholds=thresholds,
            unit_scales=unit_scales,
            cap_units=int(cap_units),
            min_signals=int(min_signals),
            confidence_level=float(confidence_level),
            bootstrap_sims=int(bootstrap_sims),
            roi_floor_mult=float(roi_floor_mult),
            threshold_select_mode=str(threshold_select_mode),
            seed=int(seed),
            weak_buckets=weak_buckets,
            weak_bucket_cap=int(weak_bucket_cap),
        )

    test = [g for g in ra if g.date == target_date]
    if not test:
        return [], {"reason": "no_target_games", "trained": bool(trained)}

    rows: List[Dict[str, object]] = []
    weak_bucket_set = set(weak_buckets)
    for g in sorted(test, key=lambda x: x.matchup):
        bucket = mid_bucket(g.mid_home)
        is_weak_bucket = bool(weak_bucket_set) and (bucket in weak_bucket_set)
        units, score, triggering = units_for_game(
            g,
            policy=policy,
            cap_units=int(cap_units),
            weak_buckets=weak_buckets,
            weak_bucket_cap=int(weak_bucket_cap),
        )
        votes, vote_models = votes_for_game(g, models=models)
        units_votes = min(int(cap_units), 1 + int(votes))
        units_votes_cap = min(int(units_votes), int(weak_bucket_cap)) if is_weak_bucket else int(units_votes)

        p_cols: Dict[str, object] = {}
        edge_cols: Dict[str, object] = {}
        for m in models:
            p = g.probs.get(str(m))
            p_cols[f"p_home_{m}"] = "" if p is None else f"{float(p):.6f}"
            edge_cols[f"edge_net_{m}"] = "" if p is None else f"{expected_edge_net_per_contract(p_home=float(p), price_away=float(g.price_away)):.6f}"
        rows.append(
            {
                "date": g.date,
                "league": g.league,
                "matchup": g.matchup,
                "event_ticker": g.event_ticker or "",
                "market_ticker_home": g.market_ticker_home or "",
                "market_ticker_away": g.market_ticker_away or "",
                "mid_home": f"{float(g.mid_home):.6f}",
                "mid_bucket": bucket,
                "weak_bucket": int(is_weak_bucket),
                "price_away": f"{float(g.price_away):.6f}",
                "contracts": int(units),
                "score": f"{float(score):.6f}",
                "triggering_models": ",".join(triggering),
                "votes": int(votes),
                "votes_models": ",".join(vote_models),
                "votes_contracts": int(units_votes),
                "votes_contracts_cap": int(units_votes_cap),
                "active_models": ",".join(sorted(policy.models.keys())),
                "confidence_level": float(policy.confidence_level),
                "unit_scale": float(policy.unit_scale),
                "trained": bool(trained),
                "train_days_used": len(train_dates),
                **p_cols,
                **edge_cols,
            }
        )

    meta = {
        "trained": bool(trained),
        "train_days_used": len(train_dates),
        "train_start": train_dates[0] if train_dates else "",
        "train_end": train_dates[-1] if train_dates else "",
        "confidence_level": float(policy.confidence_level),
        "unit_scale": float(policy.unit_scale),
        "active_models": ",".join(sorted(policy.models.keys())),
        "cap_units": int(cap_units),
        "slippage_cents": int(slippage_cents),
        "bidask_csv": str(bidask_csv),
        "bucket_cap_mode": str(bucket_cap_mode),
        "min_bucket_bets": int(min_bucket_bets),
        "weak_bucket_cap": int(weak_bucket_cap),
        "weak_mid_buckets": ",".join(weak_buckets),
        "bucket_metrics_json": str(bucket_metrics) if trained and str(bucket_cap_mode) == "votes_mean_low" else "",
    }
    return rows, meta


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plan Rule-A unit-sized trades for a target date (taker-only; bid/ask at T-minus).")
    ap.add_argument("--target-date", required=True, help="YYYY-MM-DD target ledger date.")
    ap.add_argument("--league", required=True, help="nba|nhl|nfl")
    ap.add_argument("--start-date", default="", help="Optional YYYY-MM-DD earliest ledger date to load (default: all).")
    ap.add_argument("--anchor-minutes", type=int, default=30, help="T-minus anchor in minutes (default: 30).")
    ap.add_argument("--bidask-csv", default="", help="Path to kalshi_bidask_tminus_*.csv (default derived from dates when start-date is set).")
    ap.add_argument("--export-missing", action="store_true", help="Export bid/ask snapshot CSV if missing (requires start-date).")
    ap.add_argument("--slippage-cents", type=int, default=0, help="Extra slippage to add on top of away ask (default: 0).")
    ap.add_argument("--cap-units", type=int, default=10, help="Max contracts per game (default: 10).")

    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help=f"Models to consider (default: {' '.join(DEFAULT_MODELS)}).")
    ap.add_argument("--train-days", type=int, default=0, help="Rolling train window in days (0 = expanding).")
    ap.add_argument("--min-train-days", type=int, default=5, help="Minimum distinct train days required (default: 5).")
    ap.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS, help="Delta thresholds grid (default preset).")
    ap.add_argument("--unit-scales", nargs="+", type=float, default=DEFAULT_UNIT_SCALES, help="Candidate unit_scale values (default preset).")
    ap.add_argument("--min-signals", type=int, default=10, help="Min signal games per (league,model,threshold) (default: 10).")
    ap.add_argument("--confidence-level", type=float, default=0.9, help="Bootstrap confidence level (default: 0.90).")
    ap.add_argument("--bootstrap-sims", type=int, default=1000, help="Bootstrap sims for mean distribution (default: 1000).")
    ap.add_argument(
        "--threshold-select-mode",
        choices=["max_total_mean_low", "max_mean_low", "min_threshold"],
        default="max_total_mean_low",
        help="How to choose t_m on train-only data (default: max_total_mean_low).",
    )
    ap.add_argument("--roi-floor-mult", type=float, default=0.9, help="ROI floor multiplier vs baseline when selecting unit_scale.")
    ap.add_argument(
        "--bucket-cap-mode",
        choices=["none", "votes_mean_low"],
        default="votes_mean_low",
        help="Optional risk cap for weak mid-home buckets, learned from train only (default: votes_mean_low).",
    )
    ap.add_argument("--min-bucket-bets", type=int, default=30, help="Min train games per mid bucket (default: 30).")
    ap.add_argument("--weak-bucket-cap", type=int, default=3, help="Max contracts in weak mid buckets (default: 3).")
    ap.add_argument("--seed", type=int, default=1337, help="Base RNG seed for bootstraps (default: 1337).")

    ap.add_argument("--out", default="", help="Optional output CSV path (default under reports/trade_sheets).")
    ap.add_argument("--no-write", action="store_true", help="Print meta only; do not write CSV.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    league = _normalize_league(args.league)
    start_date = args.start_date.strip() or None

    if args.bidask_csv:
        bidask_path = Path(args.bidask_csv)
    else:
        if not start_date:
            raise SystemExit("[error] provide --bidask-csv or set --start-date so the default path can be derived")
        bidask_path = Path(
            f"reports/market_snapshots/kalshi_bidask_tminus_{start_date.replace('-', '')}_{args.target_date.replace('-', '')}_m{int(args.anchor_minutes)}.csv"
        )

    if args.export_missing:
        if not start_date:
            raise SystemExit("[error] --export-missing requires --start-date (range start)")
        _ensure_bidask_csv(
            start_date=str(start_date),
            end_date=args.target_date,
            minutes=int(args.anchor_minutes),
            out_path=bidask_path,
            export_missing=True,
        )
    if not bidask_path.exists():
        raise SystemExit(f"[error] missing bid/ask csv: {bidask_path}")

    games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=start_date,
        end_date=args.target_date,
        league_filter=league,
        models=list(args.models),
    )
    if not games:
        raise SystemExit("[error] no ledger games loaded for window")

    rows, meta = plan_trades(
        games,
        bidask_csv=bidask_path,
        target_date=args.target_date,
        league=league,
        models=list(args.models),
        thresholds=[float(t) for t in args.thresholds],
        unit_scales=[float(s) for s in args.unit_scales],
        cap_units=int(args.cap_units),
        slippage_cents=int(args.slippage_cents),
        train_days=int(args.train_days),
        min_train_days=int(args.min_train_days),
        min_signals=int(args.min_signals),
        confidence_level=float(args.confidence_level),
        bootstrap_sims=int(args.bootstrap_sims),
        roi_floor_mult=float(args.roi_floor_mult),
        threshold_select_mode=str(args.threshold_select_mode),
        seed=int(args.seed),
        bucket_cap_mode=str(args.bucket_cap_mode),
        min_bucket_bets=int(args.min_bucket_bets),
        weak_bucket_cap=int(args.weak_bucket_cap),
    )

    if args.no_write:
        print(meta)
        return

    out_path = Path(args.out) if args.out else DEFAULT_OUT_DIR / f"ruleA_unit_policy_trades_{league}_{args.target_date.replace('-', '')}.csv"
    _write_csv(out_path, rows)
    print(f"[ok] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
