#!/usr/bin/env python
"""
Learn per-league "weak" mid-home buckets for Rule A (taker) and write a JSON config.

Weak buckets are defined as buckets whose bootstrap lower bound on mean net-per-contract
is negative for the chosen sizing policy; these buckets should be capped (not excluded).

This tool is read-only on daily ledgers; it uses bid/ask snapshots at a fixed T-minus anchor.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, load_games
from chimera_v2c.src.rule_a_unit_policy import iter_rule_a_games, load_bidask_csv
from chimera_v2c.src.rule_a_vote_calibration import VoteDeltaCalibration, learn_weak_mid_buckets_for_weighted_votes


DEFAULT_MODELS = ["v2c", "grok", "gemini", "gpt", "market_proxy", "moneypuck"]


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Learn Rule-A weak buckets (bootstrap mean-low) and write JSON.")
    ap.add_argument("--league", required=True, help="nba|nhl|nfl")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive.")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive.")
    ap.add_argument("--anchor-minutes", type=int, default=30, help="T-minus anchor in minutes (default: 30).")
    ap.add_argument("--bidask-csv", default="", help="Path to kalshi_bidask_tminus_*.csv (default derived from dates).")
    ap.add_argument("--export-missing", action="store_true", help="Export bid/ask snapshot CSV if missing.")
    ap.add_argument("--slippage-cents", type=int, default=1, help="Extra slippage to add on top of away ask (default: 1).")

    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help=f"Models to consider (default: {' '.join(DEFAULT_MODELS)}).")
    ap.add_argument("--vote-edge", type=float, default=0.0, help="Fee-aware edge gate (default: 0.0).")
    ap.add_argument("--base-units", type=int, default=1, help="Base contracts per qualifying game (default: 1).")
    ap.add_argument("--cap-units", type=int, default=10, help="Max contracts per game (default: 10).")
    ap.add_argument("--vote-weight", type=int, default=1, help="Contracts per normal vote (default: 1).")
    ap.add_argument("--flip-weight", type=int, default=2, help="Contracts per flip/I (default: 2).")
    ap.add_argument("--flip-delta-mode", choices=["none", "same"], default="none", help="Whether flip requires delta gate (default: none).")

    ap.add_argument("--confidence-level", type=float, default=0.9, help="Bootstrap confidence level (default: 0.90).")
    ap.add_argument("--bootstrap-sims", type=int, default=2000, help="Bootstrap sims (default: 2000).")
    ap.add_argument("--min-bucket-bets", type=int, default=30, help="Min games per bucket to evaluate (default: 30).")
    ap.add_argument("--seed", type=int, default=1337, help="Seed (default: 1337).")

    ap.add_argument(
        "--calibration-json",
        default="",
        help="Optional VoteDeltaCalibration JSON to use for deltas/edges/flip-mode/eligibility.",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: chimera_v2c/data/rule_a_weak_buckets_<league>.json).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    league = _normalize_league(args.league)

    bidask_path = Path(args.bidask_csv) if args.bidask_csv else Path(
        f"reports/market_snapshots/kalshi_bidask_tminus_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}_m{int(args.anchor_minutes)}.csv"
    )
    _ensure_bidask_csv(
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        minutes=int(args.anchor_minutes),
        out_path=bidask_path,
        export_missing=bool(args.export_missing),
    )
    if not bidask_path.exists():
        raise SystemExit(f"[error] missing bid/ask csv: {bidask_path} (pass --export-missing or --bidask-csv)")

    vote_delta_default = 0.0
    vote_delta_by_model: Dict[str, float] = {str(m): 0.0 for m in args.models}
    vote_edge_default = float(args.vote_edge)
    vote_edge_by_model: Dict[str, float] = {}
    flip_delta_mode = str(args.flip_delta_mode)
    vote_weight = int(args.vote_weight)
    flip_weight = int(args.flip_weight)
    models = list(args.models)

    if str(args.calibration_json).strip():
        calib = VoteDeltaCalibration.load_json(Path(str(args.calibration_json).strip()))
        vote_delta_default = float(calib.vote_delta_default)
        vote_delta_by_model = dict(calib.vote_delta_by_model)
        vote_edge_default = float(calib.vote_edge_default)
        vote_edge_by_model = dict(calib.vote_edge_by_model)
        flip_delta_mode = str(calib.flip_delta_mode)
        vote_weight = int(calib.vote_weight)
        flip_weight = int(calib.flip_weight)
        models = list(calib.models)

    bidask = load_bidask_csv(bidask_path)
    ledger_games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        league_filter=league,
        models=models,
    )
    if not ledger_games:
        raise SystemExit("[error] no ledger games loaded for window")

    rule_a_games = iter_rule_a_games(
        ledger_games,
        bidask=bidask,
        models=models,
        slippage_cents=int(args.slippage_cents),
        require_outcome=True,
    )
    if not rule_a_games:
        raise SystemExit("[error] no Rule-A qualifying graded games found for window + snapshot")

    metrics = learn_weak_mid_buckets_for_weighted_votes(
        rule_a_games,
        models=models,
        vote_delta_default=float(vote_delta_default),
        vote_delta_by_model=vote_delta_by_model,
        vote_edge_default=float(vote_edge_default),
        vote_edge_by_model=vote_edge_by_model,
        flip_delta_mode=str(flip_delta_mode),
        vote_weight=int(vote_weight),
        flip_weight=int(flip_weight),
        cap_units=int(args.cap_units),
        base_units=int(args.base_units),
        confidence_level=float(args.confidence_level),
        bootstrap_sims=int(args.bootstrap_sims),
        min_bucket_bets=int(args.min_bucket_bets),
        seed=int(args.seed),
    )

    weak_buckets = sorted([b for b, m in metrics.items() if float(m.mean_low) < 0.0])
    out_path = Path(args.out) if args.out else Path("chimera_v2c/data") / f"rule_a_weak_buckets_{league}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "league": league,
                "window_start": str(args.start_date),
                "window_end": str(args.end_date),
                "anchor_minutes": int(args.anchor_minutes),
                "slippage_cents": int(args.slippage_cents),
                "confidence_level": float(args.confidence_level),
                "bootstrap_sims": int(args.bootstrap_sims),
                "min_bucket_bets": int(args.min_bucket_bets),
                "models": list(models),
                "vote_edge_default": float(vote_edge_default),
                "vote_delta_default": float(vote_delta_default),
                "flip_delta_mode": str(flip_delta_mode),
                "vote_weight": int(vote_weight),
                "flip_weight": int(flip_weight),
                "weak_buckets": weak_buckets,
                "bucket_metrics": {
                    b: {
                        "n": int(m.n),
                        "mean_net_per_contract": float(m.mean_net),
                        "mean_low": float(m.mean_low),
                        "conf_gt0": float(m.conf_gt0),
                    }
                    for b, m in sorted(metrics.items())
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    print(f"[ok] wrote weak buckets JSON: {out_path}")
    print("[info] weak buckets: " + (", ".join(weak_buckets) if weak_buckets else "(none)"))


if __name__ == "__main__":
    main()

