#!/usr/bin/env python3
"""
Learn per-league model eligibility for Rule A voting (read-only).

Outputs a JSON that classifies models into:
  - primary: robustly positive (bootstrap 90% ROI lower bound > 0)
  - secondary: has enough games but not primary; counts only when at least one primary also triggers

This is designed to be learned walk-forward and then applied to future slates.

Notes on "bootstrap 90% ROI lower bound":
  - For each model we build a list of per-trade ROI samples (net / risk) using the same execution
    assumptions as the Rule A backtest (away YES ask + slippage + taker fees).
  - We resample those trades with replacement many times (bootstrap) to form a distribution of ROI.
  - `roi_lb90` is the 10th percentile of that distribution, i.e. a conservative ROI estimate with
    90% confidence. Primary models require `roi_lb90 > 0` and `n >= min_games`.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, load_games
from chimera_v2c.src.rule_a_model_eligibility import learn_eligibility
from chimera_v2c.src.rule_a_unit_policy import iter_rule_a_games, load_bidask_csv, net_pnl_taker
from chimera_v2c.src.rule_a_vote_calibration import VoteDeltaCalibration, model_contribution


def _default_bidask_path(*, start_date: str, end_date: str, anchor_minutes: int) -> Path:
    return Path("reports/market_snapshots") / (
        f"kalshi_bidask_tminus_{start_date.replace('-', '')}_{end_date.replace('-', '')}_m{int(anchor_minutes)}.csv"
    )


def _default_calibration_path(league: str) -> Path:
    return Path("chimera_v2c/data") / f"rule_a_vote_calibration_{league}.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learn per-league Rule A model eligibility (primary/secondary) from history.")
    p.add_argument("--league", required=True, choices=["nba", "nhl", "nfl"], help="League.")
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive.")
    p.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive (trained through).")
    p.add_argument("--anchor-minutes", type=int, default=30, help="T-minus anchor minutes (default: 30).")
    p.add_argument("--slippage-cents", type=int, default=1, help="Away ask slippage (default: 1).")
    p.add_argument("--bidask-csv", default="", help="Optional bid/ask CSV path; default derived from dates.")
    p.add_argument("--models", nargs="+", default=[], help="Models to evaluate (default: from vote calibration JSON if present).")
    p.add_argument("--vote-calibration-json", default="", help="Optional VoteDeltaCalibration JSON path to define trigger gates.")
    p.add_argument("--min-games", type=int, default=10, help="Min games per model to be eligible (default: 10).")
    p.add_argument("--confidence-level", type=float, default=0.9, help="Bootstrap confidence level (default: 0.9).")
    p.add_argument("--bootstrap-sims", type=int, default=2000, help="Bootstrap sims (default: 2000).")
    p.add_argument("--seed", type=int, default=7, help="Random seed (default: 7).")
    p.add_argument("--out", default="", help="Optional output JSON path (default: chimera_v2c/data/rule_a_model_eligibility_<league>.json).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    league = str(args.league)
    start = str(args.start_date).strip()
    end = str(args.end_date).strip()

    bidask_path = Path(str(args.bidask_csv).strip()) if str(args.bidask_csv).strip() else _default_bidask_path(
        start_date=start, end_date=end, anchor_minutes=int(args.anchor_minutes)
    )
    if not bidask_path.exists():
        raise SystemExit(f"[error] missing bid/ask CSV: {bidask_path}")

    if str(args.vote_calibration_json).strip():
        vote_cfg = VoteDeltaCalibration.load_json(Path(str(args.vote_calibration_json).strip()))
    else:
        p = _default_calibration_path(league)
        vote_cfg = VoteDeltaCalibration.load_json(p) if p.exists() else None
    if vote_cfg is None:
        raise SystemExit("[error] missing vote calibration JSON (pass --vote-calibration-json or create chimera_v2c/data/rule_a_vote_calibration_<league>.json)")

    if args.models:
        models = [str(m) for m in args.models]
    elif vote_cfg is not None:
        models = list(vote_cfg.models)
    else:
        raise SystemExit("[error] pass --models or provide a vote calibration JSON with models")

    # Load ledger games for the target league only + convert to Rule-A games (home-favored only; requires outcomes).
    ledger_games = load_games(daily_dir=LEDGER_DIR, start_date=start, end_date=end, league_filter=league, models=models)
    bidask = load_bidask_csv(bidask_path)
    ra_games = iter_rule_a_games(ledger_games, bidask=bidask, models=models, slippage_cents=int(args.slippage_cents), require_outcome=True)
    ra_games = [g for g in ra_games if str(g.league).strip().lower() == league]
    if not ra_games:
        raise SystemExit("[error] no qualifying Rule A games found in the given window")

    # Build per-model samples: for each game where the model would contribute under the vote calibration gates,
    # record the incremental 1-contract net and risk (same execution price used by RuleAGame).
    per_model: Dict[str, List[Tuple[float, float]]] = {m: [] for m in models}
    for g in ra_games:
        for m in models:
            if vote_cfg is None:
                # default gates: treat as a vote when p_home < mid_home and edge_net >= 0,
                # approximated by using VoteDeltaCalibration's helper with defaults.
                # For eligibility learning, we want to mirror production, so require an explicit vote_cfg.
                continue
            add, _ = model_contribution(
                g,
                model=m,
                vote_delta_default=float(vote_cfg.vote_delta_default),
                vote_delta_by_model=dict(vote_cfg.vote_delta_by_model),
                vote_edge_default=float(vote_cfg.vote_edge_default),
                vote_edge_by_model=dict(vote_cfg.vote_edge_by_model),
                flip_delta_mode=str(vote_cfg.flip_delta_mode),
                vote_weight=int(vote_cfg.vote_weight),
                flip_weight=int(vote_cfg.flip_weight),
            )
            if int(add) <= 0:
                continue
            gross, fees, risked = net_pnl_taker(contracts=1, price_away=float(g.price_away), home_win=int(g.home_win or 0))
            per_model[m].append((float(gross - fees), float(risked)))

    # Learn eligibility; treat models with n<min_games as ineligible (neither primary nor secondary).
    elig = learn_eligibility(
        league=league,
        trained_through=end,
        per_model_samples=per_model,
        min_games=int(args.min_games),
        confidence_level=float(args.confidence_level),
        bootstrap_sims=int(args.bootstrap_sims),
        seed=int(args.seed),
    )

    out_path = Path(str(args.out).strip()) if str(args.out).strip() else Path("chimera_v2c/data") / f"rule_a_model_eligibility_{league}.json"
    elig.save_json(out_path)
    print(f"[ok] wrote {out_path}")
    print(f"[info] primary={elig.primary_models} secondary={elig.secondary_models}")


if __name__ == "__main__":
    main()
