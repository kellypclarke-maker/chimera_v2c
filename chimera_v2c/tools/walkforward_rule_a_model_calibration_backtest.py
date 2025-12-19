#!/usr/bin/env python
"""
Walk-forward (train->test) backtest: does per-model Platt calibration improve Rule-A PnL/ROI?

For each test day D:
  - Fit a Platt scaler for each selected model on all prior days (<D) using daily-ledger outcomes
    (league-matched, any games with outcomes; not restricted to Rule-A).
  - Apply those scalers to the model probabilities for Rule-A qualifying games on D.
  - Compare Rule-A PnL/ROI under execution realism (taker @ away ask + fees + slippage):
      baseline: same policy using raw probabilities
      calibrated: same policy using calibrated probabilities for chosen models

Policy definition here matches the Rule-A taker track:
  - Qualifying: Kalshi home YES mid > 0.50 at the anchor
  - Trade: buy AWAY YES at AWAY YES ask (+ optional slippage)
  - Sizing: weighted votes (vote=1, flip=2) with fee-aware edge gating (edge_net >= vote_edge)
  - Optional: apply a VoteDeltaCalibration JSON (per-model deltas/weights/flip gating)
  - Optional: apply weak bucket caps (cap contracts in mid buckets listed in JSON)

This tool is read-only on daily ledgers; it writes derived CSVs under reports/.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chimera_v2c.src.calibration import PlattScaler, fit_platt
from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.rule_a_unit_policy import RuleAGame, iter_rule_a_games, load_bidask_csv, mid_bucket, net_pnl_taker
from chimera_v2c.src.rule_a_vote_calibration import VoteDeltaCalibration, contracts_for_game


DEFAULT_MODELS = ["v2c", "grok", "gemini", "gpt", "market_proxy", "moneypuck"]
DEFAULT_CALIBRATE = ["v2c", "grok", "gemini", "gpt"]


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


def _iso(d) -> str:
    return d.strftime("%Y-%m-%d")


def _fit_scaler(pairs: List[Tuple[float, int]], *, min_samples: int) -> PlattScaler:
    if len(pairs) < int(min_samples):
        return PlattScaler(a=1.0, b=0.0)
    return fit_platt(pairs)


def _apply_scalers(
    g: RuleAGame,
    *,
    scalers: Dict[str, PlattScaler],
    models_to_calibrate: Sequence[str],
) -> RuleAGame:
    probs = dict(g.probs)
    for m in models_to_calibrate:
        if m not in probs:
            continue
        probs[m] = float(scalers[m].predict(float(probs[m])))
    return replace(g, probs=probs)


def _load_weak_buckets(path: Optional[str]) -> Tuple[List[str], int]:
    if not path:
        return [], 0
    p = Path(str(path))
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        return [], 0
    wb = d.get("weak_buckets")
    cap = d.get("weak_bucket_cap")
    buckets = [str(x) for x in wb] if isinstance(wb, list) else []
    cap_i = int(cap) if cap is not None else 0
    return buckets, cap_i


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Walk-forward backtest: Platt-calibrated model probs for Rule A taker policy.")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive (analysis window).")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive (analysis window).")
    ap.add_argument("--league", default="all", help="nba|nhl|nfl|all (default: all).")
    ap.add_argument("--anchor-minutes", type=int, default=30, help="T-minus anchor minutes (default: 30).")
    ap.add_argument("--bidask-csv", default="", help="Path to kalshi_bidask_tminus_*.csv (default derived from dates).")
    ap.add_argument("--export-missing", action="store_true", help="Export bid/ask snapshot CSV if missing.")
    ap.add_argument("--slippage-cents", type=int, default=1, help="Extra slippage on away ask (default: 1).")

    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Signal models available on Rule-A games.")
    ap.add_argument(
        "--calibrate-models",
        nargs="+",
        default=DEFAULT_CALIBRATE,
        help="Which models to Platt-calibrate (default: v2c grok gemini gpt).",
    )
    ap.add_argument("--min-samples", type=int, default=30, help="Min train samples to fit Platt (default: 30).")

    ap.add_argument("--vote-calibration-json", default="", help="Optional VoteDeltaCalibration JSON (policy thresholds/weights).")
    ap.add_argument("--weak-buckets-json", default="", help="Optional weak bucket caps JSON (policy caps).")
    ap.add_argument("--weak-bucket-cap", type=int, default=3, help="Fallback cap when weak-buckets-json has no cap (default: 3).")

    ap.add_argument("--cap-units", type=int, default=10, help="Max contracts per game (default: 10).")
    ap.add_argument("--base-units", type=int, default=1, help="Base contracts per qualifying game (default: 1).")
    ap.add_argument("--vote-weight", type=int, default=1, help="Contracts per normal vote (default: 1).")
    ap.add_argument("--flip-weight", type=int, default=2, help="Contracts per flip/I (default: 2).")
    ap.add_argument("--vote-edge", type=float, default=0.0, help="Fee-aware edge gate (default: 0.0).")
    ap.add_argument("--flip-delta-mode", choices=["none", "same"], default="none", help="Whether flip requires delta gate (default: none).")
    ap.add_argument("--vote-delta-default", type=float, default=0.0, help="Default vote delta (default: 0.0).")

    ap.add_argument("--out-dir", default="reports/thesis_summaries", help="Output dir (default: reports/thesis_summaries).")
    ap.add_argument("--no-write", action="store_true", help="Print overall summary only; do not write CSVs.")
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

    games_all = load_games(
        daily_dir=LEDGER_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=league_filter,
        models=list(set(list(args.models) + list(args.calibrate_models))),
    )
    if not games_all:
        raise SystemExit("[error] no ledger games loaded for window")

    ra = iter_rule_a_games(
        games_all,
        bidask=bidask,
        models=list(args.models),
        slippage_cents=int(args.slippage_cents),
        require_outcome=True,
    )
    if not ra:
        raise SystemExit("[error] no Rule-A qualifying graded games found for window + snapshot")

    leagues = sorted({g.league for g in ra})
    dates = sorted({g.date for g in ra})

    # Index all games (not just Rule-A) for calibration training.
    by_date_league_all: Dict[Tuple[str, str], List[GameRow]] = defaultdict(list)
    for g in games_all:
        d = _iso(g.date)
        if g.home_win not in (0.0, 1.0):
            continue
        by_date_league_all[(d, g.league)].append(g)

    # Index Rule-A games for evaluation.
    by_date_league_ra: Dict[Tuple[str, str], List[RuleAGame]] = defaultdict(list)
    for g in ra:
        by_date_league_ra[(g.date, g.league)].append(g)

    vote_cfg: Optional[VoteDeltaCalibration] = None
    if str(args.vote_calibration_json).strip():
        vote_cfg = VoteDeltaCalibration.load_json(Path(str(args.vote_calibration_json).strip()))

    weak_buckets, weak_cap_json = _load_weak_buckets(str(args.weak_buckets_json).strip() or None)
    weak_set = set(weak_buckets)
    weak_cap = int(weak_cap_json) if int(weak_cap_json) > 0 else int(args.weak_bucket_cap)

    # Policy parameters (defaults are baseline weighted votes w/ fee-aware edge gate).
    policy_vote_delta_default = float(args.vote_delta_default)
    policy_vote_delta_by_model: Dict[str, float] = {str(m): 0.0 for m in args.models}
    policy_vote_edge_default = float(args.vote_edge)
    policy_vote_edge_by_model: Dict[str, float] = {}
    policy_flip_delta_mode = str(args.flip_delta_mode)
    policy_vote_weight = int(args.vote_weight)
    policy_flip_weight = int(args.flip_weight)
    policy_cap_units = int(args.cap_units)
    policy_base_units = int(args.base_units)
    policy_models = list(args.models)

    if vote_cfg is not None:
        policy_vote_delta_default = float(vote_cfg.vote_delta_default)
        policy_vote_delta_by_model = dict(vote_cfg.vote_delta_by_model)
        policy_vote_edge_default = float(vote_cfg.vote_edge_default)
        policy_vote_edge_by_model = dict(vote_cfg.vote_edge_by_model)
        policy_flip_delta_mode = str(vote_cfg.flip_delta_mode)
        policy_vote_weight = int(vote_cfg.vote_weight)
        policy_flip_weight = int(vote_cfg.flip_weight)
        policy_cap_units = int(vote_cfg.cap_units)
        policy_base_units = int(vote_cfg.base_units)
        policy_models = list(vote_cfg.models)

    # Totals per (league, variant).
    totals: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    daily_rows: List[Dict[str, object]] = []
    calib_rows: List[Dict[str, object]] = []

    for test_date in dates:
        for league in leagues:
            test_ra = by_date_league_ra.get((test_date, league), [])
            if not test_ra:
                continue
            train_dates = [d for d in dates if d < test_date]
            train_games = [g for d in train_dates for g in by_date_league_all.get((d, league), [])]

            scalers: Dict[str, PlattScaler] = {}
            for m in args.calibrate_models:
                pairs = []
                for g in train_games:
                    p = g.probs.get(str(m))
                    if p is None:
                        continue
                    y = g.home_win
                    if y not in (0.0, 1.0):
                        continue
                    pairs.append((float(p), int(y)))
                scaler = _fit_scaler(pairs, min_samples=int(args.min_samples))
                scalers[str(m)] = scaler
                calib_rows.append(
                    {
                        "date": test_date,
                        "league": league,
                        "model": str(m),
                        "train_samples": int(len(pairs)),
                        "a": float(scaler.a),
                        "b": float(scaler.b),
                        "min_samples": int(args.min_samples),
                    }
                )

            # Per-day totals
            base_net = 0.0
            base_risk = 0.0
            base_fees = 0.0
            base_contracts = 0
            cal_net = 0.0
            cal_risk = 0.0
            cal_fees = 0.0
            cal_contracts = 0

            for g in test_ra:
                # baseline
                c_base, _ = contracts_for_game(
                    g,
                    models=policy_models,
                    vote_delta_default=policy_vote_delta_default,
                    vote_delta_by_model=policy_vote_delta_by_model,
                    vote_edge_default=policy_vote_edge_default,
                    vote_edge_by_model=policy_vote_edge_by_model,
                    flip_delta_mode=policy_flip_delta_mode,
                    vote_weight=policy_vote_weight,
                    flip_weight=policy_flip_weight,
                    cap_units=policy_cap_units,
                    base_units=policy_base_units,
                )
                if weak_set and mid_bucket(float(g.mid_home)) in weak_set:
                    c_base = min(int(c_base), int(weak_cap))
                gross_b, fees_b, risked_b = net_pnl_taker(contracts=int(c_base), price_away=float(g.price_away), home_win=int(g.home_win or 0))
                base_net += float(gross_b - fees_b)
                base_fees += float(fees_b)
                base_risk += float(risked_b)
                base_contracts += int(c_base)

                # calibrated
                g_cal = _apply_scalers(g, scalers=scalers, models_to_calibrate=list(args.calibrate_models))
                c_cal, _ = contracts_for_game(
                    g_cal,
                    models=policy_models,
                    vote_delta_default=policy_vote_delta_default,
                    vote_delta_by_model=policy_vote_delta_by_model,
                    vote_edge_default=policy_vote_edge_default,
                    vote_edge_by_model=policy_vote_edge_by_model,
                    flip_delta_mode=policy_flip_delta_mode,
                    vote_weight=policy_vote_weight,
                    flip_weight=policy_flip_weight,
                    cap_units=policy_cap_units,
                    base_units=policy_base_units,
                )
                if weak_set and mid_bucket(float(g_cal.mid_home)) in weak_set:
                    c_cal = min(int(c_cal), int(weak_cap))
                gross_c, fees_c, risked_c = net_pnl_taker(contracts=int(c_cal), price_away=float(g_cal.price_away), home_win=int(g_cal.home_win or 0))
                cal_net += float(gross_c - fees_c)
                cal_fees += float(fees_c)
                cal_risk += float(risked_c)
                cal_contracts += int(c_cal)

            base_roi = 0.0 if base_risk <= 0 else base_net / base_risk
            cal_roi = 0.0 if cal_risk <= 0 else cal_net / cal_risk

            daily_rows.append(
                {
                    "date": test_date,
                    "league": league,
                    "bets": int(len(test_ra)),
                    "baseline_net_pnl": round(base_net, 6),
                    "baseline_roi_net": round(base_roi, 6),
                    "baseline_contracts": int(base_contracts),
                    "baseline_risked": round(base_risk, 6),
                    "baseline_fees": round(base_fees, 6),
                    "calibrated_net_pnl": round(cal_net, 6),
                    "calibrated_roi_net": round(cal_roi, 6),
                    "calibrated_contracts": int(cal_contracts),
                    "calibrated_risked": round(cal_risk, 6),
                    "calibrated_fees": round(cal_fees, 6),
                    "delta_net_pnl": round(cal_net - base_net, 6),
                    "delta_roi_net": round(cal_roi - base_roi, 6),
                    "calibrate_models": ",".join(list(args.calibrate_models)),
                    "policy_models": ",".join(list(policy_models)),
                }
            )

            for lg in (league, "overall"):
                totals[(lg, "baseline")]["net"] += base_net
                totals[(lg, "baseline")]["risk"] += base_risk
                totals[(lg, "baseline")]["fees"] += base_fees
                totals[(lg, "baseline")]["contracts"] += base_contracts
                totals[(lg, "calibrated")]["net"] += cal_net
                totals[(lg, "calibrated")]["risk"] += cal_risk
                totals[(lg, "calibrated")]["fees"] += cal_fees
                totals[(lg, "calibrated")]["contracts"] += cal_contracts

    summary_rows: List[Dict[str, object]] = []
    for lg in leagues + ["overall"]:
        b = totals[(lg, "baseline")]
        c = totals[(lg, "calibrated")]
        broi = 0.0 if b["risk"] <= 0 else b["net"] / b["risk"]
        croi = 0.0 if c["risk"] <= 0 else c["net"] / c["risk"]
        summary_rows.append(
            {
                "window_start": args.start_date,
                "window_end": args.end_date,
                "league": lg,
                "anchor_minutes": int(args.anchor_minutes),
                "slippage_cents": int(args.slippage_cents),
                "calibrate_models": ",".join(list(args.calibrate_models)),
                "policy_models": ",".join(list(policy_models)),
                "baseline_net_pnl": round(float(b["net"]), 6),
                "baseline_roi_net": round(float(broi), 6),
                "baseline_contracts": int(b["contracts"]),
                "baseline_risked": round(float(b["risk"]), 6),
                "baseline_fees": round(float(b["fees"]), 6),
                "calibrated_net_pnl": round(float(c["net"]), 6),
                "calibrated_roi_net": round(float(croi), 6),
                "calibrated_contracts": int(c["contracts"]),
                "calibrated_risked": round(float(c["risk"]), 6),
                "calibrated_fees": round(float(c["fees"]), 6),
                "delta_net_pnl": round(float(c["net"] - b["net"]), 6),
                "delta_roi_net": round(float(croi - broi), 6),
                "vote_calibration_json": str(args.vote_calibration_json),
                "weak_buckets_json": str(args.weak_buckets_json),
            }
        )

    if args.no_write:
        for r in summary_rows:
            if r["league"] == "overall":
                print(r)
        return

    out_dir = Path(args.out_dir)
    tag = f"{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}_m{int(args.anchor_minutes)}_s{int(args.slippage_cents)}"
    _write_csv(out_dir / f"walkforward_rule_a_model_calibration_summary_{tag}.csv", summary_rows)
    _write_csv(out_dir / f"walkforward_rule_a_model_calibration_daily_{tag}.csv", daily_rows)
    _write_csv(out_dir / f"walkforward_rule_a_model_calibration_params_{tag}.csv", calib_rows)
    print(f"[ok] wrote outputs under {out_dir}")


if __name__ == "__main__":
    main()

