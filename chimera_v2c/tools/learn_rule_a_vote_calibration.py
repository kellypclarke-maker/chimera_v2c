#!/usr/bin/env python
"""
Learn and write a per-league Rule-A vote calibration JSON (read-only on ledgers).

This is intended for daily ops:
  - After outcomes are filled for day D, learn per-model vote-delta thresholds on all days <= D
    (or a specified historical window), then write a calibration JSON file that can be consumed by
    log_rule_a_votes_plan.py for day D+1 planning.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, load_games
from chimera_v2c.src.rule_a_unit_policy import iter_rule_a_games, load_bidask_csv
from chimera_v2c.src.rule_a_vote_calibration import VoteDeltaCalibration, choose_flip_delta_mode


DEFAULT_MODELS = ["v2c", "grok", "gemini", "gpt", "market_proxy", "moneypuck"]
DEFAULT_DELTAS = [0.0] + [i / 100.0 for i in range(1, 11)]  # 0.00..0.10


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
    ap = argparse.ArgumentParser(description="Learn per-model Rule-A vote deltas and write a calibration JSON.")
    ap.add_argument("--league", required=True, help="nba|nhl|nfl")
    ap.add_argument("--start-date", default="2025-11-19", help="YYYY-MM-DD inclusive (default: 2025-11-19).")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive (train through this date).")
    ap.add_argument("--anchor-minutes", type=int, default=30, help="T-minus anchor in minutes (default: 30).")
    ap.add_argument("--bidask-csv", default="", help="Path to kalshi_bidask_tminus_*.csv (default derived from dates).")
    ap.add_argument("--export-missing", action="store_true", help="Export bid/ask snapshot CSV if missing.")
    ap.add_argument("--slippage-cents", type=int, default=1, help="Extra slippage to add on top of away ask (default: 1).")

    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help=f"Models to consider (default: {' '.join(DEFAULT_MODELS)}).")
    ap.add_argument("--deltas", nargs="+", type=float, default=DEFAULT_DELTAS, help="Candidate vote delta thresholds (default 0.00..0.10 step 0.01).")
    ap.add_argument(
        "--flip-delta-mode",
        choices=["auto", "none", "same"],
        default="auto",
        help="Whether to gate flip/I signals by delta too (default: auto = pick on train).",
    )
    ap.add_argument("--min-signals", type=int, default=10, help="Min signal games per model (default: 10).")
    ap.add_argument("--max-iters", type=int, default=3, help="Max coordinate ascent passes (default: 3).")

    ap.add_argument("--roi-guardrail", choices=["strict", "soft"], default="soft", help="ROI guardrail mode (default: soft).")
    ap.add_argument("--roi-epsilon", type=float, default=0.03, help="Soft guardrail epsilon (absolute ROI units, default: 0.03).")

    ap.add_argument("--cap-units", type=int, default=10, help="Max contracts per game (default: 10).")
    ap.add_argument("--base-units", type=int, default=1, help="Base contracts per qualifying game (default: 1).")
    ap.add_argument("--vote-weight", type=int, default=1, help="Contracts to add for a non-flip vote (default: 1).")
    ap.add_argument("--flip-weight", type=int, default=2, help="Contracts to add for a flip/I vote (default: 2).")
    ap.add_argument("--vote-edge", type=float, default=0.0, help="Fee-aware edge gate (default: 0.0).")

    ap.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: chimera_v2c/data/rule_a_vote_calibration_<league>.json).",
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

    bidask = load_bidask_csv(bidask_path)
    ledger_games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        league_filter=league,
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

    flip_modes: List[str]
    if str(args.flip_delta_mode) in {"none", "same"}:
        flip_modes = [str(args.flip_delta_mode)]
    else:
        flip_modes = ["none", "same"]

    calib, totals_c, totals_b, counts = choose_flip_delta_mode(
        rule_a_games,
        league=league,
        trained_through=str(args.end_date),
        models=list(args.models),
        vote_deltas=[float(x) for x in args.deltas],
        flip_delta_modes=flip_modes,
        cap_units=int(args.cap_units),
        base_units=int(args.base_units),
        vote_weight=int(args.vote_weight),
        flip_weight=int(args.flip_weight),
        vote_delta_default=0.0,
        vote_edge_default=float(args.vote_edge),
        vote_edge_by_model=None,
        roi_guardrail_mode=str(args.roi_guardrail),
        roi_epsilon=float(args.roi_epsilon),
        min_signals=int(args.min_signals),
        max_iters=int(args.max_iters),
    )

    out_path = Path(args.out) if args.out else Path("chimera_v2c/data") / f"rule_a_vote_calibration_{league}.json"
    calib.save_json(out_path)
    print(f"[ok] wrote calibration JSON: {out_path}")
    print(f"[info] baseline: net={totals_b.net_pnl:.4f} roi={totals_b.roi_net:.4f}")
    print(f"[info] policy:   net={totals_c.net_pnl:.4f} roi={totals_c.roi_net:.4f} flip_delta_mode={calib.flip_delta_mode}")
    top = sorted(((m, counts.get(m, 0), calib.vote_delta_by_model.get(m, 0.0)) for m in calib.models), key=lambda x: (-x[1], x[0]))
    for m, n, d in top:
        print(f"[model] {m}: signals_n={int(n)} delta_vote={float(d):.3f}")


if __name__ == "__main__":
    main()
