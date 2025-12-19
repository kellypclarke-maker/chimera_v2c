#!/usr/bin/env python
"""
Walk-forward (train->test) calibrator for Rule-A vote deltas (taker-only; fee-aware; uses bid/ask snapshots).

Goal:
  Learn per-(league,model) "sweet spot" thresholds for adding extra units when Kalshi is home-fav,
  using only prior days (no leakage), optimizing net PnL with an ROI guardrail vs a baseline.

Baseline (for ROI guardrail + reporting):
  - Qualifying game: home YES mid > 0.50 at anchor
  - Trade: buy AWAY YES at the AWAY YES ask (+ optional slippage)
  - Size: 1 + weighted votes, where a flip/I signal counts as 2 units (vote=1, flip=2)
  - Vote delta: 0.00 (no delta gate); flip delta: none (no delta gate)
  - Edge gate: fee-aware edge_net >= 0.0

Candidate policy:
  - Same as baseline, but learn per-model vote delta thresholds (and optionally whether to gate flips by delta)
    on train-only data for each test day.

Outputs (read-only on ledgers):
  - Summary + daily + model-calibration + trades CSVs under reports/thesis_summaries/.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.rule_a_unit_policy import iter_rule_a_games, load_bidask_csv
from chimera_v2c.src.rule_a_vote_calibration import (
    VoteDeltaCalibration,
    choose_flip_delta_mode,
    contracts_for_game,
    totals_for_policy,
)


DEFAULT_MODELS = ["v2c", "grok", "gemini", "gpt", "market_proxy", "moneypuck"]
DEFAULT_DELTAS = [0.0] + [i / 100.0 for i in range(1, 11)]  # 0.00..0.10


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


def _train_dates(dates: Sequence[str], *, test_date: str) -> List[str]:
    return [d for d in dates if d < test_date]


def _sum_totals(a, b):
    from chimera_v2c.src.rule_a_unit_policy import Totals

    out = Totals()
    out.bets = int(a.bets + b.bets)
    out.contracts = int(a.contracts + b.contracts)
    out.risked = float(a.risked + b.risked)
    out.gross_pnl = float(a.gross_pnl + b.gross_pnl)
    out.fees = float(a.fees + b.fees)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Walk-forward calibration of per-model Rule-A vote deltas (taker-only).")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive (analysis window).")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive (analysis window).")
    ap.add_argument("--league", default="all", help="nba|nhl|nfl|all (default: all).")
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
    ap.add_argument("--min-train-days", type=int, default=5, help="Min distinct train days required (default: 5).")
    ap.add_argument("--min-signals", type=int, default=10, help="Min signal games per model (default: 10).")
    ap.add_argument("--max-iters", type=int, default=3, help="Max coordinate ascent passes (default: 3).")

    ap.add_argument(
        "--roi-guardrail",
        choices=["strict", "soft", "both"],
        default="both",
        help="ROI guardrail mode(s) to evaluate (default: both).",
    )
    ap.add_argument("--roi-epsilon", type=float, default=0.03, help="Soft guardrail epsilon (absolute ROI units, default: 0.03).")

    ap.add_argument("--cap-units", type=int, default=10, help="Max contracts per game (default: 10).")
    ap.add_argument("--base-units", type=int, default=1, help="Base contracts per qualifying game (default: 1).")
    ap.add_argument("--vote-weight", type=int, default=1, help="Contracts to add for a non-flip vote (default: 1).")
    ap.add_argument("--flip-weight", type=int, default=2, help="Contracts to add for a flip/I vote (default: 2).")
    ap.add_argument("--vote-edge", type=float, default=0.0, help="Fee-aware edge gate (default: 0.0).")

    ap.add_argument("--out-dir", default="reports/thesis_summaries", help="Output directory (default: reports/thesis_summaries).")
    ap.add_argument("--no-write", action="store_true", help="Print overall summary only; do not write files.")
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
    by_date_league: Dict[Tuple[str, str], List] = defaultdict(list)
    for g in rule_a_games:
        by_date_league[(g.date, g.league)].append(g)

    dates_by_league: Dict[str, List[str]] = {lg: sorted({g.date for g in rule_a_games if g.league == lg}) for lg in leagues}

    roi_modes: List[str]
    if str(args.roi_guardrail) == "both":
        roi_modes = ["strict", "soft"]
    else:
        roi_modes = [str(args.roi_guardrail)]

    out_dir = Path(args.out_dir)
    league_tag = str(league_filter or "all")
    tag = f"{league_tag}_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}_m{int(args.anchor_minutes)}_s{int(args.slippage_cents)}"

    all_summary: List[Dict[str, object]] = []
    all_daily: List[Dict[str, object]] = []
    all_calib: List[Dict[str, object]] = []
    all_trades: List[Dict[str, object]] = []

    for roi_mode in roi_modes:
        totals_policy: Dict[str, object] = {}
        totals_base: Dict[str, object] = {}
        for lg in leagues + ["overall"]:
            from chimera_v2c.src.rule_a_unit_policy import Totals

            totals_policy[(roi_mode, lg)] = Totals()
            totals_base[(roi_mode, lg)] = Totals()

        for test_date in dates:
            for league in leagues:
                test = by_date_league.get((test_date, league), [])
                if not test:
                    continue
                train_dates = _train_dates(dates_by_league.get(league, []), test_date=test_date)
                train = [g for d in train_dates for g in by_date_league.get((d, league), [])]

                trained = len(set(train_dates)) >= int(args.min_train_days)
                if not trained:
                    calib = VoteDeltaCalibration(
                        league=league,
                        trained_through=max(train_dates) if train_dates else "",
                        models=[str(m) for m in args.models],
                        vote_delta_by_model={str(m): 0.0 for m in args.models},
                        vote_delta_default=0.0,
                        vote_edge_by_model={},
                        vote_edge_default=float(args.vote_edge),
                        flip_delta_mode="none",
                        vote_weight=int(args.vote_weight),
                        flip_weight=int(args.flip_weight),
                        cap_units=int(args.cap_units),
                        base_units=int(args.base_units),
                        roi_guardrail_mode=str(roi_mode),
                        roi_epsilon=float(args.roi_epsilon),
                    )
                    all_daily.append(
                        {
                            "roi_mode": roi_mode,
                            "date": test_date,
                            "league": league,
                            "trained": 0,
                            "train_days": int(len(set(train_dates))),
                            "train_through": calib.trained_through,
                            "flip_delta_mode": calib.flip_delta_mode,
                            "train_baseline_net": "",
                            "train_baseline_roi": "",
                            "train_policy_net": "",
                            "train_policy_roi": "",
                        }
                    )
                else:
                    flip_modes = [str(args.flip_delta_mode)] if str(args.flip_delta_mode) in {"none", "same"} else ["none", "same"]
                    calib, train_totals, base_train_totals, signal_counts = choose_flip_delta_mode(
                        train,
                        league=league,
                        trained_through=max(train_dates) if train_dates else "",
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
                        roi_guardrail_mode=str(roi_mode),
                        roi_epsilon=float(args.roi_epsilon),
                        min_signals=int(args.min_signals),
                        max_iters=int(args.max_iters),
                    )

                    for m in sorted(signal_counts):
                        all_calib.append(
                            {
                                "roi_mode": roi_mode,
                                "date": test_date,
                                "league": league,
                                "model": m,
                                "train_through": calib.trained_through,
                                "flip_delta_mode": calib.flip_delta_mode,
                                "delta_vote": float(calib.vote_delta_by_model.get(m, 0.0)),
                                "signals_n": int(signal_counts.get(m, 0)),
                            }
                        )

                    all_daily.append(
                        {
                            "roi_mode": roi_mode,
                            "date": test_date,
                            "league": league,
                            "trained": int(trained),
                            "train_days": int(len(set(train_dates))),
                            "train_through": calib.trained_through,
                            "flip_delta_mode": calib.flip_delta_mode,
                            "train_baseline_net": round(float(base_train_totals.net_pnl), 6),
                            "train_baseline_roi": round(float(base_train_totals.roi_net), 6),
                            "train_policy_net": round(float(train_totals.net_pnl), 6),
                            "train_policy_roi": round(float(train_totals.roi_net), 6),
                        }
                    )

                # Evaluate baseline vs calibrated on test day.
                baseline_test = totals_for_policy(
                    test,
                    models=list(args.models),
                    vote_delta_default=0.0,
                    vote_delta_by_model={str(m): 0.0 for m in args.models},
                    vote_edge_default=float(args.vote_edge),
                    vote_edge_by_model={},
                    flip_delta_mode="none",
                    vote_weight=int(args.vote_weight),
                    flip_weight=int(args.flip_weight),
                    cap_units=int(args.cap_units),
                    base_units=int(args.base_units),
                )
                policy_test = totals_for_policy(
                    test,
                    models=list(args.models),
                    vote_delta_default=float(calib.vote_delta_default),
                    vote_delta_by_model=dict(calib.vote_delta_by_model),
                    vote_edge_default=float(calib.vote_edge_default),
                    vote_edge_by_model=dict(calib.vote_edge_by_model),
                    flip_delta_mode=str(calib.flip_delta_mode),
                    vote_weight=int(calib.vote_weight),
                    flip_weight=int(calib.flip_weight),
                    cap_units=int(calib.cap_units),
                    base_units=int(calib.base_units),
                )

                for lg in (league, "overall"):
                    totals_base[(roi_mode, lg)] = _sum_totals(totals_base[(roi_mode, lg)], baseline_test)
                    totals_policy[(roi_mode, lg)] = _sum_totals(totals_policy[(roi_mode, lg)], policy_test)

                # Trades audit rows
                for g in test:
                    from chimera_v2c.src.rule_a_unit_policy import net_pnl_taker

                    contracts_base, triggers_base = contracts_for_game(
                        g,
                        models=list(args.models),
                        vote_delta_default=0.0,
                        vote_delta_by_model={str(m): 0.0 for m in args.models},
                        vote_edge_default=float(args.vote_edge),
                        vote_edge_by_model={},
                        flip_delta_mode="none",
                        vote_weight=int(args.vote_weight),
                        flip_weight=int(args.flip_weight),
                        cap_units=int(args.cap_units),
                        base_units=int(args.base_units),
                    )
                    contracts_pol, triggers_pol = contracts_for_game(
                        g,
                        models=list(args.models),
                        vote_delta_default=float(calib.vote_delta_default),
                        vote_delta_by_model=dict(calib.vote_delta_by_model),
                        vote_edge_default=float(calib.vote_edge_default),
                        vote_edge_by_model=dict(calib.vote_edge_by_model),
                        flip_delta_mode=str(calib.flip_delta_mode),
                        vote_weight=int(calib.vote_weight),
                        flip_weight=int(calib.flip_weight),
                        cap_units=int(calib.cap_units),
                        base_units=int(calib.base_units),
                    )

                    gross_b, fees_b, risked_b = net_pnl_taker(contracts=int(contracts_base), price_away=float(g.price_away), home_win=int(g.home_win))
                    gross_p, fees_p, risked_p = net_pnl_taker(contracts=int(contracts_pol), price_away=float(g.price_away), home_win=int(g.home_win))
                    all_trades.append(
                        {
                            "roi_mode": roi_mode,
                            "date": g.date,
                            "league": g.league,
                            "matchup": g.matchup,
                            "mid_home": round(float(g.mid_home), 6),
                            "price_away": round(float(g.price_away), 6),
                            "home_win": int(g.home_win or 0),
                            "baseline_contracts": int(contracts_base),
                            "baseline_triggers": ",".join(triggers_base),
                            "baseline_gross_pnl": round(float(gross_b), 6),
                            "baseline_fees": round(float(fees_b), 6),
                            "baseline_net_pnl": round(float(gross_b - fees_b), 6),
                            "baseline_risked": round(float(risked_b), 6),
                            "policy_contracts": int(contracts_pol),
                            "policy_triggers": ",".join(triggers_pol),
                            "policy_gross_pnl": round(float(gross_p), 6),
                            "policy_fees": round(float(fees_p), 6),
                            "policy_net_pnl": round(float(gross_p - fees_p), 6),
                            "policy_risked": round(float(risked_p), 6),
                        }
                    )

        for league in leagues + ["overall"]:
            b = totals_base[(roi_mode, league)]
            p = totals_policy[(roi_mode, league)]
            all_summary.append(
                {
                    "roi_mode": roi_mode,
                    "window_start": args.start_date,
                    "window_end": args.end_date,
                    "league": league,
                    "anchor_minutes": int(args.anchor_minutes),
                    "slippage_cents": int(args.slippage_cents),
                    "cap_units": int(args.cap_units),
                    "base_units": int(args.base_units),
                    "vote_weight": int(args.vote_weight),
                    "flip_weight": int(args.flip_weight),
                    "vote_edge_default": float(args.vote_edge),
                    "models": ",".join(list(args.models)),
                    "baseline_bets": int(b.bets),
                    "baseline_contracts": int(b.contracts),
                    "baseline_risked": round(float(b.risked), 6),
                    "baseline_fees": round(float(b.fees), 6),
                    "baseline_net_pnl": round(float(b.net_pnl), 6),
                    "baseline_roi_net": round(float(b.roi_net), 6),
                    "policy_bets": int(p.bets),
                    "policy_contracts": int(p.contracts),
                    "policy_risked": round(float(p.risked), 6),
                    "policy_fees": round(float(p.fees), 6),
                    "policy_net_pnl": round(float(p.net_pnl), 6),
                    "policy_roi_net": round(float(p.roi_net), 6),
                }
            )

    if args.no_write:
        for r in all_summary:
            if r["league"] == "overall":
                print(r)
        return

    _write_csv(out_dir / f"walkforward_rule_a_vote_calibration_summary_{tag}.csv", all_summary)
    _write_csv(out_dir / f"walkforward_rule_a_vote_calibration_daily_{tag}.csv", all_daily)
    _write_csv(out_dir / f"walkforward_rule_a_vote_calibration_modelcal_{tag}.csv", all_calib)
    _write_csv(out_dir / f"walkforward_rule_a_vote_calibration_trades_{tag}.csv", all_trades)
    print(f"[ok] wrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
