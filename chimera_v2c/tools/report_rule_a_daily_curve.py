#!/usr/bin/env python3
"""
Report Rule A daily net PnL + ROI over a date window (read-only).

This is a convenience wrapper that produces a month-to-date (or arbitrary window)
"one day at a time" table from:
  - Daily ledgers (model/proxy probabilities + outcomes)
  - Kalshi bid/ask T-minus snapshots (execution realism)

Assumptions (defaults mirror the operator research defaults):
  - Anchor: T-30
  - Execution: taker BUY away-YES at away ask + 1¢ slippage
  - Fees: Kalshi taker fees
  - Sizing: Rule A "blind + calibrated weighted votes" via VoteDeltaCalibration when available
  - Weak buckets: apply per-league weak bucket cap JSON when available

No ledgers are modified; outputs are written under reports/thesis_summaries/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, load_games
from chimera_v2c.src.rule_a_unit_policy import (
    RuleAGame,
    load_bidask_csv,
    iter_rule_a_games,
    mid_bucket,
    net_pnl_taker,
)
from chimera_v2c.src.rule_a_vote_calibration import VoteDeltaCalibration, contracts_for_game
from chimera_v2c.src.rule_a_vote_calibration import contracts_for_game_with_eligibility
from chimera_v2c.src.rule_a_model_eligibility import ModelEligibility


DEFAULT_MODELS: List[str] = ["v2c", "grok", "gemini", "gpt", "market_proxy", "moneypuck"]


@dataclass(frozen=True)
class WeakBucketPolicy:
    weak_buckets: List[str]
    weak_bucket_cap: int


def _default_calibration_path(league: str) -> Path:
    return Path("chimera_v2c/data") / f"rule_a_vote_calibration_{league}.json"


def _default_weak_buckets_path(league: str) -> Path:
    return Path("chimera_v2c/data") / f"rule_a_weak_buckets_{league}.json"


def _default_eligibility_path(league: str) -> Path:
    return Path("chimera_v2c/data") / f"rule_a_model_eligibility_{league}.json"


def _default_bidask_path(*, start_date: str, end_date: str, anchor_minutes: int) -> Path:
    return Path("reports/market_snapshots") / (
        f"kalshi_bidask_tminus_{start_date.replace('-', '')}_{end_date.replace('-', '')}_m{int(anchor_minutes)}.csv"
    )


def _load_weak_bucket_policy(path: Path) -> Optional[WeakBucketPolicy]:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        return None
    weak = [str(x) for x in (d.get("weak_buckets") or []) if str(x).strip()]
    cap = int(d.get("weak_bucket_cap") or 3)
    if cap < 1:
        cap = 1
    return WeakBucketPolicy(weak_buckets=weak, weak_bucket_cap=cap)


def _ensure_bidask_snapshot(
    *,
    start_date: str,
    end_date: str,
    anchor_minutes: int,
    out_path: Path,
    export_missing: bool,
) -> None:
    if out_path.exists():
        return
    if not export_missing:
        raise SystemExit(f"[error] missing bid/ask snapshot CSV: {out_path} (pass --export-missing)")

    # Import here to keep this tool lightweight unless exporting is requested.
    import subprocess
    import os

    cmd = [
        "python3",
        "chimera_v2c/tools/export_kalshi_bidask_tminus.py",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--minutes-before-start",
        str(int(anchor_minutes)),
        "--out",
        str(out_path),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
    subprocess.run(cmd, check=True, env=env)


def _compute_daily_rows(
    games: Iterable[RuleAGame],
    *,
    league: str,
    models: Sequence[str],
    vote_cfg: Optional[VoteDeltaCalibration],
    weak_policy: Optional[WeakBucketPolicy],
    default_vote_delta: float,
    default_vote_edge: float,
    default_vote_weight: int,
    default_flip_weight: int,
    default_flip_delta_mode: str,
    default_base_units: int,
    default_cap_units: int,
    eligibility: Optional[ModelEligibility],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    # Choose policy parameters.
    if vote_cfg is not None:
        policy_models = list(vote_cfg.models)
        vote_delta_default = float(vote_cfg.vote_delta_default)
        vote_delta_by_model = dict(vote_cfg.vote_delta_by_model)
        vote_edge_default = float(vote_cfg.vote_edge_default)
        vote_edge_by_model = dict(vote_cfg.vote_edge_by_model)
        flip_delta_mode = str(vote_cfg.flip_delta_mode)
        vote_weight = int(vote_cfg.vote_weight)
        flip_weight = int(vote_cfg.flip_weight)
        base_units = int(vote_cfg.base_units)
        cap_units = int(vote_cfg.cap_units)
    else:
        policy_models = list(models)
        vote_delta_default = float(default_vote_delta)
        vote_delta_by_model = {str(m): float(default_vote_delta) for m in policy_models}
        vote_edge_default = float(default_vote_edge)
        vote_edge_by_model = {}
        flip_delta_mode = str(default_flip_delta_mode)
        vote_weight = int(default_vote_weight)
        flip_weight = int(default_flip_weight)
        base_units = int(default_base_units)
        cap_units = int(default_cap_units)

    # Aggregate by date.
    by_date: Dict[str, Dict[str, float]] = {}
    by_date_counts: Dict[str, Dict[str, int]] = {}
    by_date_bets: Dict[str, int] = {}
    for g in games:
        if g.league != league:
            continue
        if g.home_win is None:
            continue
        key = str(g.date)
        if eligibility is not None:
            contracts, _ = contracts_for_game_with_eligibility(
                g,
                models=policy_models,
                vote_delta_default=vote_delta_default,
                vote_delta_by_model=vote_delta_by_model,
                vote_edge_default=vote_edge_default,
                vote_edge_by_model=vote_edge_by_model,
                flip_delta_mode=flip_delta_mode,
                vote_weight=vote_weight,
                flip_weight=flip_weight,
                cap_units=cap_units,
                base_units=base_units,
                eligibility=eligibility,
            )
        else:
            contracts, _ = contracts_for_game(
                g,
                models=policy_models,
                vote_delta_default=vote_delta_default,
                vote_delta_by_model=vote_delta_by_model,
                vote_edge_default=vote_edge_default,
                vote_edge_by_model=vote_edge_by_model,
                flip_delta_mode=flip_delta_mode,
                vote_weight=vote_weight,
                flip_weight=flip_weight,
                cap_units=cap_units,
                base_units=base_units,
            )

        if weak_policy is not None and mid_bucket(float(g.mid_home)) in set(weak_policy.weak_buckets):
            contracts = min(int(contracts), int(weak_policy.weak_bucket_cap))

        gross, fees, risked = net_pnl_taker(contracts=int(contracts), price_away=float(g.price_away), home_win=int(g.home_win))
        net = float(gross) - float(fees)

        by_date.setdefault(key, {"net": 0.0, "risk": 0.0, "fees": 0.0})
        by_date[key]["net"] += float(net)
        by_date[key]["risk"] += float(risked)
        by_date[key]["fees"] += float(fees)

        by_date_counts.setdefault(key, {"contracts": 0})
        by_date_counts[key]["contracts"] += int(contracts)

        by_date_bets[key] = int(by_date_bets.get(key, 0) + 1)

    for date in sorted(by_date.keys()):
        net = float(by_date[date]["net"])
        risk = float(by_date[date]["risk"])
        roi = 0.0 if risk <= 0 else net / risk
        rows.append(
            {
                "date": date,
                "league": league,
                "bets": int(by_date_bets.get(date, 0)),
                "contracts": int(by_date_counts.get(date, {}).get("contracts", 0)),
                "risked": round(risk, 6),
                "fees": round(float(by_date[date]["fees"]), 6),
                "net_pnl": round(net, 6),
                "roi_net": round(float(roi), 6),
            }
        )

    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report Rule A daily net PnL + ROI over a date window (read-only).")
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive.")
    p.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive.")
    p.add_argument("--league", default="all", choices=["all", "nba", "nhl", "nfl"], help="League to report (default: all).")
    p.add_argument("--anchor-minutes", type=int, default=30, help="T-minus anchor minutes (default: 30).")
    p.add_argument("--slippage-cents", type=int, default=1, help="Execution slippage on away ask (default: 1).")
    p.add_argument("--bidask-csv", default="", help="Optional bid/ask snapshot CSV path; default derived from dates.")
    p.add_argument("--export-missing", action="store_true", help="Export bid/ask snapshot CSV if missing.")
    p.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model columns to use for votes when no per-league calibration JSON exists.",
    )
    p.add_argument("--vote-calibration-json", default="", help="Optional VoteDeltaCalibration JSON path (single-league runs only).")
    p.add_argument("--eligibility-json", default="", help="Optional ModelEligibility JSON path (single-league runs only).")
    p.add_argument(
        "--no-eligibility",
        action="store_true",
        help="Disable auto-loading per-league model eligibility JSON (chimera_v2c/data/rule_a_model_eligibility_<league>.json).",
    )
    p.add_argument("--weak-buckets-json", default="", help="Optional weak buckets JSON path (single-league runs only).")
    p.add_argument("--out", default="", help="Optional output CSV path (default under reports/thesis_summaries/).")

    # Defaults used only when no VoteDeltaCalibration JSON exists.
    p.add_argument("--vote-delta-default", type=float, default=0.01, help="Default vote delta when no calibration JSON (default: 0.01).")
    p.add_argument("--vote-edge-default", type=float, default=0.0, help="Default fee-aware edge gate when no calibration JSON (default: 0.0).")
    p.add_argument("--vote-weight", type=int, default=1, help="Contracts per normal vote when no calibration JSON (default: 1).")
    p.add_argument("--flip-weight", type=int, default=2, help="Contracts per flip/I when no calibration JSON (default: 2).")
    p.add_argument(
        "--flip-delta-mode",
        default="none",
        choices=["none", "same"],
        help="Whether flip requires delta gate when no calibration JSON (default: none).",
    )
    p.add_argument("--base-units", type=int, default=1, help="Base contracts per qualifying game when no calibration JSON (default: 1).")
    p.add_argument("--cap-units", type=int, default=10, help="Max contracts per game when no calibration JSON (default: 10).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    start = str(args.start_date).strip()
    end = str(args.end_date).strip()
    anchor = int(args.anchor_minutes)
    slip = int(args.slippage_cents)

    bidask_path = Path(str(args.bidask_csv).strip()) if str(args.bidask_csv).strip() else _default_bidask_path(
        start_date=start, end_date=end, anchor_minutes=anchor
    )
    _ensure_bidask_snapshot(
        start_date=start,
        end_date=end,
        anchor_minutes=anchor,
        out_path=bidask_path,
        export_missing=bool(args.export_missing),
    )

    bidask = load_bidask_csv(bidask_path)

    league_arg = str(args.league).strip().lower()
    leagues = ["nba", "nhl", "nfl"] if league_arg == "all" else [league_arg]

    # Load ledger games once.
    ledger_games = load_games(daily_dir=LEDGER_DIR, start_date=start, end_date=end, models=list(args.models))
    if not ledger_games:
        raise SystemExit("[error] no games loaded from daily ledgers for the given window")

    # Convert to Rule-A games (home-favored only; requires outcomes).
    ra_games = iter_rule_a_games(
        ledger_games,
        bidask=bidask,
        models=list(args.models),
        slippage_cents=slip,
        require_outcome=True,
    )

    all_rows: List[Dict[str, object]] = []
    for league in leagues:
        vote_cfg: Optional[VoteDeltaCalibration] = None
        weak_policy: Optional[WeakBucketPolicy] = None
        elig: Optional[ModelEligibility] = None

        if league_arg != "all" and str(args.vote_calibration_json).strip():
            vote_cfg = VoteDeltaCalibration.load_json(Path(str(args.vote_calibration_json).strip()))
        else:
            p = _default_calibration_path(league)
            if p.exists():
                vote_cfg = VoteDeltaCalibration.load_json(p)

        if league_arg != "all" and str(args.eligibility_json).strip():
            elig = ModelEligibility.load_json(Path(str(args.eligibility_json).strip()))
        elif not bool(args.no_eligibility):
            ep = _default_eligibility_path(league)
            if ep.exists():
                elig = ModelEligibility.load_json(ep)

        if league_arg != "all" and str(args.weak_buckets_json).strip():
            weak_policy = _load_weak_bucket_policy(Path(str(args.weak_buckets_json).strip()))
        else:
            wp = _default_weak_buckets_path(league)
            weak_policy = _load_weak_bucket_policy(wp)

        league_rows = _compute_daily_rows(
            ra_games,
            league=league,
            models=list(args.models),
            vote_cfg=vote_cfg,
            weak_policy=weak_policy,
            default_vote_delta=float(args.vote_delta_default),
            default_vote_edge=float(args.vote_edge_default),
            default_vote_weight=int(args.vote_weight),
            default_flip_weight=int(args.flip_weight),
            default_flip_delta_mode=str(args.flip_delta_mode),
            default_base_units=int(args.base_units),
            default_cap_units=int(args.cap_units),
            eligibility=elig,
        )
        all_rows.extend(league_rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("[error] no qualifying Rule A games found for the given window/league")

    overall = df.groupby("date", as_index=False).agg(
        bets=("bets", "sum"),
        contracts=("contracts", "sum"),
        risked=("risked", "sum"),
        fees=("fees", "sum"),
        net_pnl=("net_pnl", "sum"),
    )
    overall["roi_net"] = overall.apply(lambda r: 0.0 if float(r["risked"]) <= 0 else float(r["net_pnl"]) / float(r["risked"]), axis=1)
    overall.insert(1, "league", "overall")

    combined = pd.concat([overall, df], ignore_index=True).sort_values(["date", "league"]).reset_index(drop=True)

    out_path = Path(str(args.out).strip()) if str(args.out).strip() else Path("reports/thesis_summaries") / (
        f"rule_a_daily_curve_{start.replace('-', '')}_{end.replace('-', '')}_m{anchor}_s{slip}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    # Print overall one-day-at-a-time.
    print(f"[ok] wrote {out_path}")
    for _, r in overall.sort_values("date").iterrows():
        date = str(r["date"])
        net = float(r["net_pnl"])
        risk = float(r["risked"])
        roi = 0.0 if risk <= 0 else net / risk
        print(f"{date}  net={net:+.2f}  risk={risk:.2f}  roi={100*roi:+.2f}%  bets={int(r['bets'])}  contracts={int(r['contracts'])}")


if __name__ == "__main__":
    main()
