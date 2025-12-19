#!/usr/bin/env python3
"""
Reconstruct a Rule A "v2c hybrid" trade sheet from historical T-minus bid/ask snapshots (read-only).

This is the historical analog of log_rule_a_votes_plan.py (which uses live quotes), but uses:
  - daily ledgers for model probabilities + outcomes
  - a fixed T-minus bid/ask snapshot CSV for execution realism
  - per-league vote calibration JSONs (base_units, vote_delta_by_model, weights, etc.)
  - per-league eligibility JSONs (primary/secondary gating) if present
  - per-league weak buckets JSONs (cap sizing in weak buckets) if present

Outputs:
  - A per-game trade sheet CSV with contracts + realized PnL (since outcomes are known).
  - A per-model per-game breakdown CSV explaining why each model did/didn't contribute.

No ledgers are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.rule_a_model_eligibility import ModelEligibility
from chimera_v2c.src.rule_a_unit_policy import (
    RuleAGame,
    expected_edge_net_per_contract,
    iter_rule_a_games,
    load_bidask_csv,
    mid_bucket,
    net_pnl_taker,
)
from chimera_v2c.src.rule_a_vote_calibration import VoteDeltaCalibration, contracts_for_game, contracts_for_game_with_eligibility, model_contribution


LEAGUES = ["nba", "nhl", "nfl"]


@dataclass(frozen=True)
class WeakBucketPolicy:
    weak_buckets: List[str]
    weak_bucket_cap: int


def _default_calibration_path(league: str) -> Path:
    return Path("chimera_v2c/data") / f"rule_a_vote_calibration_{league}.json"


def _default_eligibility_path(league: str) -> Path:
    return Path("chimera_v2c/data") / f"rule_a_model_eligibility_{league}.json"


def _default_weak_buckets_path(league: str) -> Path:
    return Path("chimera_v2c/data") / f"rule_a_weak_buckets_{league}.json"


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconstruct Rule A v2c-hybrid trades from T-minus bid/ask snapshots (read-only).")
    p.add_argument("--date", required=True, help="YYYY-MM-DD (Pacific semantics; matches daily ledger filename date).")
    p.add_argument("--anchor-minutes", type=int, default=30, help="T-minus anchor minutes (default: 30).")
    p.add_argument("--slippage-cents", type=int, default=1, help="Extra slippage added to away ask (default: 1).")
    p.add_argument("--bidask-csv", required=True, help="Path to kalshi_bidask_tminus_*.csv for this date (or a superset).")
    p.add_argument("--no-eligibility", action="store_true", help="Disable per-league eligibility gating (primary/secondary).")
    p.add_argument("--out-dir", default="reports/thesis_summaries", help="Output directory (default: reports/thesis_summaries).")
    return p.parse_args()


def _status_for_model(model: str, elig: Optional[ModelEligibility]) -> str:
    if elig is None:
        return "ungated"
    if model in set(str(m) for m in elig.primary_models):
        return "primary"
    if model in set(str(m) for m in elig.secondary_models):
        return "secondary"
    return "excluded"


def _effective_models_for_league(calib: VoteDeltaCalibration) -> List[str]:
    return [str(m) for m in calib.models]


def _read_rows(path: Path, *, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(list(rows))


def main() -> None:
    args = _parse_args()
    date_iso = str(args.date).strip()
    bidask_csv = Path(str(args.bidask_csv).strip())
    if not bidask_csv.exists():
        raise SystemExit(f"[error] missing --bidask-csv: {bidask_csv}")

    # Load per-league configs.
    calib_by_league: Dict[str, VoteDeltaCalibration] = {}
    elig_by_league: Dict[str, Optional[ModelEligibility]] = {}
    weak_by_league: Dict[str, Optional[WeakBucketPolicy]] = {}

    all_models: List[str] = []
    for league in LEAGUES:
        cp = _default_calibration_path(league)
        if not cp.exists():
            raise SystemExit(f"[error] missing vote calibration JSON: {cp}")
        calib = VoteDeltaCalibration.load_json(cp)
        calib_by_league[league] = calib
        all_models.extend(_effective_models_for_league(calib))

        ep = _default_eligibility_path(league)
        elig_by_league[league] = None if bool(args.no_eligibility) else (ModelEligibility.load_json(ep) if ep.exists() else None)

        wp = _default_weak_buckets_path(league)
        weak_by_league[league] = _load_weak_bucket_policy(wp)

    all_models = sorted(set(all_models))

    # Load the daily ledger games for the date across leagues, including all needed model columns.
    games: List[GameRow] = load_games(
        daily_dir=LEDGER_DIR,
        start_date=date_iso,
        end_date=date_iso,
        league_filter=None,
        models=list(all_models),
    )
    if not games:
        raise SystemExit(f"[error] no games loaded for date {date_iso}; expected ledger under {LEDGER_DIR}")

    bidask = load_bidask_csv(bidask_csv)

    # Convert to Rule-A games using the bid/ask snapshot and outcomes.
    ra: List[RuleAGame] = iter_rule_a_games(
        games,
        bidask=bidask,
        models=list(all_models),
        slippage_cents=int(args.slippage_cents),
        require_outcome=True,
    )
    if not ra:
        raise SystemExit("[error] no qualifying Rule-A games found for date+snapshot (home-fav at anchor)")

    # Per-game trade sheet + per-model breakdown.
    trade_rows: List[Dict[str, object]] = []
    model_rows: List[Dict[str, object]] = []

    totals_by_league: Dict[str, Dict[str, float]] = {lg: {"net": 0.0, "risk": 0.0, "fees": 0.0, "contracts": 0.0} for lg in LEAGUES}
    totals_by_league["overall"] = {"net": 0.0, "risk": 0.0, "fees": 0.0, "contracts": 0.0}

    for g in ra:
        league = str(g.league).lower().strip()
        if league not in calib_by_league:
            continue
        calib = calib_by_league[league]
        elig = elig_by_league.get(league)
        weak = weak_by_league.get(league)

        # Compute contracts + triggers (eligibility-gated if available).
        if elig is not None:
            contracts, triggers = contracts_for_game_with_eligibility(
                g,
                models=_effective_models_for_league(calib),
                vote_delta_default=float(calib.vote_delta_default),
                vote_delta_by_model=dict(calib.vote_delta_by_model),
                vote_edge_default=float(calib.vote_edge_default),
                vote_edge_by_model=dict(calib.vote_edge_by_model),
                flip_delta_mode=str(calib.flip_delta_mode),
                vote_weight=int(calib.vote_weight),
                flip_weight=int(calib.flip_weight),
                cap_units=int(calib.cap_units),
                base_units=int(calib.base_units),
                eligibility=elig,
            )
        else:
            contracts, triggers = contracts_for_game(
                g,
                models=_effective_models_for_league(calib),
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

        bucket = mid_bucket(float(g.mid_home))
        weak_bucket = False
        if weak is not None and bucket in set(weak.weak_buckets):
            weak_bucket = True
            contracts = min(int(contracts), int(weak.weak_bucket_cap))

        gross, fees, risked = net_pnl_taker(contracts=int(contracts), price_away=float(g.price_away), home_win=int(g.home_win or 0))
        net = float(gross) - float(fees)

        for lg in (league, "overall"):
            totals_by_league[lg]["net"] += net
            totals_by_league[lg]["risk"] += float(risked)
            totals_by_league[lg]["fees"] += float(fees)
            totals_by_league[lg]["contracts"] += float(int(contracts))

        trade_rows.append(
            {
                "date": g.date,
                "league": league,
                "matchup": g.matchup,
                "mid_home": round(float(g.mid_home), 6),
                "price_away": round(float(g.price_away), 6),
                "mid_bucket": bucket,
                "weak_bucket": int(weak_bucket),
                "contracts": int(contracts),
                "triggers": "|".join(triggers),
                "home_win": int(g.home_win or 0),
                "gross_pnl": round(float(gross), 6),
                "fees": round(float(fees), 6),
                "net_pnl": round(net, 6),
            }
        )

        # Per-model explanation rows.
        for m in _effective_models_for_league(calib):
            p = g.probs.get(str(m))
            status = _status_for_model(str(m), elig)
            delta_thr = float(calib.vote_delta_by_model.get(str(m), calib.vote_delta_default))
            edge_thr = float(calib.vote_edge_by_model.get(str(m), calib.vote_edge_default))
            if p is None:
                model_rows.append(
                    {
                        "date": g.date,
                        "league": league,
                        "matchup": g.matchup,
                        "model": str(m),
                        "status": status,
                        "p_home": "",
                        "mid_minus_p": "",
                        "edge_net": "",
                        "delta_thr": delta_thr,
                        "edge_thr": edge_thr,
                        "delta_ok": 0,
                        "edge_ok": 0,
                        "flip": 0,
                        "vote": 0,
                        "counted": 0,
                        "reason": "no p_home value in ledger",
                    }
                )
                continue

            p_home = float(p)
            mid_minus_p = float(g.mid_home) - p_home
            edge_net = expected_edge_net_per_contract(p_home=p_home, price_away=float(g.price_away))
            delta_ok = mid_minus_p >= delta_thr
            edge_ok = edge_net >= edge_thr
            is_flip = (p_home < 0.5) and edge_ok and (str(calib.flip_delta_mode) != "same" or delta_ok)
            is_vote = delta_ok and edge_ok

            add, typ = model_contribution(
                g,
                model=str(m),
                vote_delta_default=float(calib.vote_delta_default),
                vote_delta_by_model=dict(calib.vote_delta_by_model),
                vote_edge_default=float(calib.vote_edge_default),
                vote_edge_by_model=dict(calib.vote_edge_by_model),
                flip_delta_mode=str(calib.flip_delta_mode),
                vote_weight=int(calib.vote_weight),
                flip_weight=int(calib.flip_weight),
            )

            # Counted means it actually contributed after eligibility gating.
            counted = 0
            if int(add) > 0 and typ:
                if status == "ungated":
                    counted = 1
                elif status == "primary":
                    counted = 1
                elif status == "secondary":
                    # Only counted if any primary trigger is present in the final triggers list.
                    counted = 1 if any(t.endswith(":primary") for t in triggers) else 0
                else:
                    counted = 0

            reason = ""
            if status == "excluded":
                reason = "excluded by eligibility"
            elif not edge_ok:
                reason = "edge_net below threshold (fees+slippage aware)"
            elif is_flip:
                reason = "flip (p_home<0.50) + edge_ok"
            elif is_vote:
                reason = "vote (mid_home - p_home >= delta) + edge_ok"
            else:
                reason = "no vote/flip gate met"

            model_rows.append(
                {
                    "date": g.date,
                    "league": league,
                    "matchup": g.matchup,
                    "model": str(m),
                    "status": status,
                    "p_home": round(p_home, 6),
                    "mid_minus_p": round(mid_minus_p, 6),
                    "edge_net": round(float(edge_net), 6),
                    "delta_thr": round(delta_thr, 6),
                    "edge_thr": round(edge_thr, 6),
                    "delta_ok": int(delta_ok),
                    "edge_ok": int(edge_ok),
                    "flip": int(is_flip),
                    "vote": int(is_vote),
                    "counted": int(counted),
                    "reason": reason,
                }
            )

    out_dir = Path(str(args.out_dir).strip())
    out_dir.mkdir(parents=True, exist_ok=True)
    elig_tag = "noelig" if bool(args.no_eligibility) else "elig"
    tag = f"{date_iso.replace('-', '')}_m{int(args.anchor_minutes)}_s{int(args.slippage_cents)}_{elig_tag}"

    trades_csv = out_dir / f"rule_a_hybrid_trades_{tag}.csv"
    models_csv = out_dir / f"rule_a_hybrid_model_breakdown_{tag}.csv"

    _read_rows(
        trades_csv,
        fieldnames=[
            "date",
            "league",
            "matchup",
            "mid_home",
            "price_away",
            "mid_bucket",
            "weak_bucket",
            "contracts",
            "triggers",
            "home_win",
            "gross_pnl",
            "fees",
            "net_pnl",
        ],
        rows=trade_rows,
    )
    _read_rows(
        models_csv,
        fieldnames=[
            "date",
            "league",
            "matchup",
            "model",
            "status",
            "p_home",
            "mid_minus_p",
            "edge_net",
            "delta_thr",
            "edge_thr",
            "delta_ok",
            "edge_ok",
            "flip",
            "vote",
            "counted",
            "reason",
        ],
        rows=model_rows,
    )

    def _fmt(lg: str) -> str:
        net = float(totals_by_league[lg]["net"])
        risk = float(totals_by_league[lg]["risk"])
        roi = 0.0 if risk <= 0 else net / risk
        return f"{lg}: net={net:+.2f} risk={risk:.2f} roi={100*roi:+.2f}% contracts={int(totals_by_league[lg]['contracts'])}"

    print(f"[ok] wrote {trades_csv}")
    print(f"[ok] wrote {models_csv}")
    print(_fmt("nba"))
    print(_fmt("nhl"))
    print(_fmt("nfl"))
    print(_fmt("overall"))


if __name__ == "__main__":
    main()
