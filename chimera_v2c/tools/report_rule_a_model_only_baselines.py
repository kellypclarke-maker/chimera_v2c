#!/usr/bin/env python3
"""
Report per-model "model-only Rule A" performance (read-only).

Definition (per model m, per league):
  - Qualifying games: Kalshi favors HOME at the anchor (home mid > 0.50).
  - Trade: buy AWAY YES at away ask (+ slippage), taker fees.
  - No blind: bet 1 contract ONLY when model m says the home favorite is overpriced, i.e.
      - vote delta gate passes: (mid_home - p_home_m) >= vote_delta_m
      - fee-aware edge gate passes: expected_edge_net_per_contract(p_home_m, price_away) >= vote_edge_m
    OR when flip/I passes:
      - p_home_m < 0.50 AND edge gate passes
      - if flip_delta_mode == "same", also require delta gate for flips.

Thresholds are taken from per-league VoteDeltaCalibration JSONs under chimera_v2c/data/
unless overridden.

Outputs a CSV summary by (league, model) plus overall aggregates.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, load_games
from chimera_v2c.src.rule_a_unit_policy import expected_edge_net_per_contract, iter_rule_a_games, load_bidask_csv, net_pnl_taker
from chimera_v2c.src.rule_a_vote_calibration import VoteDeltaCalibration


LEAGUES = ["nba", "nhl", "nfl"]


@dataclass
class Totals:
    bets: int = 0
    contracts: int = 0
    risked: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0

    @property
    def roi_net(self) -> float:
        return 0.0 if float(self.risked) <= 0 else float(self.net_pnl) / float(self.risked)


def _default_calibration_path(league: str) -> Path:
    return Path("chimera_v2c/data") / f"rule_a_vote_calibration_{league}.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report per-model model-only Rule A performance (read-only).")
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive.")
    p.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive.")
    p.add_argument("--bidask-csv", required=True, help="kalshi_bidask_tminus_*.csv (may be a superset over the window).")
    p.add_argument("--anchor-minutes", type=int, default=30, help="T-minus anchor minutes (default: 30).")
    p.add_argument("--slippage-cents", type=int, default=1, help="Slippage cents added to away ask (default: 1).")
    p.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="Optional explicit model list. If omitted, uses union of models from per-league vote calibration JSONs.",
    )
    p.add_argument("--out", default="", help="Optional output CSV path (default under reports/thesis_summaries/).")
    return p.parse_args()


def _model_triggers_one(
    g,
    *,
    model: str,
    cfg: VoteDeltaCalibration,
) -> bool:
    p = g.probs.get(str(model))
    if p is None:
        return False

    delta_thr = float(cfg.vote_delta_by_model.get(str(model), cfg.vote_delta_default))
    edge_thr = float(cfg.vote_edge_by_model.get(str(model), cfg.vote_edge_default))

    mid_minus_p = float(g.mid_home) - float(p)
    delta_ok = mid_minus_p >= float(delta_thr)

    edge_net = expected_edge_net_per_contract(p_home=float(p), price_away=float(g.price_away))
    edge_ok = float(edge_net) >= float(edge_thr)

    is_flip = float(p) < 0.5
    if is_flip and edge_ok:
        if str(cfg.flip_delta_mode) == "same" and not delta_ok:
            return False
        return True

    return bool(delta_ok and edge_ok)


def _apply_one(t: Totals, *, contracts: int, price_away: float, home_win: int) -> None:
    if int(contracts) <= 0:
        return
    gross, fees, risked = net_pnl_taker(contracts=int(contracts), price_away=float(price_away), home_win=int(home_win))
    t.bets += 1
    t.contracts += int(contracts)
    t.risked += float(risked)
    t.fees += float(fees)
    t.net_pnl += float(gross - fees)


def main() -> None:
    args = _parse_args()
    start = str(args.start_date).strip()
    end = str(args.end_date).strip()
    bidask_path = Path(str(args.bidask_csv).strip())
    if not bidask_path.exists():
        raise SystemExit(f"[error] missing --bidask-csv: {bidask_path}")

    bidask = load_bidask_csv(bidask_path)

    cfg_by_league: Dict[str, VoteDeltaCalibration] = {}
    for lg in LEAGUES:
        p = _default_calibration_path(lg)
        if not p.exists():
            raise SystemExit(f"[error] missing vote calibration JSON: {p}")
        cfg_by_league[lg] = VoteDeltaCalibration.load_json(p)

    models: List[str]
    if args.models:
        models = [str(m) for m in args.models]
    else:
        models = sorted({m for lg in LEAGUES for m in cfg_by_league[lg].models})

    # Load ledger games once with all needed model columns.
    ledger_games = load_games(daily_dir=LEDGER_DIR, start_date=start, end_date=end, models=list(models))
    if not ledger_games:
        raise SystemExit("[error] no games loaded from daily ledgers for window")

    ra_games = iter_rule_a_games(
        ledger_games,
        bidask=bidask,
        models=list(models),
        slippage_cents=int(args.slippage_cents),
        require_outcome=True,
    )
    if not ra_games:
        raise SystemExit("[error] no qualifying Rule A games found for window + snapshot")

    totals: Dict[Tuple[str, str], Totals] = {}

    def tot(league: str, model: str) -> Totals:
        k = (league, model)
        if k not in totals:
            totals[k] = Totals()
        return totals[k]

    for g in ra_games:
        league = str(g.league).strip().lower()
        if league not in cfg_by_league:
            continue
        cfg = cfg_by_league[league]
        if g.home_win is None:
            continue

        for m in models:
            trig = _model_triggers_one(g, model=str(m), cfg=cfg)
            contracts = 1 if trig else 0
            _apply_one(tot(league, str(m)), contracts=contracts, price_away=float(g.price_away), home_win=int(g.home_win))
            _apply_one(tot("overall", str(m)), contracts=contracts, price_away=float(g.price_away), home_win=int(g.home_win))

    rows: List[Dict[str, object]] = []
    for league in ["overall"] + LEAGUES:
        for m in models:
            t = totals.get((league, m), Totals())
            rows.append(
                {
                    "window_start": start,
                    "window_end": end,
                    "league": league,
                    "model": m,
                    "bets": int(t.bets),
                    "contracts": int(t.contracts),
                    "risked": round(float(t.risked), 6),
                    "fees": round(float(t.fees), 6),
                    "net_pnl": round(float(t.net_pnl), 6),
                    "roi_net": round(float(t.roi_net), 6),
                }
            )

    out_path = Path(str(args.out).strip()) if str(args.out).strip() else Path("reports/thesis_summaries") / (
        f"rule_a_model_only_baselines_{start.replace('-', '')}_{end.replace('-', '')}_m{int(args.anchor_minutes)}_s{int(args.slippage_cents)}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()

