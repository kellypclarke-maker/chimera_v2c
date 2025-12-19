#!/usr/bin/env python3
"""
Report Rule A "v2c hybrid" model status per league (read-only).

Outputs a CSV that labels each candidate source as:
  - primary: n>=min_games and roi_lb90>0
  - secondary: n>=min_games and not primary (counts only when a primary triggers)
  - excluded: not enough games/signals OR never triggers under current gates

This is intended to make it obvious whether market_proxy / moneypuck are currently
helping under the production vote-calibration gates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from chimera_v2c.src.rule_a_model_eligibility import ModelEligibility, ModelPerf
from chimera_v2c.src.rule_a_vote_calibration import VoteDeltaCalibration


LEAGUES = ["nba", "nhl", "nfl"]


def _eligibility_path(league: str) -> Path:
    return Path("chimera_v2c/data") / f"rule_a_model_eligibility_{league}.json"


def _calibration_path(league: str) -> Path:
    return Path("chimera_v2c/data") / f"rule_a_vote_calibration_{league}.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report Rule A hybrid model status (primary/secondary/excluded) per league.")
    p.add_argument("--out", default="", help="Optional output CSV path (default: reports/thesis_summaries/rule_a_hybrid_model_status.csv).")
    return p.parse_args()


def _reason_for(
    *,
    model: str,
    perf: Optional[ModelPerf],
    elig: ModelEligibility,
    in_calibration: bool,
) -> str:
    if not in_calibration:
        return "excluded: not in vote calibration models"
    if perf is None:
        return "excluded: missing perf"
    if int(perf.n) <= 0:
        return "excluded: 0 signals under current gates"
    if int(perf.n) < int(elig.min_games):
        return f"excluded: n<{int(elig.min_games)}"
    if model in set(elig.primary_models):
        return "primary: roi_lb90>0"
    if model in set(elig.secondary_models):
        return "secondary: roi_lb90<=0 (counts only with primary)"
    return "excluded: not eligible"


def main() -> None:
    args = _parse_args()

    rows: List[Dict[str, object]] = []
    for league in LEAGUES:
        ep = _eligibility_path(league)
        cp = _calibration_path(league)
        if not ep.exists():
            raise SystemExit(f"[error] missing eligibility JSON: {ep} (run learn_rule_a_model_eligibility.py)")
        if not cp.exists():
            raise SystemExit(f"[error] missing vote calibration JSON: {cp} (run learn_rule_a_vote_calibration.py)")

        elig = ModelEligibility.load_json(ep)
        calib = VoteDeltaCalibration.load_json(cp)

        cal_models = [str(m) for m in calib.models]
        perf_by_model = dict(elig.perf_by_model)

        for m in cal_models:
            perf = perf_by_model.get(m)
            status = "excluded"
            if m in set(elig.primary_models):
                status = "primary"
            elif m in set(elig.secondary_models):
                status = "secondary"

            vote_delta = float(calib.vote_delta_by_model.get(m, calib.vote_delta_default))
            vote_edge = float(calib.vote_edge_by_model.get(m, calib.vote_edge_default))

            rows.append(
                {
                    "league": league,
                    "model": m,
                    "status": status,
                    "reason": _reason_for(model=m, perf=perf, elig=elig, in_calibration=True),
                    "signals_n": int(perf.n) if perf else 0,
                    "net_pnl": float(perf.net_pnl) if perf else 0.0,
                    "risked": float(perf.risked) if perf else 0.0,
                    "roi": float(perf.roi) if perf else 0.0,
                    "roi_lb90": float(perf.roi_lb90) if perf else 0.0,
                    "min_games": int(elig.min_games),
                    "confidence_level": float(elig.confidence_level),
                    "bootstrap_sims": int(elig.bootstrap_sims),
                    "trained_through": str(elig.trained_through),
                    "vote_delta": vote_delta,
                    "vote_edge": vote_edge,
                    "base_units": int(calib.base_units),
                    "cap_units": int(calib.cap_units),
                }
            )

    df = pd.DataFrame(rows).sort_values(["league", "status", "signals_n", "model"], ascending=[True, True, False, True])
    out = Path(str(args.out).strip()) if str(args.out).strip() else Path("reports/thesis_summaries/rule_a_hybrid_model_status.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()

