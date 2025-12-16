"""
ETL for NHL process factors (xGF%, HDCF%, PP/PK strength, goalie rating).

Source expectation: local CSVs (e.g., downloaded from MoneyPuck) with columns:
  team,xgf_for,xgf_against,hdcf_for,hdca,pp_pct,pk_pct,goalie_gsax60
Column names are flexible; see CANDIDATE_COLS below.

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/etl_nhl_factors.py \
    --teams-csv data/nhl_team_metrics.csv \
    --goalies-csv data/nhl_goalie_metrics.csv \
    --out chimera_v2c/data/team_four_factors_nhl.json

Notes:
- No network access; you must download CSVs separately.
- Missing values fall back to neutral (0.5 for shares, 0 for z-scores/goalie).
- Output is namespaced by league: {"NHL": {TEAM: {...}}}
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


CANDIDATE_COLS = {
    "team": ["team", "Team", "TEAM"],
    "xgf_for": ["xgf_for", "xGF", "xGoalsFor", "xGoals"],
    "xgf_against": ["xgf_against", "xGA", "xGoalsAgainst"],
    "hdcf_for": ["hdcf_for", "HDCF", "hdcf"],
    "hdca": ["hdca", "HDCA", "hdca_for"],
    "pp_pct": ["pp_pct", "PP%", "pp_percent", "pp_efficiency"],
    "pk_pct": ["pk_pct", "PK%", "pk_percent", "pk_efficiency"],
    "goalie_gsax60": ["goalie_gsax60", "gsax_per60", "gsa_x_per60", "GSAx/60"],
    "oi_sh_pct": ["oi_sh_pct", "on_ice_sv_pct", "oi_sv_pct", "team_sv_pct", "sv_pct"],
}


def _pick(row: Dict, keys: List[str], default: float = None) -> float:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            try:
                return float(row[k])
            except Exception:
                continue
    return default


def _safe_pct(num: float, denom: float, default: float = 0.5) -> float:
    try:
        if num is None or denom is None:
            return default
        denom = float(denom)
        if denom <= 0:
            return default
        return max(0.0, min(1.0, float(num) / denom))
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Build NHL factors JSON from local CSVs.")
    ap.add_argument("--teams-csv", default="data/nhl_team_metrics.csv", help="Team-level CSV (xGF/HDCF/PP/PK).")
    ap.add_argument("--goalies-csv", default="data/nhl_goalie_metrics.csv", help="Goalie-level CSV (per-team GSAx/60).")
    ap.add_argument("--out", default="chimera_v2c/data/team_four_factors_nhl.json", help="Output JSON path.")
    ap.add_argument("--window", type=int, default=10, help="Rolling window of games for recency smoothing (default 10).")
    ap.add_argument("--alpha", type=float, default=0.85, help="Exponential decay factor (newest weight).")
    args = ap.parse_args()

    teams_path = Path(args.teams_csv)
    goalies_path = Path(args.goalies_csv)
    out_path = Path(args.out)

    if not teams_path.exists():
        raise SystemExit(f"[error] teams CSV missing at {teams_path}")
    df = pd.read_csv(teams_path)

    # Optional goalie metrics
    goalie_df = pd.DataFrame()
    if goalies_path.exists():
        goalie_df = pd.read_csv(goalies_path)

    factors: Dict[str, Dict[str, float]] = {}

    def _exp_avg(series: List[float], alpha: float) -> float:
        if not series:
            return 0.0
        weight = 1.0
        num = 0.0
        den = 0.0
        for val in reversed(series):  # newest last -> highest weight
            num += val * weight
            den += weight
            weight *= alpha
        return num / den if den else 0.0

    def goalie_rating_for(team: str) -> float:
        if goalie_df.empty:
            return 0.0
        rows = goalie_df[goalie_df.apply(lambda r: str(r).upper().__contains__(team.upper()), axis=1)]
        if rows.empty:
            return 0.0
        vals = []
        for _, r in rows.iterrows():
            vals.append(_pick(r, CANDIDATE_COLS["goalie_gsax60"], default=None))
        vals = [v for v in vals if v is not None]
        if not vals:
            return 0.0
        gsax_min = min(vals)
        gsax_max = max(vals)
        if gsax_max == gsax_min:
            return 0.5
        # Scale to 0-1
        return (np.mean(vals) - gsax_min) / (gsax_max - gsax_min)

    rows_out = []
    # If per-game data present, smooth by team
    if "game_date" in df.columns or "date" in df.columns:
        date_col = "game_date" if "game_date" in df.columns else "date"
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
        for team, grp in df.groupby(df.apply(lambda r: _pick(r, CANDIDATE_COLS["team"], default=None), axis=1)):
            if not team:
                continue
            team = str(team).upper()
            recency_slice = grp.tail(args.window)
            xgf_series = []
            hdcf_series = []
            pp_series = []
            pk_series = []
            oi_sh_series = []
            for _, r in recency_slice.iterrows():
                rdict = r.to_dict()
                xf = _pick(rdict, CANDIDATE_COLS["xgf_for"], default=None)
                xa = _pick(rdict, CANDIDATE_COLS["xgf_against"], default=None)
                hf = _pick(rdict, CANDIDATE_COLS["hdcf_for"], default=None)
                ha = _pick(rdict, CANDIDATE_COLS["hdca"], default=None)
                ppv = _pick(rdict, CANDIDATE_COLS["pp_pct"], default=None)
                pkv = _pick(rdict, CANDIDATE_COLS["pk_pct"], default=None)
                oi_sv = _pick(rdict, CANDIDATE_COLS["oi_sh_pct"], default=None)
                xgf_series.append(_safe_pct(xf, (xf or 0) + (xa or 0)))
                hdcf_series.append(_safe_pct(hf, (hf or 0) + (ha or 0)))
                if ppv is not None:
                    pp_series.append(ppv)
                if pkv is not None:
                    pk_series.append(pkv)
                if oi_sv is not None:
                    oi_sh_series.append(oi_sv)

            factors[team] = {
                "xgf_pct": _exp_avg(xgf_series, args.alpha) if xgf_series else 0.5,
                "hdcf_pct": _exp_avg(hdcf_series, args.alpha) if hdcf_series else 0.5,
                "pp_index": _exp_avg(pp_series, args.alpha) if pp_series else 0.0,
                "pk_index": _exp_avg(pk_series, args.alpha) if pk_series else 0.0,
                "oi_sh_pct": _exp_avg(oi_sh_series, args.alpha) if oi_sh_series else 0.5,
                "goalie_rating": goalie_rating_for(team),
            }
            rows_out.append(team)
    else:
        for _, row in df.iterrows():
            r = row.to_dict()
            team = _pick(r, CANDIDATE_COLS["team"])
            if not team:
                continue
            xgf_for = _pick(r, CANDIDATE_COLS["xgf_for"])
            xgf_against = _pick(r, CANDIDATE_COLS["xgf_against"])
            hdcf_for = _pick(r, CANDIDATE_COLS["hdcf_for"])
            hdca = _pick(r, CANDIDATE_COLS["hdca"])
            pp_pct = _pick(r, CANDIDATE_COLS["pp_pct"])
            pk_pct = _pick(r, CANDIDATE_COLS["pk_pct"])
            oi_sv = _pick(r, CANDIDATE_COLS["oi_sh_pct"])

            factors[team.upper()] = {
                "xgf_pct": _safe_pct(xgf_for, (xgf_for or 0) + (xgf_against or 0)),
                "hdcf_pct": _safe_pct(hdcf_for, (hdcf_for or 0) + (hdca or 0)),
                "pp_index": None if pp_pct is None else float(pp_pct),
                "pk_index": None if pk_pct is None else float(pk_pct),
                "oi_sh_pct": None if oi_sv is None else float(oi_sv),
                "goalie_rating": goalie_rating_for(team),
            }
            rows_out.append(team.upper())

    # Z-score PP/PK across teams
    def zscore(values: List[float], default: float = 0.0) -> Dict[str, float]:
        arr = np.array(values, dtype=float)
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        if std == 0 or np.isnan(std):
            return {rows_out[i]: default for i in range(len(values))}
        return {rows_out[i]: float((v - mean) / std) if not np.isnan(v) else default for i, v in enumerate(values)}

    pp_vals = [factors[t].get("pp_index") for t in rows_out]
    pk_vals = [factors[t].get("pk_index") for t in rows_out]
    oi_vals = [factors[t].get("oi_sh_pct") for t in rows_out]
    pp_z = zscore([v if v is not None else np.nan for v in pp_vals])
    pk_z = zscore([v if v is not None else np.nan for v in pk_vals])
    oi_z = zscore([v if v is not None else np.nan for v in oi_vals], default=0.5)
    for t in rows_out:
        factors[t]["pp_index"] = pp_z.get(t, 0.0)
        factors[t]["pk_index"] = pk_z.get(t, 0.0)
        factors[t]["oi_sh_pct"] = oi_z.get(t, 0.5)
        if factors[t].get("goalie_rating") is None:
            factors[t]["goalie_rating"] = 0.0

    payload = {
        "NHL": factors,
        "metadata": {"generated_at": datetime.utcnow().isoformat() + "Z", "window": args.window, "alpha": args.alpha},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ok] wrote {len(factors)} teams to {out_path}")


if __name__ == "__main__":
    main()
