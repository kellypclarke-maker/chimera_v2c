"""
NHL backtest: join v2c plan log with outcomes and simulate ROI by probability buckets.

Inputs:
- Plan log: reports/execution_logs/v2c_plan_log.json (from log_plan.py)
- Outcomes: reports/daily_ledgers/YYYYMMDD_daily_game_ledger.csv

Assumptions:
- Plan log rows have p_yes (model), mid (market), yes_team, home/away, matchup.
- Outcome rows have home/away aliases and either home_win or scores.
- Quarter-Kelly sizing via stake_calculator.compute_stake_fraction (cap 1%).
- Profit per bet (bankroll fraction): stake * ((1 - price) / price if win else -1.0), price = mid (probability units).

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/nhl_backtest.py --date-range 2025-10-01:2025-12-09 --league nhl

Outputs:
- CSV/print table with buckets (0.05) => count, hit_rate, avg_edge, ev_model, ev_market.
- --override-buckets simulates without negative ROI bucket guardrails (log-only).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from chimera_v2c.lib import team_mapper
from chimera_v2c.lib.stake_calculator import compute_stake_fraction
from chimera_v2c.src.ledger.outcomes import parse_home_win


PLAN_LOG = Path("reports/execution_logs/v2c_plan_log.json")
LEDGER_DIR = Path("reports/daily_ledgers")

_TICKER_PREFIX_TO_LEAGUE = {
    "KXNBAGAME": "nba",
    "KXNHLGAME": "nhl",
    "KXNFLGAME": "nfl",
}

_TEAM_CODES_BY_LEAGUE = {
    "nba": set(team_mapper.NBA_TEAMS.keys()),
    "nhl": set(team_mapper.NHL_TEAMS.keys()),
    "nfl": set(team_mapper.NFL_TEAMS.keys()),
}


@dataclass
class PlanRow:
    date: str
    league: Optional[str]
    matchup: str
    home: str
    away: str
    yes_team: str
    p_yes: float
    mid: float
    stake_fraction: Optional[float]
    selected: Optional[bool]


def _infer_league(entry: dict) -> Optional[str]:
    league = entry.get("league")
    if league:
        return str(league).strip().lower()

    ticker = entry.get("ticker") or ""
    if ticker:
        for prefix, lg in _TICKER_PREFIX_TO_LEAGUE.items():
            if str(ticker).startswith(prefix):
                return lg

    home = (entry.get("home") or "").strip().upper()
    away = (entry.get("away") or "").strip().upper()
    if not home or not away:
        return None
    leagues_home = {lg for lg, codes in _TEAM_CODES_BY_LEAGUE.items() if home in codes}
    leagues_away = {lg for lg, codes in _TEAM_CODES_BY_LEAGUE.items() if away in codes}
    inter = leagues_home & leagues_away
    if len(inter) == 1:
        return next(iter(inter))
    return None


def _parse_optional_float(val: object) -> Optional[float]:
    if val is None:
        return None
    try:
        s = str(val).strip()
    except Exception:
        return None
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_optional_bool(val: object) -> Optional[bool]:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    try:
        s = str(val).strip().lower()
    except Exception:
        return None
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def load_plans(date_range: Tuple[str, str], league: str, *, include_all_sides: bool = False) -> List[PlanRow]:
    if not PLAN_LOG.exists():
        raise SystemExit(f"[error] plan log missing: {PLAN_LOG}")
    try:
        data = json.loads(PLAN_LOG.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"[error] failed to parse plan log: {exc}")
    start, end = date_range

    by_game: Dict[Tuple[str, str], List[PlanRow]] = defaultdict(list)
    league_norm = league.lower()
    for entry in data:
        d = entry.get("date")
        if not d or d < start or d > end:
            continue
        inferred_league = _infer_league(entry)
        if inferred_league is None or inferred_league != league_norm:
            continue
        matchup = entry.get("matchup") or ""
        yes_team = entry.get("yes_team")
        if not yes_team:
            continue
        try:
            p_yes = float(entry.get("p_yes"))
            mid = float(entry.get("mid")) if entry.get("mid") is not None else None
        except Exception:
            continue
        if mid is None:
            continue
        by_game[(d, matchup)].append(
            PlanRow(
                date=d,
                league=inferred_league,
                matchup=matchup,
                home=(entry.get("home") or "").upper(),
                away=(entry.get("away") or "").upper(),
                yes_team=yes_team.upper(),
                p_yes=p_yes,
                mid=mid,
                stake_fraction=_parse_optional_float(entry.get("stake_fraction")),
                selected=_parse_optional_bool(entry.get("selected")),
            )
        )

    rows: List[PlanRow] = []
    for _, candidates in sorted(by_game.items()):
        if include_all_sides:
            rows.extend(candidates)
            continue
        selected = [c for c in candidates if c.selected is True]
        if selected:
            # One-side-per-game; allow multiple selected only if historical data is malformed.
            rows.extend(selected)
            continue
        # Back-compat: old plan logs did not include `selected`; pick the highest-edge YES bet.
        best: Optional[PlanRow] = None
        best_edge: Optional[float] = None
        for c in candidates:
            edge = c.p_yes - c.mid
            if edge <= 0:
                continue
            if best is None or edge > (best_edge if best_edge is not None else -1e9):
                best = c
                best_edge = edge
        if best is not None:
            rows.append(best)

    return rows


def load_outcome(date: str, home: str, away: str) -> Optional[int]:
    path = LEDGER_DIR / f"{date.replace('-', '')}_daily_game_ledger.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    home_u = home.upper()
    away_u = away.upper()
    home_col = "home" if "home" in df.columns else ("home_team" if "home_team" in df.columns else None)
    away_col = "away" if "away" in df.columns else ("away_team" if "away_team" in df.columns else None)
    row = pd.DataFrame()
    if home_col and away_col:
        row = df[(df[home_col].str.upper() == home_u) & (df[away_col].str.upper() == away_u)]
    if row.empty and "matchup" in df.columns:
        # fallback: matchup like away@home
        row = df[df["matchup"].str.upper() == f"{away_u}@{home_u}"]
    if row.empty:
        return None
    r = row.iloc[0]
    if "home_win" in r and r["home_win"] in (0, 1, True, False):
        return int(bool(r["home_win"]))
    actual = r.get("actual_outcome")
    if actual is not None:
        hw = parse_home_win(actual)
        if hw == 1.0:
            return 1
        if hw == 0.0:
            return 0
        if hw == 0.5:
            return None
    hs = r.get("home_score")
    as_ = r.get("away_score")
    try:
        hs_val = float(hs)
        as_val = float(as_)
        return int(hs_val > as_val)
    except Exception:
        return None


def bucket_key(p: float) -> str:
    width = 0.05
    p = max(0.0, min(1.0, float(p)))
    idx = int(math.floor((p + 1e-12) / width))
    lo = idx * width
    lo = max(0.0, min(lo, 1.0 - width))
    hi = min(1.0, lo + width)
    return f"[{lo:.2f},{hi:.2f})"


def fill_probability(edge: float) -> float:
    # Simple maker fill heuristic: closer to mid => higher fill. Clamp to [0.3, 0.95].
    distance = abs(edge)
    return max(0.3, min(0.95, 0.95 - distance * 5.0))


def simulate(plans: List[PlanRow], override_guardrails: bool) -> pd.DataFrame:
    records = []
    for p in plans:
        outcome = load_outcome(p.date, p.home, p.away)
        if outcome is None:
            continue
        # outcome refers to home win; yes_win indicates if the yes_team won
        yes_win = outcome == 1 if p.yes_team == p.home else outcome == 0
        price = max(0.01, min(0.99, p.mid))
        stake = p.stake_fraction
        if stake is None:
            stake = compute_stake_fraction(p.p_yes, price, max_fraction=0.01) or 0.0
        if stake <= 0:
            continue
        edge_val = p.p_yes - price
        fill_prob = fill_probability(edge_val)
        roi_per_cost = ((1.0 - price) / price) if yes_win else -1.0
        profit_model = fill_prob * stake * roi_per_cost
        profit_market = profit_model
        records.append(
            {
                "bucket": bucket_key(p.p_yes),
                "edge": edge_val,
                "hit": 1 if yes_win else 0,
                "profit_model": profit_model,
                "profit_market": profit_market,
                "stake": stake,
                "calib_impact": None,  # placeholder; plan log lacks pre-cal probabilities
                "fill_prob": fill_prob,
            }
        )
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    grouped = (
        df.groupby("bucket")
        .agg(
            count=("hit", "size"),
            hit_rate=("hit", "mean"),
            avg_edge=("edge", "mean"),
            ev_model=("profit_model", "mean"),
            ev_market=("profit_market", "mean"),
            avg_fill_prob=("fill_prob", "mean"),
        )
        .reset_index()
        .sort_values("bucket")
    )
    if override_guardrails:
        grouped["guardrail_override"] = True
    return grouped


def parse_date_range(val: str) -> Tuple[str, str]:
    if ":" in val:
        a, b = val.split(":", 1)
        return a, b
    return val, val


def main() -> None:
    ap = argparse.ArgumentParser(description="NHL backtest on plan log vs outcomes.")
    ap.add_argument("--date-range", required=True, help="YYYY-MM-DD:YYYY-MM-DD or single date.")
    ap.add_argument("--league", default="nhl")
    ap.add_argument("--out", help="Optional CSV output path.")
    ap.add_argument("--override-buckets", action="store_true", help="Flag results where guardrails would block.")
    ap.add_argument(
        "--all-sides",
        action="store_true",
        help="Include both sides per game when present (default: one side per game).",
    )
    args = ap.parse_args()

    date_range = parse_date_range(args.date_range)
    plans = load_plans(date_range, args.league, include_all_sides=args.all_sides)
    if not plans:
        raise SystemExit("[error] no plans found in date range")
    summary = simulate(plans, args.override_buckets)
    if summary.empty:
        raise SystemExit("[error] no matched outcomes for backtest")

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
        print(f"[ok] wrote backtest summary to {args.out}")


if __name__ == "__main__":
    main()
