#!/usr/bin/env python
"""
Build per-league ROI-by-probability-bucket guardrails from v2c plan logs + daily ledgers.

This tool is READ-ONLY on daily ledgers (it only reads `reports/daily_ledgers/`).

Inputs:
  - Plan log: `reports/execution_logs/v2c_plan_log.json` (from `log_plan.py`)
  - Outcomes: `reports/daily_ledgers/YYYYMMDD_daily_game_ledger.csv`

Output (default):
  - `reports/roi_by_bucket_<league>.csv`

Notes:
  - Uses one-side-per-game by default (prefers `selected=true` rows when present).
  - When old plan-log rows lack `selected`/`stake_fraction`, it falls back to:
      * pick the highest-edge side (p_yes - mid) per game, and
      * size stake via `stake_calculator.compute_stake_fraction` (quarter-Kelly, capped).
  - ROI is computed per dollar spent on YES contracts at the logged mid price:
      * win:  (1 - price) / price
      * loss: -1
    The per-bucket `roi_estimate` is stake-weighted: total_profit / total_cost.

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/build_roi_by_bucket_guardrails.py \
      --league nhl --start-date 2025-12-04 --end-date 2025-12-13
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from chimera_v2c.lib import team_mapper
from chimera_v2c.lib.stake_calculator import compute_stake_fraction
from chimera_v2c.src.ledger_analysis import load_games


PLAN_LOG = Path("reports/execution_logs/v2c_plan_log.json")
DEFAULT_LEDGER_DIR = Path("reports/daily_ledgers")

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
class Candidate:
    date: str  # YYYY-MM-DD
    league: str
    matchup: str  # AWAY@HOME
    yes_team: str
    p_yes: float
    mid: float
    edge: float
    selected: Optional[bool]
    stake_fraction: Optional[float]


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


def prob_bucket(p: float, *, width: float) -> str:
    if width <= 0:
        raise ValueError("bucket width must be > 0")
    p = max(0.0, min(1.0, float(p)))
    idx = int(math.floor((p + 1e-12) / width))
    lo = idx * width
    lo = max(0.0, min(lo, 1.0 - width))
    hi = min(1.0, lo + width)
    return f"[{lo:.2f},{hi:.2f})"


def _load_outcomes(
    *,
    daily_dir: Path,
    league: str,
    start_date: str,
    end_date: str,
) -> Dict[Tuple[str, str], float]:
    games = load_games(
        daily_dir=daily_dir,
        start_date=start_date,
        end_date=end_date,
        league_filter=league,
        models=[],
    )
    outcomes: Dict[Tuple[str, str], float] = {}
    for g in games:
        if g.home_win is None or g.home_win == 0.5:
            continue
        date_iso = g.date.strftime("%Y-%m-%d")
        matchup = (g.matchup or "").strip().upper()
        if not matchup:
            continue
        outcomes[(date_iso, matchup)] = float(g.home_win)
    return outcomes


def _parse_matchup(matchup: str) -> Optional[Tuple[str, str]]:
    s = (matchup or "").strip().upper()
    if "@" not in s:
        return None
    away, home = s.split("@", 1)
    away = away.strip()
    home = home.strip()
    if not away or not home:
        return None
    return away, home


def _select_one_side_per_game(
    candidates: List[Candidate],
    *,
    require_selected: bool,
) -> List[Candidate]:
    selected = [c for c in candidates if c.selected is True and (c.stake_fraction or 0.0) > 0.0]
    if selected:
        return selected
    if require_selected:
        return []
    best: Optional[Candidate] = None
    for c in candidates:
        if c.edge <= 0:
            continue
        if best is None or c.edge > best.edge:
            best = c
    return [best] if best else []


def build_roi_by_bucket(
    *,
    league: str,
    start_date: str,
    end_date: str,
    plan_log_path: Path = PLAN_LOG,
    daily_dir: Path = DEFAULT_LEDGER_DIR,
    bucket_width: float = 0.05,
    max_fraction: float = 0.01,
    min_bets: int = 10,
    require_selected: bool = False,
    include_all_sides: bool = False,
) -> List[Dict[str, object]]:
    if not plan_log_path.exists():
        raise SystemExit(f"[error] plan log missing: {plan_log_path}")
    try:
        raw = json.loads(plan_log_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"[error] failed to parse plan log: {exc}")
    if not isinstance(raw, list):
        raise SystemExit("[error] plan log must be a JSON list")

    outcomes = _load_outcomes(daily_dir=daily_dir, league=league, start_date=start_date, end_date=end_date)
    if not outcomes:
        raise SystemExit("[error] no graded outcomes found in daily ledgers for the date range")

    league_norm = league.lower()
    by_game: Dict[Tuple[str, str], List[Candidate]] = defaultdict(list)
    for entry in raw:
        strategy = str(entry.get("strategy") or "").strip().lower()
        if strategy and strategy != "v2c":
            continue
        d = entry.get("date")
        if not d or d < start_date or d > end_date:
            continue
        inferred_league = _infer_league(entry)
        if inferred_league is None or inferred_league != league_norm:
            continue
        matchup = (entry.get("matchup") or "").strip().upper()
        yes_team = (entry.get("yes_team") or "").strip().upper()
        if not matchup or not yes_team:
            continue
        try:
            p_yes = float(entry.get("p_yes"))
            mid = float(entry.get("mid"))
        except Exception:
            continue
        mid = max(0.01, min(0.99, mid))
        edge = p_yes - mid
        by_game[(str(d), matchup)].append(
            Candidate(
                date=str(d),
                league=inferred_league,
                matchup=matchup,
                yes_team=yes_team,
                p_yes=p_yes,
                mid=mid,
                edge=edge,
                selected=_parse_optional_bool(entry.get("selected")),
                stake_fraction=_parse_optional_float(entry.get("stake_fraction")),
            )
        )

    # Aggregate per bucket.
    agg: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "n": 0.0,
            "wins": 0.0,
            "sum_p": 0.0,
            "sum_mid": 0.0,
            "sum_edge": 0.0,
            "cost": 0.0,
            "profit": 0.0,
        }
    )

    for (date_iso, matchup), candidates in sorted(by_game.items()):
        if include_all_sides:
            chosen = candidates
        else:
            chosen = _select_one_side_per_game(candidates, require_selected=require_selected)
        if not chosen:
            continue
        home_win = outcomes.get((date_iso, matchup))
        if home_win is None:
            continue
        parsed = _parse_matchup(matchup)
        if not parsed:
            continue
        away, home = parsed
        for c in chosen:
            if c.yes_team == home:
                yes_win = home_win == 1.0
            elif c.yes_team == away:
                yes_win = home_win == 0.0
            else:
                continue

            stake = c.stake_fraction
            if stake is None:
                stake = compute_stake_fraction(c.p_yes, c.mid, max_fraction=max_fraction) or 0.0
            if stake <= 0:
                continue

            roi_per_cost = ((1.0 - c.mid) / c.mid) if yes_win else -1.0
            bucket = prob_bucket(c.p_yes, width=bucket_width)
            a = agg[bucket]
            a["n"] += 1.0
            a["wins"] += 1.0 if yes_win else 0.0
            a["sum_p"] += float(c.p_yes)
            a["sum_mid"] += float(c.mid)
            a["sum_edge"] += float(c.edge)
            a["cost"] += float(stake)
            a["profit"] += float(stake) * float(roi_per_cost)

    rows: List[Dict[str, object]] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for bucket in sorted(agg.keys()):
        a = agg[bucket]
        n = int(a["n"])
        if n <= 0:
            continue
        cost = float(a["cost"]) if a["cost"] else 0.0
        profit = float(a["profit"]) if a["profit"] else 0.0
        roi_all = (profit / cost) if cost > 0 else None
        eligible = n >= min_bets
        rows.append(
            {
                "bucket": bucket,
                "n_bets": n,
                "win_rate": (float(a["wins"]) / float(n)) if n else None,
                "avg_p_yes": (float(a["sum_p"]) / float(n)) if n else None,
                "avg_mid": (float(a["sum_mid"]) / float(n)) if n else None,
                "avg_edge": (float(a["sum_edge"]) / float(n)) if n else None,
                "total_cost_fraction": cost,
                "total_profit_fraction": profit,
                "roi_estimate_all": roi_all,
                "roi_estimate": roi_all if eligible else None,
                "eligible": eligible,
                "league": league_norm,
                "start_date": start_date,
                "end_date": end_date,
                "generated_at_utc": now,
            }
        )

    return rows


def write_csv(out_path: Path, rows: List[Dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bucket",
        "n_bets",
        "win_rate",
        "avg_p_yes",
        "avg_mid",
        "avg_edge",
        "total_cost_fraction",
        "total_profit_fraction",
        "roi_estimate_all",
        "roi_estimate",
        "eligible",
        "league",
        "start_date",
        "end_date",
        "generated_at_utc",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build ROI-by-bucket guardrails from plan logs + daily ledgers.")
    ap.add_argument("--league", required=True, help="nba|nhl|nfl")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--plan-log", default=str(PLAN_LOG), help=f"Plan log path (default: {PLAN_LOG})")
    ap.add_argument(
        "--daily-dir",
        default=str(DEFAULT_LEDGER_DIR),
        help=f"Daily ledger dir (default: {DEFAULT_LEDGER_DIR})",
    )
    ap.add_argument(
        "--out",
        help="Output CSV path (default: reports/roi_by_bucket_<league>.csv)",
    )
    ap.add_argument("--bucket-width", type=float, default=0.05, help="Probability bucket width (default: 0.05).")
    ap.add_argument(
        "--max-fraction",
        type=float,
        default=0.01,
        help="Max stake fraction for fallback sizing when stake_fraction missing (default: 0.01).",
    )
    ap.add_argument(
        "--min-bets",
        type=int,
        default=10,
        help="Minimum bets in a bucket to populate roi_estimate (default: 10).",
    )
    ap.add_argument(
        "--require-selected",
        action="store_true",
        help="Only use plan-log rows with selected=true and stake_fraction present (default: allow fallback).",
    )
    ap.add_argument(
        "--all-sides",
        action="store_true",
        help="Include both sides per game when present (default: one side per game).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    league = args.league.strip().lower()
    out_path = Path(args.out) if args.out else Path("reports") / f"roi_by_bucket_{league}.csv"
    rows = build_roi_by_bucket(
        league=league,
        start_date=args.start_date,
        end_date=args.end_date,
        plan_log_path=Path(args.plan_log),
        daily_dir=Path(args.daily_dir),
        bucket_width=float(args.bucket_width),
        max_fraction=float(args.max_fraction),
        min_bets=int(args.min_bets),
        require_selected=bool(args.require_selected),
        include_all_sides=bool(args.all_sides),
    )
    if not rows:
        raise SystemExit("[error] no rows produced (no matched plans/outcomes in range)")
    write_csv(out_path, rows)
    print(f"[ok] wrote {len(rows)} buckets to {out_path}")


if __name__ == "__main__":
    main()
