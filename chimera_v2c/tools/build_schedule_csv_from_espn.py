#!/usr/bin/env python
"""
Build a league schedule CSV (for parity audits) from ESPN scoreboards.

This writes under `reports/thesis_summaries/` and is intended to support
`chimera_v2c/tools/audit_ledger_parity.py`, which expects schedule CSVs named:
  - reports/thesis_summaries/nba_schedule_*.csv
  - reports/thesis_summaries/nhl_schedule_*.csv
  - reports/thesis_summaries/nfl_schedule_*.csv

Schema (minimum required by audit_ledger_parity):
  - date (YYYY-MM-DD)
  - away
  - home
  - away_score (float)
  - home_score (float)

Notes:
  - For non-final games, scores are set to 0/0 and outcome is left blank; this
    keeps keys present for phantom detection without asserting a final score.
  - Team codes are normalized via `team_mapper.normalize_team_code`.

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/build_schedule_csv_from_espn.py \
    --league nba --start-date 2025-11-19 --end-date 2025-12-16
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from chimera_v2c.lib import nhl_scoreboard, team_mapper
from chimera_v2c.src.ledger.outcomes import format_final_score


OUT_DIR = Path("reports/thesis_summaries")


def _daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur = cur + dt.timedelta(days=1)


def _fetch_scoreboard(league: str, date_iso: str) -> Dict[str, Any]:
    league_l = league.lower()
    if league_l == "nba":
        return nhl_scoreboard.fetch_nba_scoreboard(date_iso)
    if league_l == "nfl":
        return nhl_scoreboard.fetch_nfl_scoreboard(date_iso)
    if league_l == "nhl":
        return nhl_scoreboard.fetch_nhl_scoreboard(date_iso)
    raise ValueError(f"Unsupported league: {league}")


def scoreboard_to_schedule_rows(*, league: str, date_iso: str, sb: Dict[str, Any]) -> List[Dict[str, object]]:
    league_l = league.lower()
    out: List[Dict[str, object]] = []
    if not isinstance(sb, dict) or sb.get("status") not in {"ok", "empty"}:
        return out
    for g in sb.get("games") or []:
        teams = g.get("teams") or {}
        away_alias = ((teams.get("away") or {})).get("alias")
        home_alias = ((teams.get("home") or {})).get("alias")
        if not away_alias or not home_alias:
            continue

        away = team_mapper.normalize_team_code(away_alias, league_l) or str(away_alias).strip().upper()
        home = team_mapper.normalize_team_code(home_alias, league_l) or str(home_alias).strip().upper()

        status = g.get("status") or {}
        state = (status.get("state") or "").lower()
        scores = g.get("scores") or {}
        away_score: float = 0.0
        home_score: float = 0.0
        outcome = ""
        if state == "post":
            try:
                away_score = float(scores.get("away"))
                home_score = float(scores.get("home"))
                outcome = format_final_score(away, home, away_score, home_score)
            except (TypeError, ValueError):
                away_score, home_score, outcome = 0.0, 0.0, ""

        out.append(
            {
                "date": date_iso,
                "league": league_l,
                "away": away,
                "home": home,
                "away_score": away_score,
                "home_score": home_score,
                "outcome": outcome,
                "status_state": state,
            }
        )
    return out


def build_schedule_csv(*, league: str, start_date: str, end_date: str, out_path: Optional[Path]) -> Path:
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    if end < start:
        raise ValueError("end_date must be >= start_date")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = OUT_DIR / f"{league.lower()}_schedule_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"

    rows: List[Dict[str, object]] = []
    for d in _daterange(start, end):
        date_iso = d.isoformat()
        sb = _fetch_scoreboard(league, date_iso)
        rows.extend(scoreboard_to_schedule_rows(league=league, date_iso=date_iso, sb=sb))

    fieldnames = ["date", "league", "away", "home", "away_score", "home_score", "outcome", "status_state"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build schedule CSVs for parity audits from ESPN scoreboards.")
    ap.add_argument("--league", required=True, help="League: nba|nhl|nfl")
    ap.add_argument("--start-date", required=True, help="ISO date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end-date", required=True, help="ISO date YYYY-MM-DD (inclusive)")
    ap.add_argument("--out", help="Optional output CSV path (default: reports/thesis_summaries/<league>_schedule_*.csv)")
    args = ap.parse_args()

    out_path = build_schedule_csv(
        league=args.league,
        start_date=args.start_date,
        end_date=args.end_date,
        out_path=Path(args.out) if args.out else None,
    )
    print(f"[ok] wrote schedule -> {out_path}")


if __name__ == "__main__":
    main()

