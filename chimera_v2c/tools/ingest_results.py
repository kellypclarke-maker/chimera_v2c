"""
Ingest v2c execution logs and stamp win/loss results into a v2c-specific ledger.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/ingest_results.py --date 2025-12-03 --league nba

Inputs:
  - reports/execution_logs/v2c_execution_log.csv (placed orders)
  - ESPN scoreboard via app.services.nhl_scoreboard (nba/nfl/nhl)

Output:
  - reports/execution_logs/v2c_results.csv (append/update per ticker+date)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo imports work when run directly
import os

sys.path.insert(0, os.getcwd())

from chimera_v2c.lib import nhl_scoreboard
from chimera_v2c.lib import team_mapper

EXEC_LOG = Path("reports/execution_logs/v2c_execution_log.csv")
RESULTS_LOG = Path("reports/execution_logs/v2c_results.csv")


def parse_ticker_teams(ticker: str, league: str) -> Optional[Tuple[str, str, str]]:
    """Return (home, away, pick) codes parsed from ticker; assumes ticker ends with -<TEAM>."""
    if not ticker:
        return None
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    pick = parts[-1].upper()
    core = parts[-2] if len(parts) >= 2 else parts[-1]
    teams = team_mapper.match_teams_in_string(core, league)
    if not teams or len(teams) < 2:
        return None
    return teams[0], teams[1], pick


def fetch_scoreboard(league: str, date_str: str) -> Dict:
    league_l = league.lower()
    if league_l == "nba":
        return nhl_scoreboard.fetch_nba_scoreboard(date_str)
    if league_l == "nfl":
        return nhl_scoreboard.fetch_nfl_scoreboard(date_str)
    return nhl_scoreboard.fetch_nhl_scoreboard(date_str)


def score_for_team(sb: Dict, team_code: str) -> Optional[int]:
    games = sb.get("games") or []
    for g in games:
        teams = g.get("teams") or {}
        home = team_mapper.normalize_team_code(teams.get("home", {}).get("alias"), g.get("league", "nba"))
        away = team_mapper.normalize_team_code(teams.get("away", {}).get("alias"), g.get("league", "nba"))
        if home == team_code:
            return g.get("scores", {}).get("home")
        if away == team_code:
            return g.get("scores", {}).get("away")
    return None


def load_execution_rows(target_date: str) -> List[Dict]:
    if not EXEC_LOG.exists():
        return []
    rows: List[Dict] = []
    with EXEC_LOG.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "placed":
                continue
            if row.get("date") != target_date:
                continue
            rows.append(row)
    return rows


def ensure_results_header(path: Path = RESULTS_LOG) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "date",
                "league",
                "ticker",
                "matchup",
                "pick",
                "price_cents",
                "edge",
                "stake_fraction",
                "result",
                "score_pick",
                "score_opp",
                "message",
            ]
        )


def upsert_result(result_row: Dict, path: Path) -> None:
    ensure_results_header(path)
    # Load existing to handle upsert by (date, ticker)
    existing: List[Dict] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing.append(r)
    key = (result_row["date"], result_row["ticker"])
    wrote = False
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date",
            "league",
            "ticker",
            "matchup",
            "pick",
            "price_cents",
            "edge",
            "stake_fraction",
            "result",
            "score_pick",
            "score_opp",
            "message",
        ])
        writer.writeheader()
        for r in existing:
            if (r.get("date"), r.get("ticker")) == key:
                writer.writerow(result_row)
                wrote = True
            else:
                writer.writerow(r)
        if not wrote:
            writer.writerow(result_row)


def ingest_results(target_date: str, league: str) -> None:
    rows = load_execution_rows(target_date)
    if not rows:
        print(f"[info] no placed orders on {target_date}")
        return
    sb = fetch_scoreboard(league, target_date)
    if sb.get("status") == "error":
        print(f"[error] scoreboard fetch failed: {sb.get('message')}")
        return

    for row in rows:
        ticker = row.get("ticker") or ""
        parsed = parse_ticker_teams(ticker, league)
        if not parsed:
            print(f"[warn] could not parse teams from {ticker}")
            continue
        home, away, pick = parsed
        opp = away if pick == home else home
        s_pick = score_for_team(sb, pick)
        s_opp = score_for_team(sb, opp)
        if s_pick is None or s_opp is None:
            message = "score_missing"
            result = "unknown"
        else:
            if s_pick > s_opp:
                result = "win"
            elif s_pick < s_opp:
                result = "loss"
            else:
                result = "push"
            message = ""
        matchup = f"{away}@{home}"
        result_row = {
            "date": target_date,
            "league": league.lower(),
            "ticker": ticker,
            "matchup": matchup,
            "pick": pick,
            "price_cents": row.get("price_cents"),
            "edge": row.get("edge"),
            "stake_fraction": row.get("stake_fraction"),
            "result": result,
            "score_pick": s_pick if s_pick is not None else "",
            "score_opp": s_opp if s_opp is not None else "",
            "message": message,
        }
        upsert_result(result_row, path=RESULTS_LOG)
        print(f"[info] {ticker}: {result} ({s_pick}-{s_opp})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest v2c execution log results into v2c ledger.")
    ap.add_argument("--date", required=True, help="Game date YYYY-MM-DD")
    ap.add_argument("--league", default="nba", help="League (nba/nfl/nhl)")
    args = ap.parse_args()
    ingest_results(args.date, args.league)


if __name__ == "__main__":
    main()
