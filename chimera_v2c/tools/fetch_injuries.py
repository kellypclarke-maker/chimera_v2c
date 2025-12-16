"""
Automated injury scraper for NBA using ESPN's injury endpoint.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/fetch_injuries.py --date YYYY-MM-DD --halt

Outputs:
  - Updates chimera_v2c/data/injury_adjustments.json with Elo deltas per team for the given date.
  - Optional per-team halts via sentinel.halt_game when --halt is set.

Notes:
  - Status → delta mapping is simple and can be tuned: Out=-5.0, Doubtful=-2.5, Questionable=-1.5, Probable/Active=0.
  - If ESPN fails or returns empty, fall back to news_watcher with manual text input.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

from chimera_v2c.lib import espn_schedule
from chimera_v2c.lib import team_mapper
from chimera_v2c.src import sentinel

INJURY_PATH = Path("chimera_v2c/data/injury_adjustments.json")
RAW_INJURY_PATH = Path("chimera_v2c/data/raw_injuries.json")
ESPN_INJURY_SEARCH = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/athletes"


STATUS_DELTA = {
    "out": 0.0,
    "doubtful": 0.0,
    "questionable": 0.0,
    "day-to-day": 0.0,
    "probable": 0.0,
    "active": 0.0,
}


def load_injuries() -> Dict:
    if not INJURY_PATH.exists():
        return {}
    try:
        return json.loads(INJURY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_injuries(data: Dict) -> None:
    INJURY_PATH.parent.mkdir(parents=True, exist_ok=True)
    INJURY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_raw_injuries(raw: Dict) -> None:
    RAW_INJURY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_INJURY_PATH.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def fetch_matchup_teams(target_date: str) -> Dict[str, str]:
    """Return mapping of ESPN team_id -> normalized code for the slate."""
    sb = espn_schedule.get_scoreboard("nba", datetime.fromisoformat(target_date).date())
    teams: Dict[str, str] = {}
    for ev in sb.get("events", []):
        for comp in ev.get("competitions", []):
            for c in comp.get("competitors", []):
                team = c.get("team") or {}
                tid = team.get("id")
                abbr = team.get("abbreviation") or team.get("shortDisplayName")
                code = team_mapper.normalize_team_code(abbr, "nba")
                if tid and code:
                    teams[str(tid)] = code
    return teams


def fetch_team_injuries(team_code: str) -> List[Dict]:
    raise NotImplementedError("use fetch_injuries_for_slate instead")


def fetch_team_injuries(team_id: str, headers: Dict[str, str]) -> List[Dict]:
    inj_url = f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/teams/{team_id}/injuries"
    resp = requests.get(inj_url, headers=headers, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    items = payload.get("items") or []
    records: List[Dict] = []
    for ref in items:
        href = ref.get("$ref") or ""
        if not href:
            continue
        try:
            detail = requests.get(href, headers=headers, timeout=10).json()
        except Exception:
            continue
        status = detail.get("status") or ""
        if not status:
            status = (detail.get("type") or {}).get("description") or ""
        date_str = detail.get("date") or ""
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")).astimezone(tz=datetime.utcnow().astimezone().tzinfo)
        except Exception:
            continue
        # Skip stale injuries older than 7 days
        try:
            if dt < datetime.now(tz=dt.tzinfo) - timedelta(days=7):
                continue
        except Exception:
            pass
        athlete = detail.get("athlete") or {}
        name = ""
        if isinstance(athlete, dict) and athlete.get("$ref"):
            try:
                ath_det = requests.get(athlete["$ref"], headers=headers, timeout=10).json()
                name = ath_det.get("displayName") or ath_det.get("fullName") or ""
            except Exception:
                name = ""
        if status:
            records.append({"player": name, "status": status})
    return records


def fetch_injuries_for_slate(team_map: Dict[str, str]) -> List[Dict]:
    """Fetch injuries per team using ESPN team injury endpoints."""
    headers = {"User-Agent": "Mozilla/5.0 (Aeternus-v2 injury fetcher)"}
    records: List[Dict] = []
    for tid, code in team_map.items():
        try:
            statuses = fetch_team_injuries(tid, headers)
        except Exception as exc:
            print(f"[warn] failed to fetch injuries for team_id {tid} ({code}): {exc}")
            continue
        for rec in statuses:
            records.append({"team": code, "status": rec.get("status"), "player": rec.get("player")})
    return records


def status_to_delta(status: str) -> float:
    s = (status or "").strip().lower()
    first = s.split()[0] if s else ""
    return STATUS_DELTA.get(first, -1.5)


def apply_record(data: Dict, target_date: str, team_code: str, status: str, halt: bool) -> None:
    code = team_mapper.normalize_team_code(team_code, "nba") or team_code.upper()
    delta = status_to_delta(status)
    data.setdefault("NBA", {})
    data["NBA"].setdefault(target_date, {})
    existing = data["NBA"][target_date].get(code)
    if existing is None:
        data["NBA"][target_date][code] = delta
    else:
        data["NBA"][target_date][code] = min(existing, delta)
    if halt and delta <= -2.0:
        sentinel.halt_game(code, f"injury status {status}")
    print(f"[info] injury delta for {code} on {target_date}: {delta} (status={status})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape NBA injuries (ESPN) and update injury_adjustments.json")
    ap.add_argument("--date", help="Game date YYYY-MM-DD (default: today)")
    ap.add_argument("--halt", action="store_true", help="Halt teams with negative news")
    args = ap.parse_args()

    target_date = args.date or datetime.utcnow().date().isoformat()

    data = load_injuries()
    data.setdefault("NBA", {})[target_date] = {}
    raw_inj: Dict[str, Dict] = {}
    applied = 0

    team_map = fetch_matchup_teams(target_date)
    if not team_map:
        print(f"[warn] no matchups found for {target_date}; ensure the date is correct (YYYY-MM-DD)")
        return

    try:
        records = fetch_injuries_for_slate(team_map)
        for rec in records:
            status = rec.get("status") or ""
            team = rec.get("team") or ""
            player = rec.get("player") or ""
            if not status or not team:
                continue
            apply_record(data, target_date, team, status, args.halt)
            applied += 1
            if player:
                raw_inj.setdefault("NBA", {}).setdefault(target_date, {}).setdefault(team, []).append(
                    {"player": player, "status": status}
                )
    except Exception as exc:
        print(f"[error] failed to fetch injuries from ESPN: {exc}")
        print("[hint] use news_watcher with a manual text file or pipe injury lines into stdin.")
        return

    todays = data.get("NBA", {}).get(target_date, {})
    total = len(todays)
    neg_count = sum(1 for v in todays.values() if v < 0)
    if total == 0:
        print("[warn] ESPN injury fetch produced no per-team deltas; injuries not applied.")
        return
    if neg_count > total * 0.5:
        print(f"[error] injury deltas look unrealistic ({neg_count} of {total} teams negative); ignoring this fetch and keeping previous file.")
        return

    save_injuries(data)
    if raw_inj:
        save_raw_injuries(raw_inj)
    print(f"[info] injury adjustments written to {INJURY_PATH} (rows applied: {applied})")


if __name__ == "__main__":
    main()
