"""
ESPN NHL injury scraper (adds deltas to injury_adjustments.json).

Usage:
  PYTHONPATH=. python chimera_v2c/tools/fetch_injuries_nhl.py --date YYYY-MM-DD

Notes:
  - Status → delta is coarse and conservative (Out=-5, Doubtful=-2.5, Questionable/Day-to-day=-1.5, Active/Probable=0).
  - Clamps deltas to [-10, 0] and merges per-team for the target date under league key "NHL".
  - If ESPN fails or returns empty, it prints a warning and exits without modifying the injury file.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

from chimera_v2c.lib import team_mapper

INJURY_PATH = Path("chimera_v2c/data/injury_adjustments.json")
RAW_INJURY_PATH = Path("chimera_v2c/data/raw_injuries.json")
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
ESPN_INJURY_BASE = "https://sports.core.api.espn.com/v2/sports/hockey/nhl/teams"

# Conservative, keeps impacts small unless clearly "Out".
STATUS_DELTA = {
    "out": -5.0,
    "doubtful": -2.5,
    "questionable": -1.5,
    "day-to-day": -1.5,
    "probable": 0.0,
    "active": 0.0,
}


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def status_to_delta(status: str) -> float:
    s = (status or "").strip().lower()
    first = s.split()[0] if s else ""
    return STATUS_DELTA.get(first, -1.5)


def fetch_team_injuries(team_id: str) -> List[Dict]:
    """
    ESPN's core injuries endpoint intermittently returns 404/500 for NHL teams.
    Treat those as "no reported injuries" instead of failing.
    """
    url = f"{ESPN_INJURY_BASE}/{team_id}/injuries"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        # Fallback to empty injuries when ESPN refuses the endpoint.
        status = exc.response.status_code if exc.response is not None else None
        print(f"[info] injuries unavailable for team_id {team_id} (status={status}); assuming no injuries.")
        return []
    payload = resp.json()
    items = payload.get("items") or []
    records: List[Dict] = []
    for ref in items:
        href = ref.get("$ref") or ""
        if not href:
            continue
        try:
            detail = requests.get(href, timeout=10).json()
        except Exception:
            continue
        status = detail.get("status") or ""
        if not status:
            status = (detail.get("type") or {}).get("description") or ""
        date_str = detail.get("date") or ""
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                if dt < datetime.now(dt.tzinfo) - timedelta(days=7):
                    continue
            except Exception:
                pass
        athlete = detail.get("athlete") or {}
        name = ""
        if isinstance(athlete, dict) and athlete.get("$ref"):
            try:
                ath_det = requests.get(athlete["$ref"], timeout=10).json()
                name = ath_det.get("displayName") or ath_det.get("fullName") or ""
            except Exception:
                name = ""
        if status:
            records.append({"status": status, "player": name})
    return records


def fetch_scoreboard(date: str) -> List[Dict]:
    token = date.replace("-", "")
    url = f"{ESPN_SCOREBOARD}?dates={token}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    events = payload.get("events") or []
    games: List[Dict] = []
    for ev in events:
        comps = (ev.get("competitions") or [{}])[0].get("competitors") or []
        for c in comps:
            team = c.get("team") or {}
            tid = team.get("id")
            abbr = team.get("abbreviation") or team.get("shortDisplayName")
            if tid and abbr:
                games.append({"team_id": str(tid), "abbr": abbr})
    return games


def apply_record(store: Dict, target_date: str, team_code: str, status: str) -> None:
    code = team_mapper.normalize_team_code(team_code, "nhl") or team_code.upper()
    delta = max(-10.0, min(0.0, status_to_delta(status)))
    store.setdefault("NHL", {})
    store["NHL"].setdefault(target_date, {})
    existing = store["NHL"][target_date].get(code)
    if existing is None:
        store["NHL"][target_date][code] = delta
    else:
        # keep the more negative impact if multiple records exist
        store["NHL"][target_date][code] = min(existing, delta)
    print(f"[info] NHL injury delta for {code} on {target_date}: {delta} (status={status})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape NHL injuries (ESPN) and update injury_adjustments.json")
    ap.add_argument("--date", help="Game date YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    target_date = args.date or datetime.utcnow().date().isoformat()

    slate = fetch_scoreboard(target_date)
    if not slate:
        print(f"[warn] no NHL games found for {target_date}; nothing to do.")
        return

    data = load_json(INJURY_PATH)
    raw = load_json(RAW_INJURY_PATH)
    applied = 0

    for entry in slate:
        tid = entry.get("team_id")
        abbr = entry.get("abbr")
        if not tid or not abbr:
            continue
        try:
            records = fetch_team_injuries(tid)
        except Exception as exc:
            print(f"[warn] failed to fetch injuries for team_id {tid} ({abbr}): {exc}")
            continue
        for rec in records:
            status = rec.get("status") or ""
            player = rec.get("player") or ""
            if not status:
                continue
            apply_record(data, target_date, abbr, status)
            applied += 1
            if player:
                raw.setdefault("NHL", {}).setdefault(target_date, {}).setdefault(abbr, []).append(
                    {"player": player, "status": status}
                )

    if applied == 0:
        print(f"[warn] no NHL injury records applied for {target_date}")
        return

    save_json(INJURY_PATH, data)
    save_json(RAW_INJURY_PATH, raw)
    print(f"[ok] NHL injuries applied for {target_date}: {applied} records -> {INJURY_PATH}")


if __name__ == "__main__":
    main()
