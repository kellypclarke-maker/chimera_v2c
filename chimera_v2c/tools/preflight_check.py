"""
Preflight guardrails before running v2c planning/execution.

Checks:
- Injury file freshness (`chimera_v2c/data/injury_adjustments.json`) is newer than a threshold.
- Earliest game start for the target league/date is within 30 minutes (warn/exit if too early).

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/preflight_check.py --league nba --date 2025-12-08
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from chimera_v2c.lib import nhl_scoreboard
import json


INJURY_PATH = Path("chimera_v2c/data/injury_adjustments.json")
MANIFEST_PATH = Path("chimera_v2c/data_manifest.json")
DATA_FILES = {
    "team_ratings": Path("chimera_v2c/data/team_ratings.json"),
    "team_four_factors": Path("chimera_v2c/data/team_four_factors.json"),
    "ff_model": Path("chimera_v2c/data/ff_model.json"),
}
LEAGUE_FETCHERS: Dict[str, Callable[[str], Dict]] = {
    "nba": nhl_scoreboard.fetch_nba_scoreboard,
    "nhl": nhl_scoreboard.fetch_nhl_scoreboard,
    "nfl": nhl_scoreboard.fetch_nfl_scoreboard,
}


def parse_iso_z(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def check_injury_freshness(threshold_hours: int, league: str) -> None:
    if not INJURY_PATH.exists():
        raise SystemExit(f"[error] injury file missing: {INJURY_PATH}")
    mtime = datetime.fromtimestamp(INJURY_PATH.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
    if age_hours > threshold_hours:
        raise SystemExit(
            f"[error] injury file is stale ({age_hours:.1f}h old). Refresh injuries before running."
        )
    print(f"[ok] injury file fresh ({age_hours:.1f}h old) -> {INJURY_PATH}")
    # Require per-league entry for the target date when running NHL to avoid empty injury passes.
    if league.lower() == "nhl":
        try:
            data = json.loads(INJURY_PATH.read_text(encoding="utf-8"))
        except Exception:
            raise SystemExit(f"[error] injury file unreadable: {INJURY_PATH}")
        if not isinstance(data, dict) or "NHL" not in data:
            raise SystemExit(f"[error] injury file missing NHL section for {INJURY_PATH}")


def check_data_freshness(threshold_hours: int) -> None:
    manifest = DATA_FILES.copy()
    if MANIFEST_PATH.exists():
        try:
            override = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if isinstance(override, dict):
                manifest = {k: Path(v) for k, v in override.items()}
        except Exception:
            pass
    stale = []
    for label, path in manifest.items():
        if not path.exists():
            stale.append(f"{label} missing ({path})")
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
        if age_hours > threshold_hours:
            stale.append(f"{label} stale ({age_hours:.1f}h old)")
    if stale:
        raise SystemExit("[error] data freshness failed: " + "; ".join(stale))
    print(f"[ok] data files fresh (<= {threshold_hours}h): " + ", ".join(manifest.keys()))


def check_start_windows(
    league: str,
    date: str,
    min_window_minutes: int,
) -> None:
    fetcher = LEAGUE_FETCHERS.get(league)
    if not fetcher:
        raise SystemExit(f"[error] unsupported league: {league}")
    sb = fetcher(date)
    if sb.get("status") != "ok":
        raise SystemExit(f"[error] scoreboard fetch failed: {sb.get('message')}")

    games: List[Dict] = sb.get("games") or []
    if not games:
        print(f"[warn] no games found for {league} {date}")
        return

    now = datetime.now(timezone.utc)
    starts: List[Tuple[str, datetime]] = []
    for g in games:
        start_str = g.get("start_time")
        if not start_str:
            continue
        game_id = g.get("game_id") or g.get("id") or "unknown"
        try:
            start = parse_iso_z(start_str)
        except Exception:
            continue
        starts.append((str(game_id), start))

    if not starts:
        print(f"[warn] no start times found for {league} {date}")
        return

    earliest_id, earliest_start = min(starts, key=lambda x: x[1])
    delta_minutes = (earliest_start - now).total_seconds() / 60
    if delta_minutes > min_window_minutes:
        raise SystemExit(
            f"[error] too early for planning; rerun within {min_window_minutes} minutes of slate start "
            f"(earliest game {earliest_id} starts in {delta_minutes:.1f} minutes at {earliest_start.isoformat()})."
        )

    print(f"[ok] slate start window ok: earliest {league} game at {earliest_start.isoformat()} (in {delta_minutes:.1f} minutes)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preflight checks for v2c (injury freshness + time-to-start).")
    ap.add_argument("--league", default="nba", help="League (nba|nhl|nfl). Default: nba.")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD.")
    ap.add_argument(
        "--injury-max-age-hours",
        type=int,
        default=12,
        help="Maximum allowed injury file age in hours (default: 12).",
    )
    ap.add_argument(
        "--data-max-age-hours",
        type=int,
        default=24 * 7,
        help="Maximum allowed data file age in hours (ratings/factors/model). Default: 168h (7 days).",
    )
    ap.add_argument(
        "--min-window-minutes",
        type=int,
        default=30,
        help="Require being within this many minutes of game start (default: 30).",
    )
    args = ap.parse_args()

    check_injury_freshness(args.injury_max_age_hours, args.league.lower())
    check_data_freshness(args.data_max_age_hours)
    check_start_windows(args.league.lower(), args.date, args.min_window_minutes)


if __name__ == "__main__":
    main()
