"""
Build structured, slate-level data packets for LLM directives.

This tool is intended to reduce hallucinations by exporting small, targeted
CSV files with ground-truth context for a given date/league. Typical usage:

  PYTHONPATH=. python chimera_v2c/tools/build_llm_packets.py \
    --league nba --date 2025-12-11

Outputs (by default):
- reports/llm_packets/<league>/<YYYYMMDD>/standings_form_<date>_<league>.csv
- reports/llm_packets/<league>/<YYYYMMDD>/injuries_<date>_<league>.csv
- reports/llm_packets/<league>/<YYYYMMDD>/schedule_fatigue_<date>_<league>.csv
- reports/llm_packets/<league>/<YYYYMMDD>/odds_markets_<date>_<league>.csv

These files are designed to be uploaded alongside a specialist directive so
the LLM can reason from tables instead of fabricating records/injuries/odds.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from chimera_v2c.lib import nhl_scoreboard, team_mapper

SCOREBOARD_BASE = nhl_scoreboard.SCOREBOARD_BASE
LEAGUE_PATHS = nhl_scoreboard.LEAGUE_PATHS


@dataclass
class GameInfo:
    matchup: str  # AWAY@HOME (normalized codes)
    league: str
    date: str  # YYYY-MM-DD
    home_abbr: str  # ESPN abbreviation
    away_abbr: str


def _fetch_scoreboard_raw(league: str, date_str: str) -> Dict[str, Any]:
    league = league.lower()
    path = LEAGUE_PATHS.get(league)
    if not path:
        raise SystemExit(f"[error] unsupported league for scoreboard: {league}")
    date_token = date_str.replace("-", "")
    url = f"{SCOREBOARD_BASE}/{path}/scoreboard?dates={date_token}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def parse_games_for_date(league: str, date_str: str) -> Tuple[List[GameInfo], Dict[str, Dict[str, Any]]]:
    """
    Return list of GameInfo and a mapping from game_id -> raw competition dict
    to reuse for standings/records.
    """
    payload = _fetch_scoreboard_raw(league, date_str)
    events = payload.get("events") or []
    games: List[GameInfo] = []
    comp_by_game_id: Dict[str, Dict[str, Any]] = {}

    for ev in events:
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors") or []
        home_abbr = None
        away_abbr = None
        for c in competitors:
            team = c.get("team") or {}
            abbr = team.get("abbreviation") or team.get("shortDisplayName") or team.get("displayName")
            if not abbr:
                continue
            if c.get("homeAway") == "home":
                home_abbr = abbr
            elif c.get("homeAway") == "away":
                away_abbr = abbr
        if not home_abbr or not away_abbr:
            continue
        home = team_mapper.normalize_team_code(home_abbr, league) or home_abbr.upper()
        away = team_mapper.normalize_team_code(away_abbr, league) or away_abbr.upper()
        matchup = f"{away}@{home}"
        games.append(
            GameInfo(
                matchup=matchup,
                league=league,
                date=date_str,
                home_abbr=home_abbr,
                away_abbr=away_abbr,
            )
        )
        game_id = ev.get("id") or comp.get("id")
        if game_id:
            comp_by_game_id[str(game_id)] = comp
    return games, comp_by_game_id


def _parse_record_summary(summary: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse strings like '15-6' into (15, 6).
    """
    if not summary:
        return None, None
    parts = summary.split("-")
    if len(parts) != 2:
        return None, None
    try:
        w = int(parts[0])
        l = int(parts[1])
        return w, l
    except Exception:
        return None, None


def compute_form_metrics(
    league: str,
    date_str: str,
    max_back_days: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute last-10 record and current streak for each team using ESPN scoreboards.

    Returns mapping:
      abbr -> {
        "last10_w": int,
        "last10_l": int,
        "streak_len": int,
        "streak_type": "W" or "L",
      }
    """
    target = datetime.fromisoformat(date_str).date()
    history: Dict[str, List[int]] = {}

    for delta in range(1, max_back_days + 1):
        day = target - timedelta(days=delta)
        payload = _fetch_scoreboard_raw(league, day.isoformat())
        events = payload.get("events") or []
        for ev in events:
            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors") or []
            home_abbr = away_abbr = None
            home_score = away_score = None
            for c in competitors:
                team = c.get("team") or {}
                abbr = team.get("abbreviation") or team.get("shortDisplayName") or team.get("displayName")
                if c.get("homeAway") == "home":
                    home_abbr = abbr
                    home_score = c.get("score")
                elif c.get("homeAway") == "away":
                    away_abbr = abbr
                    away_score = c.get("score")
            if home_abbr is None or away_abbr is None:
                continue
            try:
                hs = int(home_score) if home_score is not None else None
                as_ = int(away_score) if away_score is not None else None
            except Exception:
                hs = as_ = None
            if hs is None or as_ is None:
                # Skip games without final scores
                continue
            # Determine win/loss from each team's perspective; treat this day as newer than later deltas.
            for abbr, score, opp_score in (
                (home_abbr, hs, as_),
                (away_abbr, as_, hs),
            ):
                if abbr is None:
                    continue
                is_win = 1 if score > opp_score else 0
                hist = history.setdefault(abbr, [])
                # Append only until we have 10 games; keep list ordered from most recent to oldest.
                if len(hist) < 10:
                    hist.append(is_win)

        # Early exit if all teams we care about reach 10 games is possible,
        # but we don't know the slate teams here; keep simple for now.

    metrics: Dict[str, Dict[str, Any]] = {}
    for abbr, results in history.items():
        if not results:
            continue
        last10 = results[:10]
        w10 = sum(last10)
        l10 = len(last10) - w10
        # Current streak: count consecutive wins or losses from most recent game (index 0)
        first = last10[0]
        streak_len = 1
        for r in last10[1:]:
            if r == first:
                streak_len += 1
            else:
                break
        streak_type = "W" if first == 1 else "L"
        metrics[abbr] = {
            "last10_w": w10,
            "last10_l": l10,
            "streak_len": streak_len,
            "streak_type": streak_type,
        }
    return metrics


def build_standings_form_packet(
    league: str,
    date_str: str,
    games: List[GameInfo],
    comp_by_game_id: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    One row per (game, team) with overall and home/road records from ESPN.
    """
    rows: List[Dict[str, Any]] = []
    form_metrics = compute_form_metrics(league, date_str)
    payload = _fetch_scoreboard_raw(league, date_str)
    events = payload.get("events") or []
    # Map (away_abbr, home_abbr) to competition blob for convenience
    comp_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ev in events:
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors") or []
        home_abbr = None
        away_abbr = None
        for c in competitors:
            team = c.get("team") or {}
            abbr = team.get("abbreviation") or team.get("shortDisplayName") or team.get("displayName")
            if not abbr:
                continue
            if c.get("homeAway") == "home":
                home_abbr = abbr
            elif c.get("homeAway") == "away":
                away_abbr = abbr
        if home_abbr and away_abbr:
            comp_by_pair[(away_abbr, home_abbr)] = comp

    for g in games:
        comp = comp_by_pair.get((g.away_abbr, g.home_abbr))
        if not comp:
            continue
        competitors = comp.get("competitors") or []
        for c in competitors:
            team = c.get("team") or {}
            abbr = team.get("abbreviation") or team.get("shortDisplayName") or team.get("displayName")
            if not abbr:
                continue
            is_home = c.get("homeAway") == "home"
            code = team_mapper.normalize_team_code(abbr, league) or abbr.upper()
            fm = form_metrics.get(abbr, {})
            last10_w = fm.get("last10_w")
            last10_l = fm.get("last10_l")
            streak_len = fm.get("streak_len")
            streak_type = fm.get("streak_type")
            recs = c.get("records") or []
            overall_w = overall_l = None
            home_w = home_l = None
            road_w = road_l = None
            for r in recs:
                name = (r.get("name") or "").lower()
                summary = r.get("summary") or ""
                w, l = _parse_record_summary(summary)
                if name == "overall":
                    overall_w, overall_l = w, l
                elif name == "home":
                    home_w, home_l = w, l
                elif name == "road":
                    road_w, road_l = w, l
            rows.append(
                {
                    "date": date_str,
                    "league": league,
                    "matchup": g.matchup,
                    "team": code,
                    "opponent": g.home_abbr if not is_home else g.away_abbr,
                    "is_home": "1" if is_home else "0",
                    "overall_w": overall_w,
                    "overall_l": overall_l,
                    "home_w": home_w,
                    "home_l": home_l,
                    "road_w": road_w,
                    "road_l": road_l,
                    "last10_w": last10_w,
                    "last10_l": last10_l,
                    "streak_len": streak_len,
                    "streak_type": streak_type,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "league",
                "matchup",
                "team",
                "opponent",
                "is_home",
                "overall_w",
                "overall_l",
                "home_w",
                "home_l",
                "road_w",
                "road_l",
                "last10_w",
                "last10_l",
                "streak_len",
                "streak_type",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote standings/form packet -> {out_path}")


def find_last_game_for_team(league: str, team_abbr: str, target_date: str, max_days: int = 7) -> Tuple[Optional[str], Optional[int]]:
    """
    Look back up to max_days using ESPN scoreboard to find the last game date.
    Returns (YYYY-MM-DD, rest_days) or (None, None) if not found.
    """
    target = datetime.fromisoformat(target_date).date()
    for delta in range(1, max_days + 1):
        prev = target - timedelta(days=delta)
        payload = _fetch_scoreboard_raw(league, prev.isoformat())
        events = payload.get("events") or []
        for ev in events:
            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors") or []
            for c in competitors:
                team = c.get("team") or {}
                abbr = team.get("abbreviation") or team.get("shortDisplayName") or team.get("displayName")
                if abbr == team_abbr:
                    return prev.isoformat(), delta
    return None, None


def build_schedule_fatigue_packet(
    league: str,
    date_str: str,
    games: List[GameInfo],
    out_path: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    for g in games:
        # Use raw ESPN abbreviations for rest computation
        home_last_date, home_rest = find_last_game_for_team(league, g.home_abbr, date_str)
        away_last_date, away_rest = find_last_game_for_team(league, g.away_abbr, date_str)
        home_code = team_mapper.normalize_team_code(g.home_abbr, league) or g.home_abbr.upper()
        away_code = team_mapper.normalize_team_code(g.away_abbr, league) or g.away_abbr.upper()
        rows.append(
            {
                "date": date_str,
                "league": league,
                "matchup": g.matchup,
                "home_team": home_code,
                "away_team": away_code,
                "home_last_game_date": home_last_date or "",
                "home_days_rest": home_rest if home_rest is not None else "",
                "home_is_b2b": "1" if home_rest == 1 else "0",
                "away_last_game_date": away_last_date or "",
                "away_days_rest": away_rest if away_rest is not None else "",
                "away_is_b2b": "1" if away_rest == 1 else "0",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "league",
                "matchup",
                "home_team",
                "away_team",
                "home_last_game_date",
                "home_days_rest",
                "home_is_b2b",
                "away_last_game_date",
                "away_days_rest",
                "away_is_b2b",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote schedule/fatigue packet -> {out_path}")


def build_injuries_packet(
    league: str,
    date_str: str,
    games: List[GameInfo],
    raw_injury_path: Path,
    injury_adjustments_path: Path,
    out_path: Path,
) -> None:
    league_key = league.upper()
    try:
        raw_data = {}
        if raw_injury_path.exists():
            raw_data = json.loads(raw_injury_path.read_text(encoding="utf-8"))  # type: ignore[name-defined]
    except Exception:
        raw_data = {}
    try:
        adj_data = {}
        if injury_adjustments_path.exists():
            adj_data = json.loads(injury_adjustments_path.read_text(encoding="utf-8"))  # type: ignore[name-defined]
    except Exception:
        adj_data = {}

    raw_by_team = (raw_data.get(league_key) or {}).get(date_str) or {}
    adj_by_team = (adj_data.get(league_key) or {}).get(date_str) or {}

    # Restrict to teams in today's slate to keep packets small.
    teams_in_slate: set[str] = set()
    for g in games:
        home_code = team_mapper.normalize_team_code(g.home_abbr, league) or g.home_abbr.upper()
        away_code = team_mapper.normalize_team_code(g.away_abbr, league) or g.away_abbr.upper()
        teams_in_slate.add(home_code)
        teams_in_slate.add(away_code)

    # Compute team-level deltas, preferring injury_adjustments, falling back to simple
    # status-based heuristics when adjustments are missing or zero.
    STATUS_IMPACT = {
        "out": -5.0,
        "doubtful": -3.0,
        "questionable": -2.0,
        "day-to-day": -1.0,
        "injured": -4.0,
        "injured reserve": -4.0,
        "active": 0.0,
    }

    team_delta_map: Dict[str, Optional[float]] = {}
    for g in games:
        for abbr in (g.home_abbr, g.away_abbr):
            norm_team = team_mapper.normalize_team_code(abbr, league) or abbr.upper()
            if norm_team not in teams_in_slate:
                continue
            if norm_team in team_delta_map:
                continue
            # Prefer explicit adjustment if non-zero
            delta = adj_by_team.get(norm_team)
            if delta is None or delta == 0:
                # Derive from raw injury statuses if available for this team
                per_player_deltas: List[float] = []
                for team_code, entries in raw_by_team.items():
                    norm_code = team_mapper.normalize_team_code(team_code, league) or team_code.upper()
                    if norm_code != norm_team:
                        continue
                    for rec in entries or []:
                        status = (rec.get("status") or "").strip().lower()
                        key = status.split()[0] if status else ""
                        per_player_deltas.append(STATUS_IMPACT.get(key, 0.0))
                if per_player_deltas:
                    delta = min(per_player_deltas)
            team_delta_map[norm_team] = delta

    rows: List[Dict[str, Any]] = []
    # Player-level entries from raw_injuries
    for team_code, entries in raw_by_team.items():
        norm_team = team_mapper.normalize_team_code(team_code, league) or team_code.upper()
        if norm_team not in teams_in_slate:
            continue
        delta = team_delta_map.get(norm_team)
        for rec in entries or []:
            rows.append(
                {
                    "date": date_str,
                    "league": league,
                    "team": norm_team,
                    "player": rec.get("player") or "",
                    "status": rec.get("status") or "",
                    "team_delta": delta if delta is not None else "",
                }
            )

    # Team-only deltas where no player entries exist
    for team_code, adj_delta in adj_by_team.items():
        norm_team = team_mapper.normalize_team_code(team_code, league) or team_code.upper()
        if norm_team not in teams_in_slate:
            continue
        has_row = any(r["team"] == norm_team for r in rows)
        if not has_row:
            delta = team_delta_map.get(norm_team, adj_delta)
            rows.append(
                {
                    "date": date_str,
                    "league": league,
                    "team": norm_team,
                    "player": "",
                    "status": "",
                    "team_delta": delta if delta is not None else "",
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "league", "team", "player", "status", "team_delta"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote injuries packet -> {out_path}")


def build_odds_markets_packet(
    league: str,
    date_str: str,
    games: List[GameInfo],
    daily_ledger_dir: Path,
    out_path: Path,
) -> None:
    """
    Read home-implied Kalshi mids from the daily ledger for the given date.
    This assumes the per-day ledger has already been created and filled.
    """
    ledger_path = daily_ledger_dir / f"{date_str.replace('-', '')}_daily_game_ledger.csv"
    if not ledger_path.exists():
        print(f"[warn] daily ledger not found for {date_str}: {ledger_path}")
        return

    # Load ledger into (league, matchup) -> kalshi_mid
    mids: Dict[Tuple[str, str], Optional[float]] = {}
    with ledger_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lg = (row.get("league") or "").strip()
            matchup = (row.get("matchup") or "").strip()
            if not lg or not matchup:
                continue
            mid_str = row.get("kalshi_mid")
            mid_val: Optional[float]
            if mid_str is None or str(mid_str).strip() == "":
                mid_val = None
            else:
                try:
                    mid_val = float(mid_str)
                except Exception:
                    mid_val = None
            mids[(lg, matchup)] = mid_val

    rows: List[Dict[str, Any]] = []
    for g in games:
        key = (league, g.matchup)
        mid = mids.get(key)
        home_code = team_mapper.normalize_team_code(g.home_abbr, league) or g.home_abbr.upper()
        away_code = team_mapper.normalize_team_code(g.away_abbr, league) or g.away_abbr.upper()
        rows.append(
            {
                "date": date_str,
                "league": league,
                "matchup": g.matchup,
                "home_team": home_code,
                "away_team": away_code,
                "kalshi_mid_home": mid if mid is not None else "",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "league", "matchup", "home_team", "away_team", "kalshi_mid_home"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote odds/markets packet -> {out_path}")


def build_h2h_packet(
    league: str,
    date_str: str,
    games: List[GameInfo],
    out_path: Path,
    lookback_days: int = 120,
) -> None:
    """
    Build simple head-to-head summary for each matchup on the slate using recent
    ESPN scores.

    For each upcoming matchup AWAY@HOME, we walk back up to lookback_days and
    collect prior games between those two teams, then compute:
      - away_wins: number of prior games where upcoming-away team outscored upcoming-home.
      - home_wins: number of prior games where upcoming-home team outscored upcoming-away.
      - avg_margin: average (away_score - home_score) over those games.
    """
    target = datetime.fromisoformat(date_str).date()
    pairs = {(g.away_abbr, g.home_abbr) for g in games}
    results: Dict[Tuple[str, str], List[Tuple[int, int]]] = {p: [] for p in pairs}

    for delta in range(1, lookback_days + 1):
        day = target - timedelta(days=delta)
        payload = _fetch_scoreboard_raw(league, day.isoformat())
        events = payload.get("events") or []
        if not events:
            continue
        for ev in events:
            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors") or []
            home_abbr = away_abbr = None
            home_score = away_score = None
            for c in competitors:
                team = c.get("team") or {}
                abbr = team.get("abbreviation") or team.get("shortDisplayName") or team.get("displayName")
                if c.get("homeAway") == "home":
                    home_abbr = abbr
                    home_score = c.get("score")
                elif c.get("homeAway") == "away":
                    away_abbr = abbr
                    away_score = c.get("score")
            if home_abbr is None or away_abbr is None:
                continue
            try:
                hs = int(home_score) if home_score is not None else None
                as_ = int(away_score) if away_score is not None else None
            except Exception:
                hs = as_ = None
            if hs is None or as_ is None:
                continue

            # Check both orderings against our slate pairs
            for pa, ph in pairs:
                if {home_abbr, away_abbr} != {pa, ph}:
                    continue
                # Map scores to upcoming-away / upcoming-home perspective
                if home_abbr == pa:
                    away_score = hs
                    home_score_val = as_
                else:
                    away_score = as_
                    home_score_val = hs
                results[(pa, ph)].append((away_score, home_score_val))

    rows: List[Dict[str, Any]] = []
    for g in games:
        key = (g.away_abbr, g.home_abbr)
        samples = results.get(key) or []
        away_wins = home_wins = 0
        margins: List[float] = []
        for away_score, home_score_val in samples:
            if away_score is None or home_score_val is None:
                continue
            margin = away_score - home_score_val
            margins.append(float(margin))
            if margin > 0:
                away_wins += 1
            elif margin < 0:
                home_wins += 1
        avg_margin = sum(margins) / len(margins) if margins else None

        away_code = team_mapper.normalize_team_code(g.away_abbr, league) or g.away_abbr.upper()
        home_code = team_mapper.normalize_team_code(g.home_abbr, league) or g.home_abbr.upper()
        rows.append(
            {
                "date": date_str,
                "league": league,
                "matchup": f"{away_code}@{home_code}",
                "away_team": away_code,
                "home_team": home_code,
                "away_wins": away_wins,
                "home_wins": home_wins,
                "avg_margin_away_minus_home": avg_margin if avg_margin is not None else "",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "league",
                "matchup",
                "away_team",
                "home_team",
                "away_wins",
                "home_wins",
                "avg_margin_away_minus_home",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote h2h packet -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build LLM data packets (CSV) for a slate.")
    ap.add_argument("--league", default="nba", help="League (nba|nhl|nfl). Default: nba.")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD.")
    ap.add_argument(
        "--out-dir",
        default="reports/llm_packets",
        help="Base output directory for packets (default: reports/llm_packets).",
    )
    ap.add_argument(
        "--no-odds",
        action="store_true",
        help="Skip odds/markets packet generation.",
    )
    args = ap.parse_args()

    league = args.league.lower()
    date_str = args.date

    games, comp_by_game_id = parse_games_for_date(league, date_str)
    if not games:
        print(f"[warn] no {league.upper()} games found for {date_str}; nothing to write.")
        return

    base = Path(args.out_dir) / league / date_str.replace("-", "")
    base.mkdir(parents=True, exist_ok=True)

    # Standings / form
    standings_path = base / f"standings_form_{date_str.replace('-', '')}_{league}.csv"
    build_standings_form_packet(league, date_str, games, comp_by_game_id, standings_path)

    # Schedule / fatigue
    sched_path = base / f"schedule_fatigue_{date_str.replace('-', '')}_{league}.csv"
    build_schedule_fatigue_packet(league, date_str, games, sched_path)

    # Injuries
    raw_inj_path = Path("chimera_v2c/data/raw_injuries.json")
    inj_adj_path = Path("chimera_v2c/data/injury_adjustments.json")
    injuries_path = base / f"injuries_{date_str.replace('-', '')}_{league}.csv"
    build_injuries_packet(league, date_str, games, raw_inj_path, inj_adj_path, injuries_path)

    # Odds / markets (home Kalshi mid)
    if not args.no_odds:
        odds_path = base / f"odds_markets_{date_str.replace('-', '')}_{league}.csv"
        daily_ledger_dir = Path("reports/daily_ledgers")
        build_odds_markets_packet(league, date_str, games, daily_ledger_dir, odds_path)

    # Head-to-head
    h2h_path = base / f"h2h_{date_str.replace('-', '')}_{league}.csv"
    build_h2h_packet(league, date_str, games, h2h_path)


if __name__ == "__main__":
    main()
