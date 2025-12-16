#!/usr/bin/env python
"""
Refresh slate-level injuries and team-filtered news from ESPN (public endpoints).

This is intended to remove the manual "go hunt news/injuries" step. It:
  1) Pulls the league scoreboard for the target date to discover the slate.
  2) Pulls per-game summaries to extract per-team injury lists.
  3) Pulls team-filtered league news for each team on the slate.
  4) Writes:
     - chimera_v2c/data/raw_injuries.json   (per league/date/team, player-level)
     - chimera_v2c/data/raw_news.json      (per league/date/team, headline-level)
     - chimera_v2c/data/news_<date>_<league>.txt (digest for optional LLM weighting)
     - reports/llm_packets/<league>/<YYYYMMDD>/news.txt (same digest, co-located with packets)
  5) Ensures chimera_v2c/data/injury_adjustments.json has a per-team entry for the slate date
     (fills missing teams with 0.0; never overwrites existing non-zero deltas).

Usage:
  PYTHONPATH=. python chimera_v2c/tools/refresh_slate_updates.py --league nba --date 2025-12-13
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests

from chimera_v2c.lib import espn_schedule
from chimera_v2c.lib.team_mapper import normalize_team_code


ESPN_SITE_BASE = "https://site.api.espn.com/apis/site/v2/sports"
DATA_DIR = Path("chimera_v2c/data")
INJURY_ADJ_PATH = DATA_DIR / "injury_adjustments.json"
RAW_INJURIES_PATH = DATA_DIR / "raw_injuries.json"
RAW_NEWS_PATH = DATA_DIR / "raw_news.json"
PACKETS_DIR = Path("reports/llm_packets")

# Match espn_schedule.LEAGUE_PATHS keys and ESPN site paths.
LEAGUE_PATHS = espn_schedule.LEAGUE_PATHS


@dataclass(frozen=True)
class SlateTeam:
    team_id: str
    code: str
    name: str


@dataclass(frozen=True)
class SlateGame:
    event_id: str
    away: str
    home: str


def _iso_date(date_str: str) -> str:
    try:
        return dt.date.fromisoformat(str(date_str).strip()).isoformat()
    except ValueError as exc:
        raise SystemExit(f"[error] invalid --date (expected YYYY-MM-DD): {date_str}") from exc


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_get_json(url: str, *, params: Optional[Dict[str, str]] = None, timeout: int = 20) -> Dict[str, Any]:
    headers = {"User-Agent": "Mozilla/5.0 (Chimera-v2c refresh_slate_updates)"}
    resp = requests.get(url, params=params, timeout=timeout, headers=headers)
    resp.raise_for_status()
    return resp.json()


def _normalize_team(abbr: str, league: str) -> str:
    norm = normalize_team_code(abbr, league.lower())
    return (norm or abbr or "").strip().upper()


def _extract_slate(scoreboard: Dict[str, Any], league: str) -> Tuple[List[SlateTeam], List[SlateGame]]:
    teams: Dict[str, SlateTeam] = {}
    games: List[SlateGame] = []
    for event in scoreboard.get("events", []) or []:
        event_id = str(event.get("id") or "").strip()
        competitions = event.get("competitions") or []
        if not event_id or not competitions:
            continue
        comp = competitions[0]
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_team = home.get("team") or {}
        away_team = away.get("team") or {}
        home_id = str(home_team.get("id") or "").strip()
        away_id = str(away_team.get("id") or "").strip()
        home_abbr = _normalize_team(home_team.get("abbreviation") or home_team.get("shortDisplayName") or "", league)
        away_abbr = _normalize_team(away_team.get("abbreviation") or away_team.get("shortDisplayName") or "", league)
        if not home_id or not away_id or not home_abbr or not away_abbr:
            continue
        teams.setdefault(
            home_abbr,
            SlateTeam(team_id=home_id, code=home_abbr, name=str(home_team.get("displayName") or home_abbr)),
        )
        teams.setdefault(
            away_abbr,
            SlateTeam(team_id=away_id, code=away_abbr, name=str(away_team.get("displayName") or away_abbr)),
        )
        games.append(SlateGame(event_id=event_id, away=away_abbr, home=home_abbr))
    return sorted(teams.values(), key=lambda t: t.code), games


def _summary_url(league: str, event_id: str) -> str:
    league_path = LEAGUE_PATHS.get(league.lower())
    if not league_path:
        raise SystemExit(f"[error] unsupported league: {league}")
    return f"{ESPN_SITE_BASE}/{league_path}/summary"


def _news_url(league: str) -> str:
    league_path = LEAGUE_PATHS.get(league.lower())
    if not league_path:
        raise SystemExit(f"[error] unsupported league: {league}")
    return f"{ESPN_SITE_BASE}/{league_path}/news"


def fetch_injuries_for_slate(league: str, games: Iterable[SlateGame]) -> Dict[str, List[Dict[str, str]]]:
    injuries_by_team: Dict[str, List[Dict[str, str]]] = {}
    for g in games:
        url = _summary_url(league, g.event_id)
        payload = _safe_get_json(url, params={"event": g.event_id})
        for team_block in payload.get("injuries") or []:
            team = team_block.get("team") or {}
            abbr = _normalize_team(team.get("abbreviation") or team.get("shortDisplayName") or "", league)
            if not abbr:
                continue
            for inj in team_block.get("injuries") or []:
                athlete = inj.get("athlete") or {}
                position = (athlete.get("position") or {}).get("abbreviation") or ""
                injuries_by_team.setdefault(abbr, []).append(
                    {
                        "player": str(athlete.get("displayName") or athlete.get("fullName") or "").strip(),
                        "position": str(position).strip(),
                        "status": str(inj.get("status") or "").strip(),
                        "type": str((inj.get("type") or {}).get("description") or "").strip(),
                        "details": str(inj.get("details") or "").strip(),
                        "updated": str(inj.get("date") or "").strip(),
                    }
                )
    # Stable ordering for diffs
    for team, rows in injuries_by_team.items():
        rows.sort(key=lambda r: (r.get("status", ""), r.get("player", "")))
    return injuries_by_team


def fetch_team_news(
    league: str,
    teams: Iterable[SlateTeam],
    *,
    limit_per_team: int,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Fetch ESPN team-filtered news via the league news endpoint with `team=<id>`.
    """
    out: Dict[str, List[Dict[str, str]]] = {}
    base = _news_url(league)
    for t in teams:
        try:
            payload = _safe_get_json(base, params={"team": t.team_id, "limit": str(limit_per_team)})
        except Exception as exc:
            print(f"[warn] news fetch failed for {league.upper()} {t.code} (team_id={t.team_id}): {exc}")
            continue
        articles = payload.get("articles") or []
        rows: List[Dict[str, str]] = []
        for a in articles:
            links = a.get("links") or {}
            web = links.get("web") or {}
            href = web.get("href") or ""
            rows.append(
                {
                    "id": str(a.get("id") or "").strip(),
                    "published": str(a.get("published") or "").strip(),
                    "headline": str(a.get("headline") or "").strip(),
                    "description": str(a.get("description") or "").strip(),
                    "url": str(href).strip(),
                }
            )
        out[t.code] = rows
    return out


def ensure_injury_adjustments_stub(league: str, date: str, teams: Iterable[SlateTeam]) -> bool:
    """
    Ensure the injury_adjustments.json file contains a dict for league/date and
    at least a 0.0 entry for each slate team (do not overwrite existing values).
    Returns True if the file content changed.
    """
    league_key = league.upper()
    data = _load_json(INJURY_ADJ_PATH)
    if not isinstance(data, dict):
        data = {}
    before = json.dumps(data, sort_keys=True)
    data.setdefault(league_key, {})
    data[league_key].setdefault(date, {})
    if not isinstance(data[league_key][date], dict):
        data[league_key][date] = {}
    for t in teams:
        if t.code not in data[league_key][date]:
            data[league_key][date][t.code] = 0.0
    after = json.dumps(data, sort_keys=True)
    if before != after:
        _write_json(INJURY_ADJ_PATH, data)
        return True
    # Still touch the file so preflight freshness reflects that we refreshed the slate.
    if INJURY_ADJ_PATH.exists():
        try:
            INJURY_ADJ_PATH.touch()
        except Exception:
            pass
    return False


def write_digest(
    *,
    league: str,
    date: str,
    games: List[SlateGame],
    injuries_by_team: Dict[str, List[Dict[str, str]]],
    news_by_team: Dict[str, List[Dict[str, str]]],
) -> str:
    lines: List[str] = []
    lines.append(f"{date} {league.upper()} — ESPN Slate Updates")
    lines.append("")
    lines.append("Games:")
    for g in games:
        lines.append(f"- {g.away}@{g.home}")
    lines.append("")
    lines.append("Injuries:")
    any_inj = False
    for team in sorted(injuries_by_team.keys()):
        rows = injuries_by_team.get(team) or []
        if not rows:
            continue
        any_inj = True
        lines.append(f"- {team}:")
        for r in rows:
            status = r.get("status") or ""
            player = r.get("player") or ""
            details = r.get("details") or ""
            pos = r.get("position") or ""
            tail = " ".join(p for p in [pos, details] if p).strip()
            if tail:
                lines.append(f"  - {status}: {player} ({tail})")
            else:
                lines.append(f"  - {status}: {player}")
    if not any_inj:
        lines.append("- (none found via ESPN)")
    lines.append("")
    lines.append("News (team-filtered):")
    any_news = False
    for team in sorted(news_by_team.keys()):
        rows = news_by_team.get(team) or []
        if not rows:
            continue
        any_news = True
        lines.append(f"- {team}:")
        for r in rows:
            headline = r.get("headline") or ""
            published = r.get("published") or ""
            url = r.get("url") or ""
            suffix = " ".join(s for s in [published, url] if s).strip()
            if suffix:
                lines.append(f"  - {headline} ({suffix})")
            else:
                lines.append(f"  - {headline}")
    if not any_news:
        lines.append("- (none fetched)")
    lines.append("")
    return "\n".join(lines)


def refresh_slate_updates(
    *,
    league: str,
    date: str,
    limit_per_team: int = 10,
    write_packets: bool = True,
) -> None:
    league_norm = league.lower().strip()
    date_iso = _iso_date(date)
    if league_norm not in LEAGUE_PATHS:
        raise SystemExit(f"[error] unsupported league: {league}")

    scoreboard = espn_schedule.get_scoreboard(league_norm, dt.date.fromisoformat(date_iso))
    teams, games = _extract_slate(scoreboard, league_norm)
    if not games:
        raise SystemExit(f"[error] no games found on ESPN scoreboard for {league_norm.upper()} {date_iso}")

    injuries_by_team = fetch_injuries_for_slate(league_norm, games)
    news_by_team = fetch_team_news(league_norm, teams, limit_per_team=limit_per_team)

    # Persist raw injuries (merge by league/date).
    raw_inj = _load_json(RAW_INJURIES_PATH)
    raw_inj.setdefault(league_norm.upper(), {})
    raw_inj[league_norm.upper()][date_iso] = injuries_by_team
    _write_json(RAW_INJURIES_PATH, raw_inj)

    # Persist raw news (merge by league/date).
    raw_news = _load_json(RAW_NEWS_PATH)
    raw_news.setdefault(league_norm.upper(), {})
    raw_news[league_norm.upper()][date_iso] = news_by_team
    _write_json(RAW_NEWS_PATH, raw_news)

    # Ensure injury_adjustments has stubs for the slate (no overwrites).
    ensure_injury_adjustments_stub(league_norm, date_iso, teams)

    digest = write_digest(
        league=league_norm,
        date=date_iso,
        games=games,
        injuries_by_team=injuries_by_team,
        news_by_team=news_by_team,
    )
    # Write digest into data/ for operators + llm_packets/ for specialist upload.
    digest_path = DATA_DIR / f"news_{date_iso}_{league_norm}.txt"
    digest_path.write_text(digest, encoding="utf-8")
    if write_packets:
        ymd = date_iso.replace("-", "")
        out_dir = PACKETS_DIR / league_norm / ymd
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "news.txt").write_text(digest, encoding="utf-8")

    print(f"[ok] refreshed {league_norm.upper()} slate updates for {date_iso}")
    print(f"[ok] raw injuries -> {RAW_INJURIES_PATH}")
    print(f"[ok] raw news     -> {RAW_NEWS_PATH}")
    print(f"[ok] digest       -> {digest_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh slate injuries + news from ESPN (public endpoints).")
    ap.add_argument("--league", required=True, help="League (nba|nhl|nfl)")
    ap.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    ap.add_argument("--limit-per-team", type=int, default=10, help="News article limit per team (default: 10)")
    ap.add_argument(
        "--no-packets",
        action="store_true",
        help="Do not write reports/llm_packets/<league>/<YYYYMMDD>/news.txt (default: write).",
    )
    args = ap.parse_args()

    refresh_slate_updates(
        league=args.league,
        date=args.date,
        limit_per_team=int(args.limit_per_team),
        write_packets=not args.no_packets,
    )


if __name__ == "__main__":
    main()
