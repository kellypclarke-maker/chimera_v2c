from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from typing import Dict, Iterable, List, Optional, Tuple

import requests

from chimera_v2c.lib.team_mapper import normalize_team_code


MONEYPUCK_BASE = "https://moneypuck.com"
CURRENT_INJURIES_PATH = "moneypuck/playerData/playerNews/current_injuries.csv"


@dataclass(frozen=True)
class MoneyPuckInjuryRow:
    player_id: str
    player_name: str
    team: str
    position: str
    date_of_return: str
    last_game_date: str
    injury_description: str
    injury_status: str
    games_still_to_miss: str
    games_missed_so_far: str


def _safe_str(val: object) -> str:
    if val is None:
        return ""
    return str(val).strip()


def _normalize_team(team: str) -> str:
    raw = (team or "").strip().upper()
    if not raw:
        return ""
    norm = normalize_team_code(raw, "nhl")
    return (norm or raw).strip().upper()


def fetch_current_injuries_csv(*, timeout: int = 20) -> Tuple[str, Dict[str, str]]:
    url = f"{MONEYPUCK_BASE}/{CURRENT_INJURIES_PATH}"
    headers = {"User-Agent": "Mozilla/5.0 (Chimera-v2c MoneyPuck injuries fetch)"}
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    meta = {
        "source_url": url,
        "etag": resp.headers.get("ETag", "") or "",
        "last_modified": resp.headers.get("Last-Modified", "") or "",
        "content_type": resp.headers.get("Content-Type", "") or "",
    }
    return resp.text, meta


def parse_current_injuries_csv(text: str) -> List[MoneyPuckInjuryRow]:
    reader = csv.DictReader(StringIO(text))
    out: List[MoneyPuckInjuryRow] = []
    for row in reader:
        pid = _safe_str(row.get("playerId"))
        name = _safe_str(row.get("playerName"))
        team = _normalize_team(_safe_str(row.get("teamCode")))
        pos = _safe_str(row.get("position")).upper()
        date_of_return = _safe_str(row.get("dateOfReturn"))
        last_game_date = _safe_str(row.get("lastGameDate"))
        desc = _safe_str(row.get("yahooInjuryDescription"))
        status = _safe_str(row.get("playerInjuryStatus")).upper()
        games_still = _safe_str(row.get("gamesStillToMiss"))
        games_missed = _safe_str(row.get("gamesMissedSoFar"))
        if not pid or not team:
            continue
        out.append(
            MoneyPuckInjuryRow(
                player_id=pid,
                player_name=name,
                team=team,
                position=pos,
                date_of_return=date_of_return,
                last_game_date=last_game_date,
                injury_description=desc,
                injury_status=status,
                games_still_to_miss=games_still,
                games_missed_so_far=games_missed,
            )
        )
    return out


def canonical_rows(rows: Iterable[MoneyPuckInjuryRow]) -> List[Dict[str, str]]:
    """
    Convert to a stable, JSON-serializable, sorted representation for hashing/diffs.
    """
    canon = [
        {
            "player_id": r.player_id,
            "player_name": r.player_name,
            "team": r.team,
            "position": r.position,
            "injury_status": r.injury_status,
            "injury_description": r.injury_description,
            "date_of_return": r.date_of_return,
            "last_game_date": r.last_game_date,
            "games_still_to_miss": r.games_still_to_miss,
            "games_missed_so_far": r.games_missed_so_far,
        }
        for r in rows
    ]
    canon.sort(key=lambda x: (x["team"], x["player_id"], x["injury_status"], x["date_of_return"]))
    return canon


def sha256_of_rows(canon_rows: List[Dict[str, str]]) -> str:
    payload = json.dumps(canon_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def diff_by_player_id(
    *,
    old_rows: List[Dict[str, str]],
    new_rows: List[Dict[str, str]],
) -> Dict[str, List[Dict[str, object]]]:
    """
    Diff keyed by player_id. Output is stable for logging.
    """
    old = {str(r.get("player_id") or ""): r for r in old_rows if r.get("player_id")}
    new = {str(r.get("player_id") or ""): r for r in new_rows if r.get("player_id")}
    added = []
    removed = []
    changed = []

    for pid in sorted(set(old.keys()) | set(new.keys())):
        o = old.get(pid)
        n = new.get(pid)
        if o is None and n is not None:
            added.append(n)
            continue
        if n is None and o is not None:
            removed.append(o)
            continue
        if o is None or n is None:
            continue
        if o != n:
            changed.append({"player_id": pid, "before": o, "after": n})

    return {"added": added, "removed": removed, "changed": changed}


def render_team_digest(
    *,
    date_iso: str,
    rows: List[Dict[str, str]],
    teams: Optional[List[str]] = None,
    max_last_game_age_days: int = 400,
) -> str:
    """
    Render a deterministic text digest suitable for feeding to the NHL LLM injury applier.
    If teams is provided, filter to that set (case-insensitive).
    """
    team_set = {t.strip().upper() for t in (teams or []) if t and t.strip()}
    try:
        date_obj = datetime.fromisoformat(date_iso).date()
    except Exception:
        date_obj = None

    by_team: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        team = (r.get("team") or "").strip().upper()
        if not team:
            continue
        if team_set and team not in team_set:
            continue
        if date_obj is not None and max_last_game_age_days > 0:
            lgd = (r.get("last_game_date") or "").strip()
            if lgd:
                try:
                    lgd_date = datetime.fromisoformat(lgd).date()
                except Exception:
                    lgd_date = None
                if lgd_date is not None:
                    age_days = (date_obj - lgd_date).days
                    if age_days > max_last_game_age_days:
                        continue
        by_team.setdefault(team, []).append(r)
    for t in by_team.keys():
        by_team[t].sort(key=lambda x: (x.get("injury_status", ""), x.get("position", ""), x.get("player_name", "")))

    lines: List[str] = []
    lines.append(f"{date_iso} NHL — MoneyPuck current injuries")
    lines.append("")
    if not by_team:
        lines.append("(No injuries listed for filtered teams.)")
        lines.append("")
        return "\n".join(lines)

    for team in sorted(by_team.keys()):
        lines.append(f"{team}:")
        for r in by_team[team]:
            name = r.get("player_name") or ""
            pos = r.get("position") or ""
            status = r.get("injury_status") or ""
            desc = r.get("injury_description") or ""
            dor = r.get("date_of_return") or ""
            lgd = r.get("last_game_date") or ""
            gsm = r.get("games_still_to_miss") or ""
            gms = r.get("games_missed_so_far") or ""
            parts = [p for p in [name, pos and f"({pos})", status and f"[{status}]", desc] if p]
            tail = []
            if dor and dor != "2099-12-31":
                tail.append(f"return={dor}")
            if lgd:
                tail.append(f"last_game={lgd}")
            if gsm and gsm != "-999":
                tail.append(f"games_still={gsm}")
            if gms:
                tail.append(f"missed={gms}")
            if tail:
                parts.append(" ".join(tail))
            lines.append("- " + " ".join(parts).strip())
        lines.append("")
    return "\n".join(lines)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
