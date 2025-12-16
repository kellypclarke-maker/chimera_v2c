#!/usr/bin/env python
"""
Update a canonical copy of MoneyPuck's NHL injury table and optionally emit a
slate-filtered text digest for the LLM injury delta applier.

Source (public, no API key):
  https://moneypuck.com/moneypuck/playerData/playerNews/current_injuries.csv

Writes (append-safe via snapshots on overwrite of these *data* files):
  - chimera_v2c/data/moneypuck_injuries_nhl.json   (rows + metadata + sha256)
  - chimera_v2c/data/moneypuck_injuries_nhl.csv    (rows only; stable sort)
  - chimera_v2c/data/moneypuck_injuries_<date>_nhl.txt (optional LLM digest)
  - reports/alerts/moneypuck_injuries/moneypuck_injuries_diff_<ts>.json (when changed)

This tool does NOT touch daily ledgers.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import date as dt_date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from chimera_v2c.lib.moneypuck_injuries import (
    canonical_rows,
    diff_by_player_id,
    fetch_current_injuries_csv,
    parse_current_injuries_csv,
    render_team_digest,
    sha256_of_rows,
    utc_now_iso,
)
from chimera_v2c.lib.nhl_scoreboard import fetch_nhl_scoreboard
from chimera_v2c.lib.team_mapper import normalize_team_code
from chimera_v2c.src.ledger.guard import snapshot_file


DATA_DIR = Path("chimera_v2c/data")
OUT_JSON = DATA_DIR / "moneypuck_injuries_nhl.json"
OUT_CSV = DATA_DIR / "moneypuck_injuries_nhl.csv"
ALERT_DIR = Path("reports/alerts/moneypuck_injuries")
SNAPSHOT_DIR = Path("reports/moneypuck_injury_snapshots")


def _read_existing_rows() -> Tuple[str, List[Dict[str, str]]]:
    if not OUT_JSON.exists():
        return "", []
    try:
        payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    except Exception:
        return "", []
    sha = str(payload.get("sha256") or "")
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return sha, []
    out: List[Dict[str, str]] = []
    for r in rows:
        if isinstance(r, dict):
            out.append({str(k): str(v) for k, v in r.items()})
    return sha, out


def _write_rows_csv(rows: List[Dict[str, str]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "player_id",
        "player_name",
        "team",
        "position",
        "injury_status",
        "injury_description",
        "date_of_return",
        "last_game_date",
        "games_still_to_miss",
        "games_missed_so_far",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k) or "") for k in fieldnames})


def slate_teams_for_date(date_iso: str) -> List[str]:
    sb = fetch_nhl_scoreboard(date_iso)
    teams = set()
    for g in sb.get("games") or []:
        t = g.get("teams") or {}
        away_raw = str((t.get("away") or {}).get("alias") or "").strip().upper()
        home_raw = str((t.get("home") or {}).get("alias") or "").strip().upper()
        away = normalize_team_code(away_raw, "nhl") or away_raw
        home = normalize_team_code(home_raw, "nhl") or home_raw
        if away:
            teams.add(away)
        if home:
            teams.add(home)
    return sorted(teams)


def update_moneypuck_injuries_nhl(
    *,
    date_iso: Optional[str],
    write_digest: bool,
    force: bool,
) -> Dict[str, object]:
    prev_sha, prev_rows = _read_existing_rows()

    raw_csv, meta = fetch_current_injuries_csv()
    parsed = parse_current_injuries_csv(raw_csv)
    canon = canonical_rows(parsed)
    new_sha = sha256_of_rows(canon)

    changed = (new_sha != prev_sha) or (not prev_sha)
    if force:
        changed = True

    result: Dict[str, object] = {
        "changed": bool(changed),
        "prev_sha256": prev_sha,
        "new_sha256": new_sha,
        "row_count": len(canon),
        "source": meta,
        "fetched_at_utc": utc_now_iso(),
        "out_json": str(OUT_JSON),
        "out_csv": str(OUT_CSV),
    }

    # Only overwrite canonical artifacts when changed (avoid snapshot spam on pollers).
    if changed:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if OUT_JSON.exists():
            snapshot_file(OUT_JSON, SNAPSHOT_DIR)
        if OUT_CSV.exists():
            snapshot_file(OUT_CSV, SNAPSHOT_DIR)

        OUT_JSON.write_text(
            json.dumps(
                {
                    "fetched_at_utc": result["fetched_at_utc"],
                    "source": meta,
                    "sha256": new_sha,
                    "rows": canon,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _write_rows_csv(canon)

    digest_path = ""
    if write_digest and date_iso:
        try:
            teams = slate_teams_for_date(date_iso)
        except Exception:
            teams = []
        digest = render_team_digest(date_iso=date_iso, rows=canon, teams=teams if teams else None)
        digest_file = DATA_DIR / f"moneypuck_injuries_{date_iso}_nhl.txt"
        digest_file.write_text(digest, encoding="utf-8")
        digest_path = str(digest_file)
        result["digest_path"] = digest_path
        result["digest_team_count"] = len(teams)

    if changed:
        ALERT_DIR.mkdir(parents=True, exist_ok=True)
        diff = diff_by_player_id(old_rows=prev_rows, new_rows=canon)
        diff_path = ALERT_DIR / f"moneypuck_injuries_diff_{utc_now_iso().replace(':','').replace('-','')}.json"
        diff_path.write_text(
            json.dumps(
                {
                    "fetched_at_utc": result["fetched_at_utc"],
                    "prev_sha256": prev_sha,
                    "new_sha256": new_sha,
                    "row_count": len(canon),
                    "diff": diff,
                    "source": meta,
                    "digest_path": digest_path,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        result["diff_path"] = str(diff_path)

    return result


def _validate_date(date_iso: str) -> str:
    try:
        return dt_date.fromisoformat(str(date_iso).strip()).isoformat()
    except ValueError as exc:
        raise SystemExit(f"[error] invalid --date (expected YYYY-MM-DD): {date_iso}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description="Update MoneyPuck NHL injury snapshot (read-only on ledgers).")
    ap.add_argument("--date", help="Optional YYYY-MM-DD; when set, writes a slate-filtered digest for LLM.")
    ap.add_argument("--no-digest", action="store_true", help="Do not write the LLM digest file (default: write when --date is set).")
    ap.add_argument("--force", action="store_true", help="Force treat as changed (still snapshots previous files).")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON result.")
    args = ap.parse_args()

    date_iso = _validate_date(args.date) if args.date else None
    res = update_moneypuck_injuries_nhl(date_iso=date_iso, write_digest=bool(date_iso and not args.no_digest), force=bool(args.force))

    if args.json:
        print(json.dumps(res, indent=2, sort_keys=True))
        return

    changed = bool(res.get("changed"))
    print(f"[ok] MoneyPuck injuries updated ({res.get('row_count')} rows), changed={changed}")
    if res.get("digest_path"):
        print(f"[info] digest: {res['digest_path']}")
    if changed and res.get("diff_path"):
        print(f"[alert] diff: {res['diff_path']}")


if __name__ == "__main__":
    main()
