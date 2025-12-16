from __future__ import annotations

"""
Backfill missing `actual_outcome` cells in daily ledgers using ESPN scoreboards.

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/fill_missing_daily_outcomes.py
  PYTHONPATH=. python chimera_v2c/tools/fill_missing_daily_outcomes.py --date 2025-11-22

This script:
  - Scans `reports/daily_ledgers/*_daily_game_ledger.csv` (or a single date).
  - For any row with a blank `actual_outcome`, fetches the ESPN scoreboard for
    that league/date and (only when ESPN reports the game is final) fills in:
      "<AWAY> a_score-h_score <HOME>".
  - With `--overwrite-existing`, also corrects non-blank `actual_outcome` cells
    when they do not match ESPN finals (use with care; snapshots before writing),
    and clears obviously-non-final placeholders like `0-0 (push)` back to blank
    when ESPN shows the game is not final.
  - Leaves all existing rows/columns untouched except for previously blank
    `actual_outcome` fields.

It is intended for historical backfill (e.g., pre-2025-12-02) so that all
daily ledgers have verifiable final scores for model grading.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from chimera_v2c.lib import nhl_scoreboard, team_mapper
from chimera_v2c.src.ledger.guard import (
    LedgerGuardError,
    compute_append_only_diff,
    load_locked_dates,
    snapshot_file,
)
from chimera_v2c.src.ledger.outcomes import format_final_score, is_placeholder_outcome


DAILY_DIR = Path("reports/daily_ledgers")
LOCK_DIR = DAILY_DIR / "locked"
SNAPSHOT_DIR = DAILY_DIR / "snapshots" / "fill_missing_daily_outcomes"
OVERWRITE_EXISTING = False


def fetch_scoreboard(league: str, date_str: str) -> Dict:
    league_l = league.lower()
    if league_l == "nba":
        return nhl_scoreboard.fetch_nba_scoreboard(date_str)
    if league_l == "nfl":
        return nhl_scoreboard.fetch_nfl_scoreboard(date_str)
    return nhl_scoreboard.fetch_nhl_scoreboard(date_str)


def find_game(sb: Dict, league: str, away_raw: str, home_raw: str) -> Optional[Dict]:
    """Return the raw ESPN game dict for the matchup if found."""
    games = sb.get("games") or []
    league_l = league.lower()
    away_norm = team_mapper.normalize_team_code(away_raw, league_l)
    home_norm = team_mapper.normalize_team_code(home_raw, league_l)
    if not away_norm or not home_norm:
        return None

    candidate: Optional[Dict] = None
    for g in games:
        teams = g.get("teams") or {}
        home_alias = (teams.get("home") or {}).get("alias")
        away_alias = (teams.get("away") or {}).get("alias")
        if not home_alias or not away_alias:
            continue
        sb_home = team_mapper.normalize_team_code(home_alias, league_l)
        sb_away = team_mapper.normalize_team_code(away_alias, league_l)
        if sb_home == home_norm and sb_away == away_norm:
            # Prefer final games when duplicates appear (defensive coding).
            status = g.get("status") or {}
            if (status.get("state") or "").lower() == "post":
                return g
            candidate = g
    return candidate


def match_game(
    sb: Dict,
    league: str,
    away_raw: str,
    home_raw: str,
) -> Optional[Tuple[float, float]]:
    """Return (away_score, home_score) for the given matchup if found."""
    g = find_game(sb, league, away_raw, home_raw)
    if not g:
        return None
    status = g.get("status") or {}
    if (status.get("state") or "").lower() != "post":
        # Guard against ESPN pre-game / in-progress rows.
        return None
    scores = g.get("scores") or {}
    try:
        home_score = float(scores.get("home"))
        away_score = float(scores.get("away"))
    except (TypeError, ValueError):
        return None
    return away_score, home_score


def _clean_cell(value: object) -> str:
    s = str(value).strip() if value is not None else ""
    if not s or s.lower() in {"nan", "none", "na"}:
        return ""
    return s


def _clean_outcome_cell(value: object) -> str:
    s = _clean_cell(value)
    if s.upper() == "NR":
        return ""
    return s


def _expected_outcome(away_raw: str, home_raw: str, away_score: float, home_score: float) -> str:
    return format_final_score(away_raw, home_raw, away_score, home_score)


def fill_missing_for_file(path: Path, *, allow_locked: bool) -> bool:
    """Fill missing actual_outcome cells in a single daily ledger file."""
    locked = load_locked_dates(LOCK_DIR)
    ymd = path.name.split("_", 1)[0]
    if ymd in locked and not allow_locked:
        return False

    changed = False
    rows: List[Dict[str, str]] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    old_rows = [dict(r) for r in rows]

    # Cache scoreboards by (league, date)
    sb_cache: Dict[Tuple[str, str], Dict] = {}

    for row in rows:
        date = _clean_cell(row.get("date"))
        league = _clean_cell(row.get("league")).lower()
        matchup = _clean_cell(row.get("matchup"))
        if not date or not league or "@" not in matchup:
            continue

        away_raw, home_raw = matchup.split("@", 1)
        away_raw = away_raw.strip()
        home_raw = home_raw.strip()

        existing_outcome_raw = _clean_cell(row.get("actual_outcome"))
        existing_outcome = _clean_outcome_cell(existing_outcome_raw)

        cache_key = (league, date)
        if cache_key not in sb_cache:
            sb = fetch_scoreboard(league, date)
            sb_cache[cache_key] = sb
        else:
            sb = sb_cache[cache_key]

        if sb.get("status") != "ok":
            continue

        game = find_game(sb, league, away_raw, home_raw)
        if not game:
            continue

        status = game.get("status") or {}
        state = (status.get("state") or "").lower()
        if state != "post":
            if OVERWRITE_EXISTING and existing_outcome_raw and is_placeholder_outcome(existing_outcome_raw):
                row["actual_outcome"] = ""
                changed = True
            continue

        scores = game.get("scores") or {}
        try:
            away_score = float(scores.get("away"))
            home_score = float(scores.get("home"))
        except (TypeError, ValueError):
            continue
        new_outcome = _expected_outcome(away_raw, home_raw, away_score, home_score)

        if not existing_outcome:
            row["actual_outcome"] = new_outcome
            changed = True
        elif OVERWRITE_EXISTING and existing_outcome_raw != new_outcome:
            row["actual_outcome"] = new_outcome
            changed = True

    if not changed:
        return False

    try:
        if not OVERWRITE_EXISTING:
            compute_append_only_diff(
                old_rows=old_rows,
                new_rows=rows,
                key_fields=["date", "league", "matchup"],
                value_fields=[c for c in fieldnames if c not in ("date", "league", "matchup")],
                blank_sentinels={"NR"},
            )
        else:
            # Overwrite mode: only allow changes to `actual_outcome` that do not
            # clear a non-empty value. Any other field overwrite is forbidden.
            old_map = {(r.get("date"), r.get("league"), r.get("matchup")): r for r in old_rows}
            new_map = {(r.get("date"), r.get("league"), r.get("matchup")): r for r in rows}
            removed = set(old_map) - set(new_map)
            if removed:
                raise LedgerGuardError(f"Append-only violation: attempted to drop keys {sorted(removed)[:5]}...")
            for key in set(old_map) & set(new_map):
                o = old_map[key]
                n = new_map[key]
                for field in fieldnames:
                    if field in ("date", "league", "matchup"):
                        continue
                    old_val = _clean_cell(o.get(field))
                    new_val = _clean_cell(n.get(field))
                    if field == "actual_outcome":
                        if old_val and not new_val and not is_placeholder_outcome(old_val):
                            raise LedgerGuardError(
                                f"Append-only violation for {key}: actual_outcome lost value '{old_val}'."
                            )
                        continue
                    if old_val and not new_val:
                        raise LedgerGuardError(
                            f"Append-only violation for {key}: field '{field}' lost value '{old_val}'."
                        )
                    if old_val and new_val and old_val != new_val:
                        raise LedgerGuardError(
                            f"Append-only violation for {key}: field '{field}' overwrite '{old_val}' -> '{new_val}'."
                        )
    except LedgerGuardError as exc:
        raise SystemExit(f"[error] append-only violation for {path}: {exc}") from exc

    snapshot_file(path, SNAPSHOT_DIR)

    # Rewrite file with updated rows (surgical changes to actual_outcome only).
    # Temporarily make file writable if needed.
    original_mode = None
    try:
        original_mode = path.stat().st_mode
        path.chmod(original_mode | 0o200)
    except Exception:
        original_mode = None

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Restore original mode (best-effort).
    if original_mode is not None:
        try:
            path.chmod(original_mode)
        except Exception:
            pass

    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fill missing actual_outcome cells in daily ledgers using ESPN scoreboards."
    )
    ap.add_argument(
        "--date",
        help="Optional YYYY-MM-DD to restrict to a single date. "
        "If omitted, all *_daily_game_ledger.csv files are scanned.",
    )
    ap.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Also correct non-blank actual_outcome values when ESPN shows a final (snapshots before writing).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Allow filling locked daily ledgers (should only be used with explicit human approval).",
    )
    args = ap.parse_args()

    global OVERWRITE_EXISTING
    OVERWRITE_EXISTING = bool(args.overwrite_existing)

    if not DAILY_DIR.exists():
        raise SystemExit(f"Daily ledgers directory not found: {DAILY_DIR}")

    if args.date:
        pattern = f"{args.date.replace('-', '')}_daily_game_ledger.csv"
        targets = [DAILY_DIR / pattern] if (DAILY_DIR / pattern).exists() else []
    else:
        targets = sorted(DAILY_DIR.glob("*_daily_game_ledger.csv"))

    if not targets:
        print("[info] no matching daily ledgers found.")
        return

    total_changed = 0
    for path in targets:
        if fill_missing_for_file(path, allow_locked=args.force):
            total_changed += 1
            print(f"[info] updated outcomes in {path}")

    print(f"[info] files updated: {total_changed}")


if __name__ == "__main__":
    main()
