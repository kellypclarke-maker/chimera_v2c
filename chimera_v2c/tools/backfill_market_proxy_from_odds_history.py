#!/usr/bin/env python
"""
Backfill sportsbook-implied market_proxy probabilities into daily ledgers (append-only).

- Does NOT touch kalshi_mid.
- Adds/fills `market_proxy` column in daily ledgers and propagates into the master via build_master_ledger.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone, date as dt_date
from pathlib import Path
from typing import Dict, Optional, Tuple, Mapping

import pandas as pd

from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.lib import espn_schedule, team_mapper
from chimera_v2c.lib.odds_history import BooksSnapshot, fetch_books_snapshot, normalize_team_name
from chimera_v2c.src.ledger.guard import (
    LedgerGuardError,
    compute_append_only_diff,
    load_locked_dates,
    snapshot_file,
    write_csv,
)


DAILY_LEDGER_DIR = Path("reports/daily_ledgers")
LOCK_DIR = DAILY_LEDGER_DIR / "locked"
LEDGER_SNAPSHOT_DIR = DAILY_LEDGER_DIR / "snapshots"


def ensure_market_proxy_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Ensure the `market_proxy` column exists. Returns (df, added_any)."""
    added = False
    if "market_proxy" not in df.columns:
        df["market_proxy"] = ""
        added = True
    return df, added


def normalize_matchup_key(matchup: str, league: str) -> Optional[str]:
    """
    Normalize a ledger matchup like "GS@PHI" or "UTAH@NO" to canonical codes
    for Odds API lookup (e.g. "GSW@PHI", "UTA@NOP"). Does not mutate ledgers.
    """
    if not matchup or "@" not in matchup:
        return None
    away_raw, home_raw = [p.strip() for p in str(matchup).split("@", 1)]
    if not away_raw or not home_raw:
        return None
    away = normalize_team_name(away_raw, league)
    home = normalize_team_name(home_raw, league)
    if not away or not home:
        return None
    return f"{away}@{home}"


def apply_market_proxy(
    ledger_path: Path,
    league: Optional[str],
    books_map: Dict[str, BooksSnapshot],
    allow_overwrite: bool,
    apply: bool,
) -> Tuple[bool, pd.DataFrame]:
    original = pd.read_csv(ledger_path, dtype=str, keep_default_na=False).fillna("")
    df, added_col = ensure_market_proxy_column(original.copy())
    updated = added_col
    for idx, row in df.iterrows():
        if league and str(row.get("league", "")).lower() != league.lower():
            continue
        matchup = row.get("matchup")
        if not matchup:
            continue
        key = str(matchup)
        lookup_key = key
        if lookup_key not in books_map:
            row_league = (league or str(row.get("league", ""))).lower()
            norm_key = normalize_matchup_key(lookup_key, row_league) if row_league else None
            if norm_key and norm_key in books_map:
                lookup_key = norm_key
            else:
                continue

        snap = books_map.get(lookup_key) or {}
        existing_proxy = str(df.at[idx, "market_proxy"]).strip()
        existing_missing = (not existing_proxy) or existing_proxy.upper() == "NR"

        new_proxy = snap.get("market_proxy")

        # Default behavior: fill blanks only (per-field).
        if new_proxy is not None and (allow_overwrite or existing_missing):
            from chimera_v2c.src.ledger.formatting import format_prob_cell

            formatted = format_prob_cell(new_proxy, decimals=2, drop_leading_zero=True)
            if formatted:
                df.at[idx, "market_proxy"] = formatted
                updated = True

    if not updated or not apply:
        return updated, df

    try:
        old_rows = original.fillna("").astype(str).to_dict("records")
        new_rows = df.fillna("").astype(str).to_dict("records")
        key_fields = ["date", "league", "matchup"]
        compute_append_only_diff(
            old_rows=old_rows,
            new_rows=new_rows,
            key_fields=key_fields,
            value_fields=[c for c in df.columns if c not in key_fields],
            blank_sentinels={"NR"},
        )
    except LedgerGuardError:
        if not allow_overwrite:
            raise
    snapshot_file(ledger_path, LEDGER_SNAPSHOT_DIR)
    write_csv(ledger_path, new_rows, fieldnames=list(df.columns))
    return updated, df


def ledger_date(path: Path) -> str:
    # expects YYYYMMDD_daily_game_ledger.csv
    return path.name.split("_")[0]


def _parse_utc_iso(text: str) -> Optional[datetime]:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        out = datetime.fromisoformat(s)
    except ValueError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def fetch_start_times_by_matchup(league: str, day: dt_date) -> Dict[str, datetime]:
    sb = espn_schedule.get_scoreboard(league, day)
    out: Dict[str, datetime] = {}
    for event in sb.get("events", []) or []:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        def _abbr(team: Mapping[str, object]) -> str:
            team_info = team.get("team") if isinstance(team.get("team"), dict) else {}
            return str(
                (team_info.get("abbreviation") or team_info.get("shortDisplayName") or "")
            ).upper()

        start_iso = event.get("date") or comp.get("date")
        start_dt = _parse_utc_iso(str(start_iso) if start_iso else "")
        if start_dt is None:
            continue
        away_code = team_mapper.normalize_team_code(_abbr(away), league)
        home_code = team_mapper.normalize_team_code(_abbr(home), league)
        if not away_code or not home_code:
            continue
        out[f"{away_code}@{home_code}"] = start_dt
    return out


def _normalize_matchup_codes(matchup: str, league: str) -> Optional[str]:
    if not matchup or "@" not in matchup:
        return None
    away_raw, home_raw = [p.strip() for p in str(matchup).split("@", 1)]
    if not away_raw or not home_raw:
        return None
    away = team_mapper.normalize_team_code(away_raw, league) or away_raw.strip().upper()
    home = team_mapper.normalize_team_code(home_raw, league) or home_raw.strip().upper()
    if not away or not home:
        return None
    return f"{away}@{home}"


def apply_market_proxy_tminus(
    *,
    ledger_path: Path,
    league: Optional[str],
    day: dt_date,
    minutes_before_start: int,
    api_key: str,
    allow_overwrite: bool,
    apply: bool,
) -> Tuple[bool, pd.DataFrame]:
    original = pd.read_csv(ledger_path, dtype=str, keep_default_na=False).fillna("")
    df, added_col = ensure_market_proxy_column(original.copy())
    updated = added_col

    leagues_present = sorted(
        {
            str(x).lower().strip()
            for x in df.get("league", pd.Series([], dtype=str)).tolist()
            if str(x).strip()
        }
    )
    leagues = [league.lower()] if league else leagues_present
    leagues = [lg for lg in leagues if lg in ("nba", "nhl", "nfl")]
    start_times = {lg: fetch_start_times_by_matchup(lg, day) for lg in leagues}

    # Cache Odds API history calls per (league, snapshot_iso).
    books_cache: Dict[Tuple[str, str], Dict[str, BooksSnapshot]] = {}

    for idx, row in df.iterrows():
        row_league = str(row.get("league", "")).lower().strip()
        if row_league not in leagues:
            continue
        matchup = str(row.get("matchup", "")).strip()
        if "@" not in matchup:
            continue

        matchup_codes = _normalize_matchup_codes(matchup, row_league)
        if not matchup_codes:
            continue
        start_dt = start_times.get(row_league, {}).get(matchup_codes)
        if start_dt is None:
            continue

        target_dt = (start_dt - timedelta(minutes=int(minutes_before_start))).replace(second=0, microsecond=0)
        snapshot_iso = target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        cache_key = (row_league, snapshot_iso)
        if cache_key not in books_cache:
            books_cache[cache_key] = fetch_books_snapshot(league=row_league, snapshot_iso=snapshot_iso, api_key=api_key)
        books_map = books_cache[cache_key]

        lookup_key = matchup_codes
        if lookup_key not in books_map:
            norm_key = normalize_matchup_key(matchup_codes, row_league)
            if norm_key and norm_key in books_map:
                lookup_key = norm_key
            else:
                continue

        snap = books_map.get(lookup_key) or {}
        new_proxy = snap.get("market_proxy")
        if new_proxy is None:
            continue

        existing_proxy = str(df.at[idx, "market_proxy"]).strip()
        existing_missing = (not existing_proxy) or existing_proxy.upper() == "NR"
        if not (allow_overwrite or existing_missing):
            continue

        from chimera_v2c.src.ledger.formatting import format_prob_cell

        formatted = format_prob_cell(new_proxy, decimals=2, drop_leading_zero=True)
        if formatted:
            df.at[idx, "market_proxy"] = formatted
            updated = True

    if not updated or not apply:
        return updated, df

    old_rows = original.fillna("").astype(str).to_dict("records")
    new_rows = df.fillna("").astype(str).to_dict("records")
    key_fields = ["date", "league", "matchup"]
    try:
        compute_append_only_diff(
            old_rows=old_rows,
            new_rows=new_rows,
            key_fields=key_fields,
            value_fields=[c for c in df.columns if c not in key_fields],
            blank_sentinels={"NR"},
        )
    except LedgerGuardError:
        if not allow_overwrite:
            raise
    snapshot_file(ledger_path, LEDGER_SNAPSHOT_DIR)
    write_csv(ledger_path, new_rows, fieldnames=list(df.columns))
    return updated, df


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill market_proxy from Odds API history (append-only).")
    ap.add_argument("--league", help="Optional league filter (nba|nhl|nfl).")
    ap.add_argument("--start-date", help="Optional start date (YYYY-MM-DD).")
    ap.add_argument("--end-date", help="Optional end date (YYYY-MM-DD).")
    ap.add_argument("--snapshot-time", default="18:00:00Z", help="Snapshot time suffix for odds-history (default: 18:00:00Z).")
    ap.add_argument(
        "--minutes-before-start",
        type=int,
        help="If set, backfill per-game market_proxy at ESPN start minus N minutes (e.g. 30 for T-30). Overrides --snapshot-time.",
    )
    ap.add_argument("--api-key", help="Odds history API key (fallback to THE_ODDS_API_HISTORY_KEY or THE_ODDS_API_HISTORY).")
    ap.add_argument("--allow-locked", action="store_true", help="Allow filling locked daily ledgers (still only fills blanks).")
    ap.add_argument("--overwrite-existing", action="store_true", help="Allow overwriting non-blank market_proxy cells (dangerous).")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Back-compat: equivalent to --allow-locked + --overwrite-existing.",
    )
    ap.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run).")
    args = ap.parse_args()

    load_env_from_env_list()
    api_key = args.api_key or os.getenv("THE_ODDS_API_HISTORY_KEY") or os.getenv("THE_ODDS_API_HISTORY")
    if not api_key:
        raise SystemExit("Missing THE_ODDS_API_HISTORY_KEY / THE_ODDS_API_HISTORY or --api-key.")

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None
    locked = load_locked_dates(LOCK_DIR)
    allow_locked = bool(args.force or args.allow_locked)
    allow_overwrite = bool(args.force or args.overwrite_existing)

    ledger_paths = sorted(DAILY_LEDGER_DIR.glob("*_daily_game_ledger.csv"))
    updated_files = []
    skipped_locked = []

    for path in ledger_paths:
        date_token = ledger_date(path)
        dt = datetime.strptime(date_token, "%Y%m%d").date()
        if start and dt < start:
            continue
        if end and dt > end:
            continue
        if date_token in locked and not allow_locked:
            skipped_locked.append(path)
            continue

        if args.minutes_before_start is not None:
            updated, _ = apply_market_proxy_tminus(
                ledger_path=path,
                league=args.league,
                day=dt,
                minutes_before_start=int(args.minutes_before_start),
                api_key=api_key,
                allow_overwrite=allow_overwrite,
                apply=args.apply,
            )
        else:
            # Fetch once per date/league
            books_map: Dict[str, BooksSnapshot] = {}
            leagues = [args.league] if args.league else ["nba", "nhl", "nfl"]
            snapshot_iso = f"{dt.isoformat()}T{args.snapshot_time}"
            for lg in leagues:
                daily_map = fetch_books_snapshot(league=lg, snapshot_iso=snapshot_iso, api_key=api_key)
                books_map.update(daily_map)

            updated, _ = apply_market_proxy(
                path,
                args.league,
                books_map,
                allow_overwrite=allow_overwrite,
                apply=args.apply,
            )
        if updated:
            updated_files.append(path)

    print(f"Processed {len(ledger_paths)} ledgers; updated {len(updated_files)}")
    if skipped_locked:
        print(f"Skipped locked files (use --allow-locked or --force to override): {[p.name for p in skipped_locked]}")
    if not args.apply:
        print("[dry-run] No files were written. Use --apply to persist market_proxy fills.")


if __name__ == "__main__":
    main()
