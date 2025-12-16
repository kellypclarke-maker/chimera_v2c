#!/usr/bin/env python
"""
Align daily-ledger `market_proxy` to sportsbook odds at each game's ESPN start time.

This is useful when comparing against `kalshi_mid` captured at/near game start so
the two market baselines are time-aligned ("apples to apples").

Safety:
- Dry-run by default.
- Respects `reports/daily_ledgers/locked/YYYYMMDD.lock` unless --force.
- Snapshots each touched daily ledger to `reports/daily_ledgers/snapshots/` before writing.
- Only edits the `market_proxy` column (no row adds/drops).
- Overwriting existing `market_proxy` values requires --overwrite-existing.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.lib.odds_history import BooksSnapshot, fetch_books_snapshot
from chimera_v2c.lib.team_mapper import normalize_team_code
from chimera_v2c.src.ledger.formatting import format_prob_cell
from chimera_v2c.src.ledger.guard import LedgerGuardError, load_locked_dates, snapshot_file
from chimera_v2c.lib import espn_schedule


DAILY_DIR = Path("reports/daily_ledgers")
LOCK_DIR = DAILY_DIR / "locked"
SNAPSHOT_DIR = DAILY_DIR / "snapshots"


def _parse_date_token(token: str) -> dt.date:
    return dt.datetime.strptime(token, "%Y%m%d").date()


def _parse_iso_z(s: str) -> dt.datetime:
    s = (s or "").strip()
    if not s:
        raise ValueError("empty datetime")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return dt.datetime.fromisoformat(s)


def _to_iso_z(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    ts = ts.astimezone(dt.timezone.utc)
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _anchor_iso(start_iso: str, *, minutes_before_start: int) -> str:
    """
    Convert an ESPN game start timestamp into the market snapshot anchor timestamp.
    Example: minutes_before_start=30 targets T-30 minutes.
    """
    if minutes_before_start < 0:
        raise ValueError("minutes_before_start must be >= 0")
    start_dt = _parse_iso_z(start_iso).astimezone(dt.timezone.utc)
    anchor = start_dt - dt.timedelta(minutes=int(minutes_before_start))
    return _to_iso_z(anchor)


def _game_starts_by_matchup(league: str, date: dt.date) -> Dict[str, str]:
    """
    Return mapping AWAY@HOME -> start_ts_iso_z (seconds resolution).
    Uses ESPN scoreboard for the given league/date.
    """
    sb = espn_schedule.get_scoreboard(league, date)
    out: Dict[str, str] = {}
    for event in sb.get("events", []):
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        start_raw = comp.get("date") or event.get("date") or ""
        if not start_raw:
            continue
        try:
            start_dt = _parse_iso_z(str(start_raw))
        except ValueError:
            continue
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_abbr = ((home.get("team") or {}).get("abbreviation") or "").upper()
        away_abbr = ((away.get("team") or {}).get("abbreviation") or "").upper()
        home_code = normalize_team_code(home_abbr, league)
        away_code = normalize_team_code(away_abbr, league)
        if not home_code or not away_code:
            continue
        out[f"{away_code}@{home_code}"] = _to_iso_z(start_dt)
    return out


def _iter_daily_ledgers(start: Optional[dt.date], end: Optional[dt.date]) -> List[Path]:
    paths = sorted(DAILY_DIR.glob("*_daily_game_ledger.csv"))
    out: List[Path] = []
    for p in paths:
        token = p.name.split("_", 1)[0]
        if len(token) != 8 or not token.isdigit():
            continue
        d = _parse_date_token(token)
        if start and d < start:
            continue
        if end and d > end:
            continue
        out.append(p)
    return out


def _safe_float(val: object) -> Optional[float]:
    if val is None:
        return None
    try:
        x = float(str(val).strip())
    except Exception:
        return None
    if x < 0.0 or x > 1.0:
        return None
    return x


def _fetch_books_cached(
    cache: Dict[Tuple[str, str], Dict[str, BooksSnapshot]],
    *,
    league: str,
    snapshot_iso: str,
) -> Dict[str, BooksSnapshot]:
    key = (league.lower().strip(), snapshot_iso)
    if key not in cache:
        cache[key] = fetch_books_snapshot(league=league, snapshot_iso=snapshot_iso)
    return cache[key]


def _find_proxy_at_or_before_start(
    *,
    books_cache: Dict[Tuple[str, str], Dict[str, BooksSnapshot]],
    league: str,
    matchup: str,
    start_iso: str,
    fallback_before_minutes: int,
    fallback_step_minutes: int,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Return (market_proxy, snapshot_iso_used) for the matchup.
    If the anchor snapshot doesn't include the matchup, walk backwards in time (strictly before the anchor).
    """
    start_dt = _parse_iso_z(start_iso)
    # 1) exact start
    books = _fetch_books_cached(books_cache, league=league, snapshot_iso=start_iso)
    snap = books.get(matchup)
    if snap and snap.get("market_proxy") is not None:
        return float(snap["market_proxy"]), start_iso

    # 2) fallback backwards
    if fallback_before_minutes <= 0 or fallback_step_minutes <= 0:
        return None, None

    steps = int(fallback_before_minutes // fallback_step_minutes)
    for i in range(1, steps + 1):
        ts = start_dt - dt.timedelta(minutes=fallback_step_minutes * i)
        iso = _to_iso_z(ts)
        books = _fetch_books_cached(books_cache, league=league, snapshot_iso=iso)
        snap = books.get(matchup)
        if snap and snap.get("market_proxy") is not None:
            return float(snap["market_proxy"]), iso
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Align market_proxy to ESPN game start times (Odds API history).")
    ap.add_argument("--league", help="Optional league filter (nba|nhl|nfl).")
    ap.add_argument("--start-date", help="Start date (YYYY-MM-DD).")
    ap.add_argument("--end-date", help="End date (YYYY-MM-DD).")
    ap.add_argument("--min-abs-diff", type=float, default=0.05, help="Min |kalshi_mid-market_proxy| to target (default 0.05).")
    ap.add_argument("--max-abs-diff", type=float, default=0.10, help="Max |kalshi_mid-market_proxy| to target (default 0.10).")
    ap.add_argument(
        "--minutes-before-start",
        type=int,
        default=0,
        help="Anchor offset relative to ESPN start time (default: 0 for game start; 30 targets T-30m).",
    )
    ap.add_argument("--overwrite-existing", action="store_true", help="Allow overwriting non-blank market_proxy cells.")
    ap.add_argument("--force", action="store_true", help="Allow editing locked daily ledgers (explicit intent).")
    ap.add_argument("--fallback-before-minutes", type=int, default=180, help="If start snapshot missing, look back this many minutes.")
    ap.add_argument("--fallback-step-minutes", type=int, default=15, help="Step minutes for fallback lookback.")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run).")
    args = ap.parse_args()

    load_env_from_env_list()

    league_filter = (args.league or "").strip().lower() or None
    if league_filter and league_filter not in {"nba", "nhl", "nfl"}:
        raise SystemExit(f"[error] unsupported --league: {args.league}")

    start = dt.datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end = dt.datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None
    if int(args.minutes_before_start) < 0:
        raise SystemExit("[error] --minutes-before-start must be >= 0")

    locked = load_locked_dates(LOCK_DIR)
    books_cache: Dict[Tuple[str, str], Dict[str, BooksSnapshot]] = {}
    start_time_cache: Dict[Tuple[str, str], Dict[str, str]] = {}

    ledger_paths = _iter_daily_ledgers(start, end)
    total_candidates = 0
    total_updates = 0
    total_missing_odds = 0

    for ledger_path in ledger_paths:
        token = ledger_path.name.split("_", 1)[0]
        ymd = token
        date_iso = f"{token[:4]}-{token[4:6]}-{token[6:8]}"
        date_obj = dt.datetime.strptime(date_iso, "%Y-%m-%d").date()

        df = pd.read_csv(ledger_path, dtype=str).fillna("")
        if not {"date", "league", "matchup", "kalshi_mid", "market_proxy"} <= set(df.columns):
            continue

        df["league"] = df["league"].astype(str).str.lower()
        if league_filter:
            df = df[df["league"] == league_filter].copy()
            if df.empty:
                continue

        # Identify candidates within abs-diff bounds and with numeric values.
        candidates: List[int] = []
        diffs: Dict[int, float] = {}
        for idx, row in df.iterrows():
            kalshi = _safe_float(row.get("kalshi_mid"))
            proxy = _safe_float(row.get("market_proxy"))
            if kalshi is None or proxy is None:
                continue
            abs_diff = abs(kalshi - proxy)
            if abs_diff < float(args.min_abs_diff) or abs_diff > float(args.max_abs_diff):
                continue
            candidates.append(idx)
            diffs[idx] = abs_diff

        if not candidates:
            continue

        if args.apply and ymd in locked and not args.force:
            raise SystemExit(f"[error] {ymd} is locked; refusing to modify {ledger_path} (pass --force)")

        # Re-read full df for safe in-place editing.
        full_df = pd.read_csv(ledger_path, dtype=str).fillna("")
        full_df["league"] = full_df["league"].astype(str).str.lower()

        # Build start-time map for each league encountered in this file.
        leagues_in_file = sorted(set(full_df["league"].tolist()))
        for lg in leagues_in_file:
            if lg not in {"nba", "nhl", "nfl"}:
                continue
            key = (lg, date_iso)
            if key not in start_time_cache:
                start_time_cache[key] = _game_starts_by_matchup(lg, date_obj)

        updates_for_file: List[Tuple[str, str, str, str, str, str, str]] = []
        for idx in candidates:
            row = df.loc[idx]
            lg = str(row["league"]).strip().lower()
            matchup = str(row["matchup"]).strip()
            kalshi = _safe_float(row.get("kalshi_mid"))
            proxy_existing = str(row.get("market_proxy", "")).strip()
            if not matchup or kalshi is None:
                continue
            start_map = start_time_cache.get((lg, date_iso), {})
            espn_start_iso = start_map.get(matchup)
            if not espn_start_iso:
                total_missing_odds += 1
                continue
            anchor_iso = _anchor_iso(espn_start_iso, minutes_before_start=int(args.minutes_before_start))

            proxy_val, used_iso = _find_proxy_at_or_before_start(
                books_cache=books_cache,
                league=lg,
                matchup=matchup,
                start_iso=anchor_iso,
                fallback_before_minutes=int(args.fallback_before_minutes),
                fallback_step_minutes=int(args.fallback_step_minutes),
            )
            if proxy_val is None:
                total_missing_odds += 1
                continue

            new_cell = format_prob_cell(proxy_val, decimals=2, drop_leading_zero=True) or ""
            if not new_cell:
                total_missing_odds += 1
                continue

            if proxy_existing and not args.overwrite_existing:
                continue

            # No-op if unchanged.
            if proxy_existing == new_cell:
                continue

            # Update matching row in full_df by exact key fields.
            mask = (
                (full_df["date"].astype(str) == date_iso)
                & (full_df["league"].astype(str).str.lower() == lg)
                & (full_df["matchup"].astype(str) == matchup)
            )
            if not mask.any():
                continue
            full_df.loc[mask, "market_proxy"] = new_cell
            updates_for_file.append((date_iso, lg, matchup, str(kalshi), proxy_existing, new_cell, used_iso or ""))
            total_updates += 1

        total_candidates += len(candidates)

        if not updates_for_file:
            continue

        if not args.apply:
            print(f"[dry-run] {ledger_path.name}: updates={len(updates_for_file)} candidates={len(candidates)}")
            for d, lg, m, k, old_p, new_p, used_iso in updates_for_file:
                snap_label = f" @ {used_iso}" if used_iso else ""
                print(f"  {d} {lg} {m}{snap_label} kalshi_mid={k} market_proxy {old_p or '(blank)'} -> {new_p}")
            continue

        # Snapshot, then write.
        snapshot_file(ledger_path, SNAPSHOT_DIR)
        full_df.to_csv(ledger_path, index=False)
        print(f"[apply] {ledger_path.name}: updated market_proxy for {len(updates_for_file)} rows")

    print(
        f"Processed {len(ledger_paths)} ledgers; candidates={total_candidates} updated={total_updates} missing_odds={total_missing_odds}"
    )
    if not args.apply:
        print("[dry-run] No files were written. Use --apply to persist changes.")


if __name__ == "__main__":
    try:
        main()
    except LedgerGuardError as exc:
        raise SystemExit(f"[error] {exc}")
