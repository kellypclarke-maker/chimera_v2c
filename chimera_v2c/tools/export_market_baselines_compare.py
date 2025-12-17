#!/usr/bin/env python
"""
Export a read-only comparison CSV for daily-ledger market baselines.

This tool recomputes:
  - Kalshi `kalshi_mid` at T-minus (via candlesticks)
  - Sportsbook `market_proxy` at the same T-minus (via Odds API history)

and writes a separate CSV for audit/compare purposes. It does NOT modify:
  - daily ledgers
  - master ledger

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/export_market_baselines_compare.py \
    --start-date YYYY-MM-DD --end-date YYYY-MM-DD --minutes-before-start 30
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from chimera_v2c.lib import espn_schedule, team_mapper
from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.lib.odds_history import BooksSnapshot, fetch_books_snapshot
from chimera_v2c.src.ledger.formatting import format_prob_cell
from chimera_v2c.tools.backfill_kalshi_mid_from_candlesticks import (
    SERIES_TICKER_BY_LEAGUE,
    ResolvedMarket,
    fetch_candlestick_mids,
    resolve_home_yes_market_ticker,
)


DAILY_DIR = Path("reports/daily_ledgers")
OUT_DIR = Path("reports/market_snapshots")


OUT_COLUMNS = [
    "date",
    "league",
    "matchup",
    "start_time_utc",
    "snapshot_iso",
    "ledger_kalshi_mid",
    "candlestick_kalshi_mid",
    "kalshi_mid_diff",
    "kalshi_event_ticker",
    "kalshi_market_ticker_home_yes",
    "ledger_market_proxy",
    "odds_market_proxy",
    "market_proxy_diff",
    "books_home_ml",
    "books_away_ml",
]


def _iter_dates(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def _ledger_path(date_iso: str) -> Path:
    return DAILY_DIR / f"{date_iso.replace('-', '')}_daily_game_ledger.csv"


def _parse_date(text: str) -> dt.date:
    return dt.datetime.strptime(text, "%Y-%m-%d").date()


def _parse_utc_iso(text: str) -> Optional[dt.datetime]:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def fetch_start_times_by_matchup(league: str, day: dt.date) -> Dict[str, dt.datetime]:
    sb = espn_schedule.get_scoreboard(league, day)
    out: Dict[str, dt.datetime] = {}
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

        def _abbr(team: Dict[str, object]) -> str:
            team_info = team.get("team") if isinstance(team.get("team"), dict) else {}
            return str(team_info.get("abbreviation") or team_info.get("shortDisplayName") or "").upper()

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


def _parse_prob_cell(text: object) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip()
    if not s or s.upper() == "NR" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        # daily ledgers may store ".85"
        if s.startswith("."):
            try:
                return float("0" + s)
            except ValueError:
                return None
        return None


def _fmt_prob_2(val: Optional[float]) -> str:
    if val is None:
        return ""
    return format_prob_cell(val, decimals=2, drop_leading_zero=True)


def _diff(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return ""
    return f"{(b - a):+.4f}"


def _chunked(items: Sequence[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


@dataclass(frozen=True)
class RowKey:
    date_iso: str
    league: str
    matchup: str


def export_compare_csv(
    *,
    start_date: str,
    end_date: str,
    league: Optional[str],
    minutes_before_start: int,
    period_interval_min: int,
    kalshi_public_base: str,
    out_path: Optional[Path],
    include_kalshi: bool,
    include_odds: bool,
) -> Path:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise SystemExit("[error] --end-date must be >= --start-date")

    league_filter = (league or "").lower().strip() or None
    if league_filter and league_filter not in SERIES_TICKER_BY_LEAGUE:
        raise SystemExit("[error] --league must be nba|nhl|nfl")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = OUT_DIR / (
            f"compare_kalshi_mid_market_proxy_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
            + (f"_{league_filter}" if league_filter else "")
            + f"_tminus{minutes_before_start}.csv"
        )

    session = requests.Session()
    session.headers.update({"User-Agent": "chimera_v2c/export_market_baselines_compare/1.0"})

    # Cache start times per (league, date).
    start_times_cache: Dict[Tuple[str, str], Dict[str, dt.datetime]] = {}
    # Cache Kalshi event->market tickers per (league, matchup).
    kalshi_market_cache: Dict[Tuple[str, str, str], Optional[ResolvedMarket]] = {}

    # Cache Odds API snapshots per (league, snapshot_iso).
    books_cache: Dict[Tuple[str, str], Dict[str, BooksSnapshot]] = {}

    # Batch Kalshi candlestick fetches per target timestamp.
    kalshi_tasks_by_ts: Dict[int, List[str]] = {}
    kalshi_event_by_market: Dict[str, str] = {}

    rows_out: List[Dict[str, str]] = []

    for day in _iter_dates(start, end):
        date_iso = day.isoformat()
        ledger_path = _ledger_path(date_iso)
        if not ledger_path.exists():
            continue
        df = pd.read_csv(ledger_path, dtype=str, keep_default_na=False).fillna("")
        for _, row in df.iterrows():
            lg = str(row.get("league", "")).lower().strip()
            if lg not in SERIES_TICKER_BY_LEAGUE:
                continue
            if league_filter and lg != league_filter:
                continue
            matchup_raw = str(row.get("matchup", "")).strip()
            if "@" not in matchup_raw:
                continue
            away_raw, home_raw = [p.strip() for p in matchup_raw.split("@", 1)]
            away = team_mapper.normalize_team_code(away_raw, lg) or away_raw.upper()
            home = team_mapper.normalize_team_code(home_raw, lg) or home_raw.upper()
            if not away or not home or away == home:
                continue
            matchup = f"{away}@{home}"

            cache_key = (lg, date_iso)
            if cache_key not in start_times_cache:
                start_times_cache[cache_key] = fetch_start_times_by_matchup(lg, day)
            start_dt = start_times_cache[cache_key].get(matchup)
            if start_dt is None:
                continue

            target_dt = (start_dt - dt.timedelta(minutes=int(minutes_before_start))).replace(second=0, microsecond=0)
            snapshot_iso = target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            ledger_kalshi = _parse_prob_cell(row.get("kalshi_mid"))
            ledger_proxy = _parse_prob_cell(row.get("market_proxy"))

            event_ticker = ""
            market_ticker = ""
            target_ts = int(target_dt.timestamp())
            if include_kalshi:
                mk = (lg, date_iso, matchup)
                if mk not in kalshi_market_cache:
                    kalshi_market_cache[mk] = resolve_home_yes_market_ticker(
                        league=lg,
                        date=day,
                        away=away,
                        home=home,
                        kalshi_public_base=kalshi_public_base,
                        session=session,
                    )
                resolved = kalshi_market_cache.get(mk)
                if resolved is not None:
                    event_ticker = resolved.event_ticker
                    market_ticker = resolved.market_ticker_home_yes
                    kalshi_tasks_by_ts.setdefault(target_ts, []).append(market_ticker)
                    kalshi_event_by_market[market_ticker] = event_ticker

            odds_proxy: Optional[float] = None
            books_home_ml = ""
            books_away_ml = ""
            if include_odds:
                books_key = (lg, snapshot_iso)
                if books_key not in books_cache:
                    books_cache[books_key] = fetch_books_snapshot(league=lg, snapshot_iso=snapshot_iso)
                snap = books_cache[books_key].get(matchup)
                if snap:
                    odds_proxy = snap.get("market_proxy")
                    if snap.get("books_home_ml") is not None:
                        books_home_ml = str(int(snap["books_home_ml"]))
                    if snap.get("books_away_ml") is not None:
                        books_away_ml = str(int(snap["books_away_ml"]))

            rows_out.append(
                {
                    "date": date_iso,
                    "league": lg,
                    "matchup": matchup,
                    "start_time_utc": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "snapshot_iso": snapshot_iso,
                    "ledger_kalshi_mid": _fmt_prob_2(ledger_kalshi) or (str(row.get("kalshi_mid", "")).strip()),
                    "candlestick_kalshi_mid": "",
                    "kalshi_mid_diff": "",
                    "kalshi_event_ticker": event_ticker,
                    "kalshi_market_ticker_home_yes": market_ticker,
                    "ledger_market_proxy": _fmt_prob_2(ledger_proxy) or (str(row.get("market_proxy", "")).strip()),
                    "odds_market_proxy": _fmt_prob_2(odds_proxy),
                    "market_proxy_diff": _diff(ledger_proxy, odds_proxy),
                    "books_home_ml": books_home_ml,
                    "books_away_ml": books_away_ml,
                    "__target_ts": str(target_ts),
                    "__ledger_kalshi": "" if ledger_kalshi is None else str(ledger_kalshi),
                }
            )

    # Resolve Kalshi candlestick mids in batches.
    cand_mid_by_ts_ticker: Dict[Tuple[int, str], Optional[float]] = {}
    if include_kalshi and kalshi_tasks_by_ts:
        for target_ts, tickers in sorted(kalshi_tasks_by_ts.items(), key=lambda kv: kv[0]):
            uniq = sorted(set(tickers))
            for chunk in _chunked(uniq, 50):
                mids = fetch_candlestick_mids(
                    market_tickers=chunk,
                    target_ts=int(target_ts),
                    period_interval_min=int(period_interval_min),
                    kalshi_public_base=kalshi_public_base,
                    session=session,
                )
                for t, mid in mids.items():
                    cand_mid_by_ts_ticker[(int(target_ts), str(t))] = mid

        for r in rows_out:
            mt = str(r.get("kalshi_market_ticker_home_yes", "")).strip()
            if not mt:
                continue
            try:
                ts = int(str(r.get("__target_ts", "")).strip())
            except Exception:
                continue
            cand_mid = cand_mid_by_ts_ticker.get((ts, mt))
            ledger_kalshi = _parse_prob_cell(r.get("__ledger_kalshi"))
            r["candlestick_kalshi_mid"] = _fmt_prob_2(cand_mid)
            r["kalshi_mid_diff"] = _diff(ledger_kalshi, cand_mid)
            if not r.get("kalshi_event_ticker") and mt in kalshi_event_by_market:
                r["kalshi_event_ticker"] = kalshi_event_by_market[mt]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in OUT_COLUMNS})

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export read-only comparison CSV for Kalshi mids and market_proxy.")
    ap.add_argument("--start-date", required=True, help="ISO start date (inclusive).")
    ap.add_argument("--end-date", required=True, help="ISO end date (inclusive).")
    ap.add_argument("--league", help="Optional league filter: nba|nhl|nfl.")
    ap.add_argument("--minutes-before-start", type=int, default=30, help="Minutes before ESPN start time (default: 30).")
    ap.add_argument("--period-interval-min", type=int, default=1, help="Candlestick interval in minutes (default: 1).")
    ap.add_argument(
        "--kalshi-public-base",
        default="https://api.elections.kalshi.com/trade-api/v2",
        help="Kalshi public base (default: live trade-api/v2).",
    )
    ap.add_argument("--out", help="Optional output CSV path.")
    ap.add_argument("--no-kalshi", action="store_true", help="Skip Kalshi candlestick mids.")
    ap.add_argument("--no-odds", action="store_true", help="Skip Odds API market_proxy.")
    args = ap.parse_args()

    if int(args.minutes_before_start) < 0:
        raise SystemExit("[error] --minutes-before-start must be >= 0")

    load_env_from_env_list()
    out_path = export_compare_csv(
        start_date=args.start_date,
        end_date=args.end_date,
        league=args.league,
        minutes_before_start=int(args.minutes_before_start),
        period_interval_min=int(args.period_interval_min),
        kalshi_public_base=str(args.kalshi_public_base),
        out_path=Path(args.out) if args.out else None,
        include_kalshi=not bool(args.no_kalshi),
        include_odds=not bool(args.no_odds),
    )
    print(f"[ok] wrote compare CSV -> {out_path}")


if __name__ == "__main__":
    main()
