#!/usr/bin/env python
"""
Export Kalshi bid/ask snapshots at T-minus using candlesticks (read-only).

This is meant to build an analysis-grade table for past games that captures the
best-available YES bid/ask close from Kalshi candlesticks at a consistent anchor
timestamp relative to ESPN scheduled start (default: T-30 minutes).

It does NOT modify daily ledgers.

Output columns (one row per market ticker):
  - date, league, matchup (AWAY@HOME)
  - start_time_utc, minutes_before_start, target_ts, target_iso
  - event_ticker, market_ticker, yes_team
  - yes_bid_cents, yes_ask_cents, mid, spread, candle_end_ts
  - ok (1/0), error (string)

Usage:
  PYTHONPATH=. python chimera_v2c/tools/export_kalshi_bidask_tminus.py \
    --start-date 2025-11-19 --end-date 2025-12-15 --minutes-before-start 30
"""

from __future__ import annotations

import argparse
import datetime as dt
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from chimera_v2c.lib import espn_schedule, team_mapper
from chimera_v2c.src.ledger_analysis import load_games
from chimera_v2c.tools.backfill_kalshi_mid_from_candlesticks import (
    SERIES_TICKER_BY_LEAGUE,
    _align_ts_to_minute,  # reuse alignment
    fetch_candlestick_quotes,
    resolve_home_yes_market_ticker,
)


DAILY_DIR = Path("reports/daily_ledgers")
OUT_DIR = Path("reports/market_snapshots")


def _iter_dates(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


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


def _chunked(items: Sequence[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


@dataclass(frozen=True)
class MarketRef:
    league: str
    date: dt.date
    matchup: str
    event_ticker: str
    market_ticker: str
    yes_team: str
    start_time_utc: dt.datetime
    target_ts: int


def _event_markets_for_matchup(
    *,
    league: str,
    day: dt.date,
    matchup: str,
    kalshi_public_base: str,
    session: requests.Session,
) -> List[MarketRef]:
    """
    Resolve both HOME-YES and AWAY-YES tickers for an ESPN matchup using the Kalshi event endpoint.
    """
    if "@" not in matchup:
        return []
    away_raw, home_raw = matchup.split("@", 1)
    away = team_mapper.normalize_team_code(away_raw, league)
    home = team_mapper.normalize_team_code(home_raw, league)
    if not away or not home:
        return []

    resolved = resolve_home_yes_market_ticker(
        league=league,
        date=day,
        away=away,
        home=home,
        kalshi_public_base=kalshi_public_base,
        session=session,
    )
    if resolved is None:
        return []

    # Fetch event details once to get both market tickers.
    url = f"{kalshi_public_base.rstrip('/')}/events/{resolved.event_ticker}"
    resp = session.get(url, timeout=20)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    data = resp.json()
    markets = data.get("markets") or []
    if not isinstance(markets, list):
        return []

    tickers: Dict[str, str] = {}
    for m in markets:
        if not isinstance(m, dict):
            continue
        t = m.get("ticker")
        if not t:
            continue
        suffix = str(t).split("-")[-1]
        suffix_can = team_mapper.normalize_team_code(suffix, league)
        if suffix_can in {home, away}:
            tickers[suffix_can] = str(t)

    if home not in tickers or away not in tickers:
        return []

    # start time / target ts must be computed by caller; fill placeholders here.
    return [
        MarketRef(
            league=league,
            date=day,
            matchup=f"{away}@{home}",
            event_ticker=resolved.event_ticker,
            market_ticker=tickers[home],
            yes_team=home,
            start_time_utc=dt.datetime.now(dt.timezone.utc),
            target_ts=0,
        ),
        MarketRef(
            league=league,
            date=day,
            matchup=f"{away}@{home}",
            event_ticker=resolved.event_ticker,
            market_ticker=tickers[away],
            yes_team=away,
            start_time_utc=dt.datetime.now(dt.timezone.utc),
            target_ts=0,
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Kalshi YES bid/ask snapshots at T-minus using candlesticks.")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD (inclusive).")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive).")
    ap.add_argument("--minutes-before-start", type=int, default=30, help="Minutes before ESPN start (default: 30).")
    ap.add_argument("--period-interval-min", type=int, default=1, help="Candlestick interval minutes (default: 1).")
    ap.add_argument("--lookback-min", type=int, default=60, help="Candlestick lookback window minutes (default: 60).")
    ap.add_argument(
        "--kalshi-public-base",
        default="https://api.elections.kalshi.com/trade-api/v2",
        help="Kalshi public base (default: live trade-api/v2).",
    )
    ap.add_argument("--out", help="Optional output CSV path.")
    args = ap.parse_args()

    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date)
    if end < start:
        raise SystemExit("[error] --end-date must be >= --start-date")

    # Use daily ledgers to determine which games exist for each date/league.
    games = load_games(
        daily_dir=DAILY_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        models=["kalshi_mid"],
    )
    if not games:
        raise SystemExit("[error] no games found in daily ledgers for requested window")

    by_day_league: Dict[Tuple[str, str], List[str]] = {}
    for g in games:
        date_iso = g.date.date().isoformat()
        by_day_league.setdefault((date_iso, g.league), []).append(g.matchup)

    session = requests.Session()
    session.headers.update({"User-Agent": "chimera_v2c/export_kalshi_bidask_tminus/1.0"})

    # Cache ESPN start times per (league,date).
    start_times_cache: Dict[Tuple[str, str], Dict[str, dt.datetime]] = {}
    # Cache market refs per (league,date,matchup).
    market_cache: Dict[Tuple[str, str, str], List[MarketRef]] = {}

    # Candlestick fetch batching by (target_ts).
    tasks_by_ts: Dict[int, List[MarketRef]] = {}
    rows: List[Dict[str, object]] = []

    for (date_iso, league), matchups in sorted(by_day_league.items()):
        day = _parse_date(date_iso)
        if league not in SERIES_TICKER_BY_LEAGUE:
            continue
        st_key = (league, date_iso)
        if st_key not in start_times_cache:
            start_times_cache[st_key] = fetch_start_times_by_matchup(league, day)
        starts = start_times_cache[st_key]

        for matchup_raw in sorted(set(matchups)):
            matchup = matchup_raw.strip().upper()
            if "@" not in matchup:
                continue
            away_raw, home_raw = matchup.split("@", 1)
            away = team_mapper.normalize_team_code(away_raw, league) or away_raw
            home = team_mapper.normalize_team_code(home_raw, league) or home_raw
            matchup_can = f"{away}@{home}"

            start_dt = starts.get(matchup_can)
            if start_dt is None:
                continue
            target_dt = start_dt - dt.timedelta(minutes=int(args.minutes_before_start))
            target_ts = _align_ts_to_minute(int(target_dt.timestamp()))

            mk = (league, date_iso, matchup_can)
            if mk not in market_cache:
                try:
                    refs = _event_markets_for_matchup(
                        league=league,
                        day=day,
                        matchup=matchup_can,
                        kalshi_public_base=str(args.kalshi_public_base),
                        session=session,
                    )
                except Exception:
                    refs = []
                market_cache[mk] = refs

            for ref in market_cache[mk]:
                # patch in computed times
                patched = MarketRef(
                    league=ref.league,
                    date=ref.date,
                    matchup=ref.matchup,
                    event_ticker=ref.event_ticker,
                    market_ticker=ref.market_ticker,
                    yes_team=ref.yes_team,
                    start_time_utc=start_dt,
                    target_ts=target_ts,
                )
                tasks_by_ts.setdefault(target_ts, []).append(patched)

    # Execute candlestick pulls
    for target_ts, refs in sorted(tasks_by_ts.items(), key=lambda kv: kv[0]):
        tickers = sorted({r.market_ticker for r in refs if r.market_ticker})
        if not tickers:
            continue
        for chunk in _chunked(tickers, 50):
            try:
                quotes = fetch_candlestick_quotes(
                    market_tickers=chunk,
                    target_ts=int(target_ts),
                    period_interval_min=int(args.period_interval_min),
                    lookback_min=int(args.lookback_min),
                    kalshi_public_base=str(args.kalshi_public_base),
                    session=session,
                )
            except Exception as exc:
                for t in chunk:
                    quotes = {t: {"yes_bid": None, "yes_ask": None, "mid": None, "spread": None, "candle_end_ts": None}}
                    _ = quotes
                # record failures as rows below using error string
                for t in chunk:
                    for r in [x for x in refs if x.market_ticker == t]:
                        rows.append(
                            {
                                "date": r.date.isoformat(),
                                "league": r.league,
                                "matchup": r.matchup,
                                "start_time_utc": r.start_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "minutes_before_start": int(args.minutes_before_start),
                                "target_ts": int(r.target_ts),
                                "target_iso": dt.datetime.fromtimestamp(int(r.target_ts), tz=dt.timezone.utc).strftime(
                                    "%Y-%m-%dT%H:%M:%SZ"
                                ),
                                "event_ticker": r.event_ticker,
                                "market_ticker": r.market_ticker,
                                "yes_team": r.yes_team,
                                "yes_bid_cents": "",
                                "yes_ask_cents": "",
                                "mid": "",
                                "spread": "",
                                "candle_end_ts": "",
                                "ok": 0,
                                "error": str(exc),
                            }
                        )
                continue

            for r in [x for x in refs if x.market_ticker in chunk]:
                q = quotes.get(r.market_ticker) or {}
                ok = 1 if q.get("mid") is not None else 0
                rows.append(
                    {
                        "date": r.date.isoformat(),
                        "league": r.league,
                        "matchup": r.matchup,
                        "start_time_utc": r.start_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "minutes_before_start": int(args.minutes_before_start),
                        "target_ts": int(r.target_ts),
                        "target_iso": dt.datetime.fromtimestamp(int(r.target_ts), tz=dt.timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "event_ticker": r.event_ticker,
                        "market_ticker": r.market_ticker,
                        "yes_team": r.yes_team,
                        "yes_bid_cents": "" if q.get("yes_bid") is None else int(float(q["yes_bid"])),
                        "yes_ask_cents": "" if q.get("yes_ask") is None else int(float(q["yes_ask"])),
                        "mid": "" if q.get("mid") is None else float(q["mid"]),
                        "spread": "" if q.get("spread") is None else float(q["spread"]),
                        "candle_end_ts": "" if q.get("candle_end_ts") is None else int(float(q["candle_end_ts"])),
                        "ok": ok,
                        "error": "",
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "league", "matchup", "yes_team"])

    out_path = Path(args.out) if args.out else OUT_DIR / (
        f"kalshi_bidask_tminus_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}_m{int(args.minutes_before_start)}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[ok] wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()

