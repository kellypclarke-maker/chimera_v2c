#!/usr/bin/env python
"""
Backfill historical Kalshi `kalshi_mid` into daily ledgers using candlesticks (append-only).

This is intended for older games where we did not capture Kalshi mids live.
Kalshi's settled market endpoint often reflects settlement (0/1) and is not a
usable pre-game probability. Instead, this tool queries:
  GET /trade-api/v2/markets/candlesticks
and computes mid = (yes_bid.close + yes_ask.close) / 2 at (or just before) a
target timestamp using the most recent candlestick <= target (short lookback).

By default, the target timestamp is T-30 minutes from ESPN's scheduled game start.

Safety:
- Respects `reports/daily_ledgers/locked/YYYYMMDD.lock` unless --force.
- Fills blanks only (treats `kalshi_mid` == NR as fillable); does not overwrite non-blank `kalshi_mid` unless `--overwrite-existing`.
- Snapshots the ledger to `reports/daily_ledgers/snapshots/` before writing.
- Dry-run by default; pass --apply to write.

Reliability:
- Retries Kalshi 429/5xx responses with backoff.
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from chimera_v2c.lib import espn_schedule, team_mapper
from chimera_v2c.src.ledger.formatting import format_prob_cell
from chimera_v2c.src.ledger.guard import (
    LedgerGuardError,
    compute_append_only_diff,
    load_locked_dates,
    snapshot_file,
)


DAILY_DIR = Path("reports/daily_ledgers")
LOCK_DIR = DAILY_DIR / "locked"
LEDGER_SNAPSHOT_DIR = DAILY_DIR / "snapshots"

SERIES_TICKER_BY_LEAGUE = {
    "nba": "KXNBAGAME",
    "nhl": "KXNHLGAME",
    "nfl": "KXNFLGAME",
}


_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
_NON_ALNUM = re.compile(r"[^A-Z0-9]+")

_RETRY_STATUS = {429, 500, 502, 503, 504}


def _get_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, object]] = None,
    timeout: float = 30.0,
    max_retries: int = 8,
) -> requests.Response:
    last_exc: Optional[BaseException] = None
    last_resp: Optional[requests.Response] = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            last_resp = resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= max_retries - 1:
                raise
            sleep_s = min(30.0, 0.75 * (2**attempt))
            time.sleep(sleep_s + random.uniform(0.0, 0.25))
            continue

        if resp.status_code not in _RETRY_STATUS:
            return resp

        if attempt >= max_retries - 1:
            return resp

        retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after") or ""
        sleep_s: float
        try:
            sleep_s = float(str(retry_after).strip()) if str(retry_after).strip() else min(30.0, 0.75 * (2**attempt))
        except Exception:
            sleep_s = min(30.0, 0.75 * (2**attempt))
        time.sleep(sleep_s + random.uniform(0.0, 0.25))

    if last_resp is not None:
        return last_resp
    if last_exc is not None:
        raise RuntimeError(f"request failed for {url}: {last_exc}") from last_exc
    raise RuntimeError(f"request failed for {url}: unknown error")


def _ledger_path(date_iso: str) -> Path:
    return DAILY_DIR / f"{date_iso.replace('-', '')}_daily_game_ledger.csv"


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
    # ESPN strings often look like "2025-11-20T00:30Z".
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def _kalshi_date_token(date: dt.date) -> str:
    yy = date.strftime("%y")
    mon = _MONTHS[date.month - 1]
    dd = date.strftime("%d")
    return f"{yy}{mon}{dd}"


def _team_tokens(code: str, league: str) -> List[str]:
    if not code:
        return []
    lg = (league or "").lower().strip()
    canonical = team_mapper.normalize_team_code(code, lg) or code
    candidates = team_mapper.get_alias_candidates(canonical, lg) or [canonical]
    tokens: set[str] = set()
    for a in candidates:
        cleaned = _NON_ALNUM.sub("", str(a).upper())
        if 2 <= len(cleaned) <= 4:
            tokens.add(cleaned)

    canonical_clean = _NON_ALNUM.sub("", str(canonical).upper())
    ordered: List[str] = []
    if canonical_clean in tokens:
        ordered.append(canonical_clean)
    ordered.extend(sorted(tokens - {canonical_clean}, key=lambda x: (-len(x), x)))
    return ordered


def _candlestick_mid_prob(candle: Dict[str, object]) -> Optional[float]:
    yes_bid = (candle.get("yes_bid") or {}) if isinstance(candle.get("yes_bid"), dict) else {}
    yes_ask = (candle.get("yes_ask") or {}) if isinstance(candle.get("yes_ask"), dict) else {}
    bid = yes_bid.get("close")
    ask = yes_ask.get("close")
    if bid is None or ask is None:
        return None
    try:
        bid_i = int(bid)
        ask_i = int(ask)
    except Exception:
        return None
    if bid_i < 0 or bid_i > 100 or ask_i < 0 or ask_i > 100:
        return None
    return ((bid_i + ask_i) / 2.0) / 100.0


def _candlestick_bid_ask_close(candle: Dict[str, object]) -> Tuple[Optional[int], Optional[int]]:
    yes_bid = (candle.get("yes_bid") or {}) if isinstance(candle.get("yes_bid"), dict) else {}
    yes_ask = (candle.get("yes_ask") or {}) if isinstance(candle.get("yes_ask"), dict) else {}
    bid = yes_bid.get("close")
    ask = yes_ask.get("close")
    try:
        bid_i = int(bid) if bid is not None else None
    except Exception:
        bid_i = None
    try:
        ask_i = int(ask) if ask is not None else None
    except Exception:
        ask_i = None
    if bid_i is not None and (bid_i < 0 or bid_i > 100):
        bid_i = None
    if ask_i is not None and (ask_i < 0 or ask_i > 100):
        ask_i = None
    return bid_i, ask_i


def _chunked(items: Sequence[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def fetch_start_times_by_matchup(league: str, date: dt.date) -> Dict[str, dt.datetime]:
    sb = espn_schedule.get_scoreboard(league, date)
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


@dataclass(frozen=True)
class ResolvedMarket:
    event_ticker: str
    market_ticker_home_yes: str


def resolve_home_yes_market_ticker(
    *,
    league: str,
    date: dt.date,
    away: str,
    home: str,
    kalshi_public_base: str,
    session: requests.Session,
) -> Optional[ResolvedMarket]:
    lg = league.lower().strip()
    series = SERIES_TICKER_BY_LEAGUE.get(lg)
    if not series:
        return None
    date_token = _kalshi_date_token(date)
    away_can = team_mapper.normalize_team_code(away, lg) or away
    home_can = team_mapper.normalize_team_code(home, lg) or home
    away_tokens = _team_tokens(away_can, lg)
    home_tokens = _team_tokens(home_can, lg)
    if not away_tokens or not home_tokens:
        return None

    for away_tok in away_tokens:
        for home_tok in home_tokens:
            event_ticker = f"{series}-{date_token}{away_tok}{home_tok}"
            url = f"{kalshi_public_base.rstrip('/')}/events/{event_ticker}"
            resp = _get_with_retries(session, url, timeout=20)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            markets = data.get("markets") or []
            if not isinstance(markets, list):
                continue
            for m in markets:
                if not isinstance(m, dict):
                    continue
                ticker = m.get("ticker")
                if not ticker:
                    continue
                suffix = str(ticker).split("-")[-1]
                suffix_can = team_mapper.normalize_team_code(suffix, lg)
                if suffix_can and suffix_can == home_can:
                    return ResolvedMarket(event_ticker=event_ticker, market_ticker_home_yes=str(ticker))
    return None


def fetch_candlestick_mids(
    *,
    market_tickers: Sequence[str],
    target_ts: int,
    period_interval_min: int,
    lookback_min: int = 60,
    kalshi_public_base: str,
    session: requests.Session,
) -> Dict[str, Optional[float]]:
    url = f"{kalshi_public_base.rstrip('/')}/markets/candlesticks"
    out: Dict[str, Optional[float]] = {t: None for t in market_tickers}
    if not market_tickers:
        return out
    target_ts_int = int(target_ts)
    start_ts = target_ts_int - int(lookback_min) * 60
    # Kalshi appears to treat end_ts as exclusive; add one interval so a candle at target_ts can be included.
    end_ts = target_ts_int + int(period_interval_min) * 60
    params = {
        "market_tickers": ",".join(market_tickers),
        "start_ts": int(start_ts),
        "end_ts": int(end_ts),
        "period_interval": int(period_interval_min),
    }
    resp = _get_with_retries(session, url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    for entry in data.get("markets") or []:
        if not isinstance(entry, dict):
            continue
        ticker = entry.get("market_ticker")
        if not ticker:
            continue
        candles = entry.get("candlesticks") or []
        if not isinstance(candles, list) or not candles:
            continue
        best_end_ts: Optional[int] = None
        best_mid: Optional[float] = None
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            try:
                candle_end_ts = int(candle.get("end_period_ts", -1))
            except Exception:
                continue
            if candle_end_ts > target_ts_int:
                continue
            mid = _candlestick_mid_prob(candle)
            if mid is None:
                continue
            if best_end_ts is None or candle_end_ts > best_end_ts:
                best_end_ts = candle_end_ts
                best_mid = mid
        out[str(ticker)] = best_mid
    return out


def fetch_candlestick_quotes(
    *,
    market_tickers: Sequence[str],
    target_ts: int,
    period_interval_min: int,
    lookback_min: int = 60,
    kalshi_public_base: str,
    session: requests.Session,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Fetch bid/ask/mid at the most recent candlestick end_ts <= target_ts.

    Returns mapping market_ticker -> {
      'yes_bid': int cents (or None),
      'yes_ask': int cents (or None),
      'mid': float in [0,1] (or None),
      'spread': float in [0,1] (or None),
      'candle_end_ts': int seconds (or None),
    }
    """
    url = f"{kalshi_public_base.rstrip('/')}/markets/candlesticks"
    out: Dict[str, Dict[str, Optional[float]]] = {t: {"yes_bid": None, "yes_ask": None, "mid": None, "spread": None, "candle_end_ts": None} for t in market_tickers}
    if not market_tickers:
        return out

    target_ts_int = int(target_ts)
    start_ts = target_ts_int - int(lookback_min) * 60
    end_ts = target_ts_int + int(period_interval_min) * 60
    params = {
        "market_tickers": ",".join(market_tickers),
        "start_ts": int(start_ts),
        "end_ts": int(end_ts),
        "period_interval": int(period_interval_min),
    }
    resp = _get_with_retries(session, url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    for entry in data.get("markets") or []:
        if not isinstance(entry, dict):
            continue
        ticker = entry.get("market_ticker")
        if not ticker:
            continue
        candles = entry.get("candlesticks") or []
        if not isinstance(candles, list) or not candles:
            continue

        best_end_ts: Optional[int] = None
        best_bid: Optional[int] = None
        best_ask: Optional[int] = None
        best_mid: Optional[float] = None
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            try:
                candle_end_ts = int(candle.get("end_period_ts", -1))
            except Exception:
                continue
            if candle_end_ts > target_ts_int:
                continue
            mid = _candlestick_mid_prob(candle)
            bid_i, ask_i = _candlestick_bid_ask_close(candle)
            if mid is None or bid_i is None or ask_i is None:
                continue
            if best_end_ts is None or candle_end_ts > best_end_ts:
                best_end_ts = candle_end_ts
                best_bid = bid_i
                best_ask = ask_i
                best_mid = mid

        if best_end_ts is None or best_mid is None or best_bid is None or best_ask is None:
            continue
        spread = (best_ask - best_bid) / 100.0
        out[str(ticker)] = {
            "yes_bid": float(best_bid),
            "yes_ask": float(best_ask),
            "mid": float(best_mid),
            "spread": float(spread),
            "candle_end_ts": float(best_end_ts),
        }
    return out


def _kalshi_mid_is_fillable(val: object) -> bool:
    s = str(val).strip()
    return (not s) or s.upper() == "NR"


def _align_ts_to_minute(ts: int) -> int:
    return int(ts) - (int(ts) % 60)


def _diff_rows_allow_fill_nr_kalshi_mid(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        r = dict(row)
        if str(r.get("kalshi_mid", "")).strip().upper() == "NR":
            r["kalshi_mid"] = ""
        out.append(r)
    return out


def backfill_ledger(
    *,
    ledger_path: Path,
    date: dt.date,
    league_filter: Optional[str],
    minutes_before_start: int,
    period_interval_min: int,
    kalshi_public_base: str,
    overwrite_existing: bool,
    apply: bool,
    force: bool,
) -> Tuple[int, int, int]:
    if not ledger_path.exists():
        return 0, 0, 0
    ymd = ledger_path.name.split("_")[0]
    locked = load_locked_dates(LOCK_DIR)
    if apply and ymd in locked and not force:
        raise SystemExit(f"[error] {ymd} is locked; refusing to modify {ledger_path} (pass --force to override)")

    original = pd.read_csv(ledger_path, dtype=str).fillna("")
    df = original.copy()
    if "kalshi_mid" not in df.columns:
        df["kalshi_mid"] = ""

    leagues_present = sorted({str(x).lower().strip() for x in df.get("league", pd.Series([], dtype=str)).tolist() if str(x).strip()})
    leagues = [league_filter.lower()] if league_filter else leagues_present
    leagues = [lg for lg in leagues if lg in SERIES_TICKER_BY_LEAGUE]
    if not leagues:
        return 0, 0, 0

    session = requests.Session()
    session.headers.update({"User-Agent": "chimera_v2c/backfill_kalshi_mid/1.0"})

    start_times: Dict[str, Dict[str, dt.datetime]] = {}
    for lg in leagues:
        start_times[lg] = fetch_start_times_by_matchup(lg, date)

    market_cache: Dict[Tuple[str, str], Optional[ResolvedMarket]] = {}
    tasks_by_ts: Dict[int, List[Tuple[int, str]]] = {}
    considered_rows = 0

    for idx, row in df.iterrows():
        row_date = str(row.get("date", "")).strip()
        if row_date and row_date != date.isoformat():
            continue
        lg = str(row.get("league", "")).lower().strip()
        if lg not in leagues:
            continue
        matchup_raw = str(row.get("matchup", "")).strip()
        if "@" not in matchup_raw:
            continue

        existing = row.get("kalshi_mid", "")
        if not overwrite_existing and not _kalshi_mid_is_fillable(existing):
            continue

        away_raw, home_raw = [p.strip() for p in matchup_raw.split("@", 1)]
        away = team_mapper.normalize_team_code(away_raw, lg)
        home = team_mapper.normalize_team_code(home_raw, lg)
        if not away or not home or away == home:
            continue
        matchup_key = f"{away}@{home}"
        start_dt = start_times.get(lg, {}).get(matchup_key)
        if start_dt is None:
            continue
        target_dt = start_dt - dt.timedelta(minutes=int(minutes_before_start))
        target_ts = _align_ts_to_minute(int(target_dt.timestamp()))

        cache_key = (lg, matchup_key)
        if cache_key not in market_cache:
            market_cache[cache_key] = resolve_home_yes_market_ticker(
                league=lg,
                date=date,
                away=away,
                home=home,
                kalshi_public_base=kalshi_public_base,
                session=session,
            )
        resolved = market_cache.get(cache_key)
        if resolved is None:
            continue
        considered_rows += 1
        tasks_by_ts.setdefault(target_ts, []).append((int(idx), resolved.market_ticker_home_yes))

    if not tasks_by_ts:
        return 0, 0, considered_rows

    filled = 0
    for target_ts, jobs in sorted(tasks_by_ts.items()):
        tickers = sorted({t for _, t in jobs if t})
        mids: Dict[str, Optional[float]] = {}
        for chunk in _chunked(tickers, 50):
            mids.update(
                fetch_candlestick_mids(
                    market_tickers=chunk,
                    target_ts=target_ts,
                    period_interval_min=period_interval_min,
                    kalshi_public_base=kalshi_public_base,
                    session=session,
                )
            )
        for idx, ticker in jobs:
            mid = mids.get(ticker)
            if mid is None:
                continue
            formatted = format_prob_cell(mid, decimals=2, drop_leading_zero=True)
            if not formatted:
                continue
            df.at[idx, "kalshi_mid"] = formatted
            filled += 1

    if not apply or filled == 0:
        return 0, filled, considered_rows

    try:
        old_rows = original.fillna("").astype(str).to_dict("records")
        new_rows = df.fillna("").astype(str).to_dict("records")
        key_fields = ["date", "league", "matchup"]
        compute_append_only_diff(
            old_rows=_diff_rows_allow_fill_nr_kalshi_mid(old_rows),
            new_rows=_diff_rows_allow_fill_nr_kalshi_mid(new_rows),
            key_fields=key_fields,
            value_fields=[c for c in df.columns if c not in key_fields],
        )
    except LedgerGuardError as exc:
        if not overwrite_existing:
            raise SystemExit(f"[error] append-only guard failed: {exc}") from exc

    snapshot_file(ledger_path, LEDGER_SNAPSHOT_DIR)
    df.to_csv(ledger_path, index=False)
    return 0, filled, considered_rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backfill Kalshi mids using candlesticks at game start (or T-minus). (append-only)",
    )
    ap.add_argument("--date", help="Single date YYYY-MM-DD.")
    ap.add_argument("--start-date", help="Start date YYYY-MM-DD (inclusive).")
    ap.add_argument("--end-date", help="End date YYYY-MM-DD (inclusive).")
    ap.add_argument("--league", help="Optional league filter (nba|nhl|nfl). Default: all present in the ledger.")
    ap.add_argument("--minutes-before-start", type=int, default=30, help="Minutes before ESPN start time (default: 30).")
    ap.add_argument("--period-interval-min", type=int, default=1, help="Candlestick interval in minutes (default: 1).")
    ap.add_argument(
        "--kalshi-public-base",
        default="https://api.elections.kalshi.com/trade-api/v2",
        help="Kalshi public base (default: live trade-api/v2).",
    )
    ap.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite non-blank kalshi_mid cells (dangerous; snapshots before writing).",
    )
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run).")
    ap.add_argument("--force", action="store_true", help="Allow edits to locked ledgers.")
    args = ap.parse_args()

    if args.date and (args.start_date or args.end_date):
        raise SystemExit("[error] pass --date OR --start-date/--end-date, not both")

    if int(args.minutes_before_start) < 0:
        raise SystemExit("[error] --minutes-before-start must be >= 0")
    if int(args.period_interval_min) <= 0:
        raise SystemExit("[error] --period-interval-min must be >= 1")

    if args.date:
        start = end = _parse_date(args.date)
    else:
        if not args.start_date and not args.end_date:
            raise SystemExit("[error] pass --date or a --start-date/--end-date range")
        start = _parse_date(args.start_date or args.end_date)
        end = _parse_date(args.end_date or args.start_date)
        if end < start:
            raise SystemExit("[error] --end-date must be >= --start-date")

    league_filter = (args.league or "").lower().strip() or None
    if league_filter and league_filter not in SERIES_TICKER_BY_LEAGUE:
        raise SystemExit("[error] --league must be nba|nhl|nfl")

    total_files = 0
    total_filled = 0
    total_considered = 0
    skipped_missing = 0

    for day in _iter_dates(start, end):
        total_files += 1
        date_iso = day.isoformat()
        path = _ledger_path(date_iso)
        if not path.exists():
            skipped_missing += 1
            continue
        _, filled, considered = backfill_ledger(
            ledger_path=path,
            date=day,
            league_filter=league_filter,
            minutes_before_start=int(args.minutes_before_start),
            period_interval_min=int(args.period_interval_min),
            kalshi_public_base=str(args.kalshi_public_base),
            overwrite_existing=bool(args.overwrite_existing),
            apply=bool(args.apply),
            force=bool(args.force),
        )
        total_filled += filled
        total_considered += considered
        mode = "APPLY" if args.apply else "DRY-RUN"
        print(f"[{mode}] {path} kalshi_mid_filled={filled} candidates={considered}")

    print(
        f"Processed {total_files} date(s); missing_ledgers={skipped_missing}; candidates={total_considered}; filled={total_filled}"
    )
    if not args.apply:
        print("[dry-run] No files were written. Use --apply to persist fills.")


if __name__ == "__main__":
    main()
