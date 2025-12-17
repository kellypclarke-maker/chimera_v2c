#!/usr/bin/env python
"""
Hand-audit a small random sample of Rule A qualifying games end-to-end (read-only).

This is a strict validator intended to catch mapping/timestamp/label bugs:
  - Daily ledger row (date/league/matchup/outcome + model cells)
  - Kalshi candlestick snapshot (bid/ask/mid at exact target_ts) for both teams
  - ESPN scheduled start time -> target_iso consistency
  - Odds API market_proxy at the same snapshot_iso
  - Canonical specialist reports -> grok/gemini/gpt p_home consistency

It stops on the first discrepancy (non-zero exit code) unless --continue-on-mismatch.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/audit_rule_a_bidask_samples.py \
    --start-date 2025-11-19 --end-date 2025-12-15 \
    --bidask-csv reports/market_snapshots/kalshi_bidask_tminus_20251119_20251215_m30.csv \
    --n 10 --seed 1337
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from chimera_v2c.lib import espn_schedule, team_mapper
from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.lib.odds_history import fetch_books_snapshot
from chimera_v2c.src.ledger.formatting import format_prob_cell
from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.tools.backfill_kalshi_mid_from_candlesticks import fetch_candlestick_quotes
from chimera_v2c.tools.backfill_market_proxy_from_odds_history import normalize_matchup_key
from chimera_v2c.tools.refresh_model_probs_from_canonicals import CANON_ROOTS, parse_canonical, p_home_from


MODELS_CANON = ["grok", "gemini", "gpt"]


@dataclass(frozen=True)
class Snapshot:
    date: str
    league: str
    matchup: str
    minutes_before_start: int
    target_ts: int
    target_iso: str
    start_time_utc: str
    market_ticker: str
    yes_team: str
    yes_bid_cents: int
    yes_ask_cents: int
    mid: float
    spread: float
    candle_end_ts: int


def _parse_matchup(*, league: str, matchup: str) -> Optional[Tuple[str, str]]:
    if "@" not in matchup:
        return None
    away_raw, home_raw = matchup.split("@", 1)
    away = team_mapper.normalize_team_code(away_raw.strip(), league)
    home = team_mapper.normalize_team_code(home_raw.strip(), league)
    if not away or not home:
        return None
    return away, home


def _parse_iso_utc(text: str) -> Optional[datetime]:
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


def _index_canonicals() -> Dict[Tuple[str, str, str, str], float]:
    """
    (date, league, matchup, model) -> p_home float
    """
    out: Dict[Tuple[str, str, str, str], float] = {}
    for root in CANON_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.txt"):
            parsed = parse_canonical(path)
            if not parsed:
                continue
            p = p_home_from(parsed)
            if p is None:
                continue
            out[(parsed.date, parsed.league, parsed.matchup, parsed.model_label)] = float(p)
    return out


def _load_snapshots(path: Path) -> Dict[Tuple[str, str, str, str], Snapshot]:
    df = pd.read_csv(path)
    df = df[df.get("ok", 0) == 1].copy()
    out: Dict[Tuple[str, str, str, str], Snapshot] = {}
    for _, r in df.iterrows():
        try:
            s = Snapshot(
                date=str(r["date"]).strip(),
                league=str(r["league"]).strip().lower(),
                matchup=str(r["matchup"]).strip(),
                minutes_before_start=int(r["minutes_before_start"]),
                target_ts=int(r["target_ts"]),
                target_iso=str(r["target_iso"]).strip(),
                start_time_utc=str(r["start_time_utc"]).strip(),
                market_ticker=str(r["market_ticker"]).strip(),
                yes_team=str(r["yes_team"]).strip().upper(),
                yes_bid_cents=int(r["yes_bid_cents"]),
                yes_ask_cents=int(r["yes_ask_cents"]),
                mid=float(r["mid"]),
                spread=float(r["spread"]),
                candle_end_ts=int(r["candle_end_ts"]),
            )
        except Exception:
            continue
        if not s.date or not s.league or not s.matchup or not s.yes_team or not s.market_ticker:
            continue
        out[(s.date, s.league, s.matchup, s.yes_team)] = s
    return out


def _expected_proxy_from_odds(*, league: str, matchup: str, snapshot_iso: str, api_key: str) -> Optional[float]:
    books_map = fetch_books_snapshot(league=league, snapshot_iso=snapshot_iso, api_key=api_key)
    norm_key = normalize_matchup_key(matchup, league)
    if not norm_key:
        return None
    snap = books_map.get(norm_key)
    if not snap:
        return None
    proxy = snap.get("market_proxy")
    if proxy is None:
        return None
    try:
        return float(proxy)
    except (TypeError, ValueError):
        return None


def _qualifying_games(
    games: Iterable[GameRow],
    *,
    snapshots: Dict[Tuple[str, str, str, str], Snapshot],
    canon_idx: Dict[Tuple[str, str, str, str], float],
) -> List[GameRow]:
    out: List[GameRow] = []
    for g in games:
        if g.home_win is None or g.home_win == 0.5:
            continue
        parsed = _parse_matchup(league=g.league, matchup=g.matchup)
        if parsed is None:
            continue
        away, home = parsed
        date = g.date.strftime("%Y-%m-%d")
        home_snap = snapshots.get((date, g.league, f"{away}@{home}", home))
        away_snap = snapshots.get((date, g.league, f"{away}@{home}", away))
        if not home_snap or not away_snap:
            continue
        if float(home_snap.mid) <= 0.5:
            continue
        # Ensure at least one canonical exists for this game to audit specialist parsing.
        if not any((date, g.league, f"{away}@{home}", m) in canon_idx for m in MODELS_CANON):
            continue
        out.append(g)
    return out


def _mismatch(msg: str) -> None:
    raise AssertionError(msg)


def audit_game(
    g: GameRow,
    *,
    snapshots: Dict[Tuple[str, str, str, str], Snapshot],
    canon_idx: Dict[Tuple[str, str, str, str], float],
    kalshi_public_base: str,
    odds_api_key: str,
    session: requests.Session,
) -> None:
    parsed = _parse_matchup(league=g.league, matchup=g.matchup)
    if parsed is None:
        _mismatch(f"bad matchup parse: league={g.league} matchup={g.matchup!r}")
    away, home = parsed
    date = g.date.strftime("%Y-%m-%d")
    matchup = f"{away}@{home}"

    home_snap = snapshots.get((date, g.league, matchup, home))
    away_snap = snapshots.get((date, g.league, matchup, away))
    if home_snap is None or away_snap is None:
        _mismatch(f"missing snapshot rows: {date} {g.league} {matchup}")

    # 1) ESPN start time consistency
    sb = espn_schedule.get_scoreboard(g.league, g.date.date())
    start_dt: Optional[datetime] = None
    for event in sb.get("events", []) or []:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        competitors = comp.get("competitors") or []
        home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home_c or not away_c:
            continue

        def _abbr(team: Dict[str, object]) -> str:
            team_info = team.get("team") if isinstance(team.get("team"), dict) else {}
            return str(team_info.get("abbreviation") or "").upper()

        away_code = team_mapper.normalize_team_code(_abbr(away_c), g.league)
        home_code = team_mapper.normalize_team_code(_abbr(home_c), g.league)
        if away_code == away and home_code == home:
            start_dt = _parse_iso_utc(str(event.get("date") or comp.get("date") or ""))
            break

    if start_dt is None:
        _mismatch(f"ESPN start time not found: {date} {g.league} {matchup}")

    snap_start = _parse_iso_utc(home_snap.start_time_utc)
    if snap_start is None:
        _mismatch(f"snapshot start_time_utc unparsable: {home_snap.start_time_utc!r} ({date} {g.league} {matchup})")
    if abs((snap_start - start_dt).total_seconds()) > 61:
        _mismatch(f"ESPN vs snapshot start mismatch >61s: espn={start_dt} snapshot={snap_start} ({date} {g.league} {matchup})")

    target_iso = _parse_iso_utc(home_snap.target_iso)
    if target_iso is None:
        _mismatch(f"snapshot target_iso unparsable: {home_snap.target_iso!r} ({date} {g.league} {matchup})")
    expected_target_dt = start_dt.replace(second=0, microsecond=0) - pd.Timedelta(minutes=int(home_snap.minutes_before_start))
    if abs((target_iso - expected_target_dt).total_seconds()) > 61:
        _mismatch(f"target_iso mismatch >61s: expected={expected_target_dt} got={target_iso} ({date} {g.league} {matchup})")

    # 2) Kalshi candlestick quote verification (both teams)
    quotes = fetch_candlestick_quotes(
        market_tickers=[home_snap.market_ticker, away_snap.market_ticker],
        target_ts=int(home_snap.target_ts),
        period_interval_min=1,
        lookback_min=90,
        kalshi_public_base=kalshi_public_base,
        session=session,
    )
    for snap in (home_snap, away_snap):
        q = quotes.get(snap.market_ticker) or {}
        bid = q.get("yes_bid")
        ask = q.get("yes_ask")
        mid = q.get("mid")
        end_ts = q.get("candle_end_ts")
        if bid is None or ask is None or mid is None or end_ts is None:
            _mismatch(f"missing kalshi quote for {snap.market_ticker} ({date} {g.league} {matchup})")
        if int(round(float(bid))) != int(snap.yes_bid_cents):
            _mismatch(f"kalshi bid cents mismatch {snap.market_ticker}: snapshot={snap.yes_bid_cents} api={int(round(float(bid)))}")
        if int(round(float(ask))) != int(snap.yes_ask_cents):
            _mismatch(f"kalshi ask cents mismatch {snap.market_ticker}: snapshot={snap.yes_ask_cents} api={int(round(float(ask)))}")
        if abs(float(mid) - float(snap.mid)) > 1e-9:
            _mismatch(f"kalshi mid mismatch {snap.market_ticker}: snapshot={snap.mid} api={mid}")
        if int(round(float(end_ts))) != int(snap.candle_end_ts):
            _mismatch(f"kalshi candle_end_ts mismatch {snap.market_ticker}: snapshot={snap.candle_end_ts} api={int(round(float(end_ts)))}")

    mid_home = float(home_snap.mid)
    if mid_home <= 0.5:
        _mismatch(f"home mid unexpectedly not >0.5: {mid_home} ({date} {g.league} {matchup})")

    # 3) Ledger kalshi_mid consistency (formatted to 2dp)
    ledger_mid = g.kalshi_mid
    if ledger_mid is None:
        _mismatch(f"ledger kalshi_mid missing: {date} {g.league} {matchup}")
    expected_mid_cell = format_prob_cell(mid_home, decimals=2, drop_leading_zero=True)
    actual_mid_cell = format_prob_cell(float(ledger_mid), decimals=2, drop_leading_zero=True)
    if expected_mid_cell != actual_mid_cell:
        _mismatch(f"ledger kalshi_mid mismatch: ledger={actual_mid_cell} expected_from_snapshot={expected_mid_cell} ({date} {g.league} {matchup})")

    # 4) Odds API market_proxy consistency at same snapshot_iso (formatted to 2dp)
    proxy = _expected_proxy_from_odds(league=g.league, matchup=matchup, snapshot_iso=home_snap.target_iso, api_key=odds_api_key)
    if proxy is None:
        _mismatch(f"could not compute market_proxy from Odds API: {date} {g.league} {matchup} iso={home_snap.target_iso}")
    expected_proxy_cell = format_prob_cell(proxy, decimals=2, drop_leading_zero=True)
    ledger_proxy = g.probs.get("market_proxy")
    if ledger_proxy is None:
        _mismatch(f"ledger market_proxy missing: {date} {g.league} {matchup}")
    actual_proxy_cell = format_prob_cell(float(ledger_proxy), decimals=2, drop_leading_zero=True)
    if expected_proxy_cell != actual_proxy_cell:
        _mismatch(f"market_proxy mismatch: ledger={actual_proxy_cell} expected_from_odds={expected_proxy_cell} ({date} {g.league} {matchup})")

    # 5) Canonical specialist reports (grok/gemini/gpt)
    for model in MODELS_CANON:
        key = (date, g.league, matchup, model)
        if key not in canon_idx:
            continue
        expected = format_prob_cell(float(canon_idx[key]), decimals=2, drop_leading_zero=True)
        val = g.probs.get(model)
        if val is None:
            _mismatch(f"ledger {model} missing but canonical exists: {date} {g.league} {matchup}")
        actual = format_prob_cell(float(val), decimals=2, drop_leading_zero=True)
        if expected != actual:
            _mismatch(f"{model} mismatch: ledger={actual} expected_from_canonical={expected} ({date} {g.league} {matchup})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Hand-audit random Rule A qualifying games end-to-end (read-only).")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive (ledger filename date).")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive (ledger filename date).")
    ap.add_argument("--bidask-csv", required=True, help="Path to exported Kalshi bid/ask snapshot CSV (T-minus).")
    ap.add_argument("--n", type=int, default=10, help="Number of games to audit (default: 10).")
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed (default: 1337).")
    ap.add_argument("--continue-on-mismatch", action="store_true", help="Do not stop at first mismatch (still exits non-zero if any).")
    args = ap.parse_args()

    load_env_from_env_list()
    kalshi_public_base = os.environ.get("KALSHI_PUBLIC_BASE") or os.environ.get("KALSHI_BASE") or os.environ.get("KALSHI_API_BASE")
    if not kalshi_public_base:
        raise SystemExit("[error] missing KALSHI_PUBLIC_BASE (or KALSHI_BASE/KALSHI_API_BASE) in env")
    odds_api_key = (
        os.environ.get("THE_ODDS_API_HISTORY_KEY")
        or os.environ.get("THE_ODDS_API_HISTORY")
        or os.environ.get("THE_ODDS_API_KEY")
        or ""
    )
    if not odds_api_key:
        raise SystemExit("[error] missing THE_ODDS_API_HISTORY_KEY (needed to recompute market_proxy for audit)")

    snapshots = _load_snapshots(Path(args.bidask_csv))
    canon_idx = _index_canonicals()
    games = load_games(daily_dir=LEDGER_DIR, start_date=args.start_date, end_date=args.end_date, models=MODELS_CANON + ["market_proxy", "kalshi_mid"])
    qual = _qualifying_games(games, snapshots=snapshots, canon_idx=canon_idx)
    if len(qual) < args.n:
        raise SystemExit(f"[error] only {len(qual)} qualifying games available; requested n={args.n}")

    rng = random.Random(int(args.seed))
    sample = rng.sample(qual, k=int(args.n))

    session = requests.Session()
    session.headers.update({"User-Agent": "chimera_v2c/audit_rule_a_bidask_samples/1.0"})

    mismatches: List[str] = []
    for i, g in enumerate(sample, start=1):
        try:
            audit_game(
                g,
                snapshots=snapshots,
                canon_idx=canon_idx,
                kalshi_public_base=kalshi_public_base,
                odds_api_key=odds_api_key,
                session=session,
            )
            print(f"[ok] {i}/{len(sample)} {g.date.strftime('%Y-%m-%d')} {g.league} {g.matchup}")
        except AssertionError as e:
            msg = str(e)
            print(f"[MISMATCH] {i}/{len(sample)} {g.date.strftime('%Y-%m-%d')} {g.league} {g.matchup}: {msg}")
            mismatches.append(msg)
            if not args.continue_on_mismatch:
                raise SystemExit(2)

    if mismatches:
        raise SystemExit(2)
    print(f"[ok] audited {len(sample)} games; no mismatches")


if __name__ == "__main__":
    main()
