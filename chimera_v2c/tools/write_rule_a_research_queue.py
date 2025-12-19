#!/usr/bin/env python
"""
Write a pre-filtered "research queue" for Rule A (taker) so operators only run LLMs
on games we might trade.

Definition (default):
  - Include only games where Kalshi favors the HOME team (home YES mid > 0.50).

This tool does NOT call any LLM APIs. It only:
  - reads the daily ledger to get today's slate (matchup keys),
  - fetches Kalshi public quotes (bid/ask) for the league series,
  - fetches ESPN start times,
  - writes a queue CSV + a copy/paste prompt pack for manual LLM runs.

Safety:
  - Read-only on daily ledgers.
  - Writes only under reports/execution_logs/ by default.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from chimera_v2c.lib import nhl_scoreboard, team_mapper
from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.src import market_linker


SERIES_TICKER_BY_LEAGUE = {
    "nba": "KXNBAGAME",
    "nhl": "KXNHLGAME",
    "nfl": "KXNFLGAME",
}


def _normalize_league(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"nba", "nhl", "nfl"}:
        return v
    raise SystemExit("[error] --league must be one of: nba, nhl, nfl")


def _parse_iso_utc(text: str) -> Optional[datetime]:
    s = (text or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _clamp_price(p: float) -> float:
    return max(0.01, min(0.99, float(p)))


def _fetch_start_times_by_matchup(league: str, date_iso: str) -> Dict[str, datetime]:
    fetcher = {
        "nba": nhl_scoreboard.fetch_nba_scoreboard,
        "nhl": nhl_scoreboard.fetch_nhl_scoreboard,
        "nfl": nhl_scoreboard.fetch_nfl_scoreboard,
    }.get(league)
    if fetcher is None:
        return {}

    sb = fetcher(date_iso)
    if sb.get("status") not in {"ok", "empty"}:
        return {}

    out: Dict[str, datetime] = {}
    for g in sb.get("games") or []:
        teams = g.get("teams") or {}
        away_alias = ((teams.get("away") or {}).get("alias") or "").strip()
        home_alias = ((teams.get("home") or {}).get("alias") or "").strip()
        away = team_mapper.normalize_team_code(away_alias, league)
        home = team_mapper.normalize_team_code(home_alias, league)
        if not away or not home:
            continue
        start = _parse_iso_utc(str(g.get("start_time") or ""))
        if start is None:
            continue
        out[f"{away}@{home}"] = start
    return out


def _read_matchups_from_daily_ledger(*, date_iso: str, league: str) -> List[str]:
    ymd = date_iso.replace("-", "")
    path = Path("reports/daily_ledgers") / f"{ymd}_daily_game_ledger.csv"
    if not path.exists():
        raise SystemExit(f"[error] missing daily ledger: {path}")
    rows: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("date") or "").strip() != date_iso:
                continue
            if str(r.get("league") or "").strip().lower() != league:
                continue
            m = str(r.get("matchup") or "").strip().upper()
            if "@" not in m:
                continue
            rows.append(m)
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for m in rows:
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def _markets_by_matchup(markets: Sequence[market_linker.MarketQuote]) -> Dict[str, Dict[str, market_linker.MarketQuote]]:
    out: Dict[str, Dict[str, market_linker.MarketQuote]] = {}
    for mq in markets:
        if not mq.away or not mq.home:
            continue
        yes_team = (mq.yes_team or "").strip().upper()
        if not yes_team:
            continue
        out.setdefault(f"{mq.away}@{mq.home}", {})[yes_team] = mq
    return out


def _fmt_price(p: Optional[float]) -> str:
    return "" if p is None else f"{float(p):.6f}"


@dataclass(frozen=True)
class QueueRow:
    date: str
    league: str
    matchup: str
    away: str
    home: str
    event_ticker: str
    market_ticker_home: str
    market_ticker_away: str
    start_time_utc: str
    minutes_to_start: str
    home_yes_bid_cents: str
    home_yes_ask_cents: str
    mid_home: str
    away_yes_bid_cents: str
    away_yes_ask_cents: str
    away_yes_ask_dollars: str
    slippage_cents: str
    price_away_taker_assumed: str

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def build_queue_rows(
    *,
    date_iso: str,
    league: str,
    matchups: Sequence[str],
    markets: Sequence[market_linker.MarketQuote],
    slippage_cents: int,
    min_minutes_to_start: Optional[float],
    max_minutes_to_start: Optional[float],
    home_mid_min: float,
) -> List[QueueRow]:
    starts = _fetch_start_times_by_matchup(league, date_iso)
    by_matchup = _markets_by_matchup(markets)
    now = datetime.now(timezone.utc)
    slip = float(int(slippage_cents)) / 100.0

    rows: List[QueueRow] = []
    for m in matchups:
        away_raw, home_raw = m.split("@", 1)
        away = away_raw.strip().upper()
        home = home_raw.strip().upper()

        quotes = by_matchup.get(m)
        if not quotes:
            continue
        home_q = quotes.get(home)
        away_q = quotes.get(away)
        if home_q is None or away_q is None:
            continue
        if home_q.mid is None:
            continue

        mid_home = float(home_q.mid)
        if mid_home <= float(home_mid_min):
            continue

        start = starts.get(m)
        minutes_to_start = ""
        if start is not None:
            mts = (start - now).total_seconds() / 60.0
            if min_minutes_to_start is not None and mts < float(min_minutes_to_start):
                continue
            if max_minutes_to_start is not None and mts > float(max_minutes_to_start):
                continue
            minutes_to_start = f"{mts:.2f}"

        away_ask = away_q.yes_ask
        away_ask_dollars = "" if away_ask is None else f"{float(away_ask) / 100.0:.6f}"
        price_away_assumed = ""
        if away_ask is not None:
            price_away_assumed = f"{_clamp_price(float(away_ask) / 100.0 + slip):.6f}"

        rows.append(
            QueueRow(
                date=date_iso,
                league=league,
                matchup=m,
                away=away,
                home=home,
                event_ticker=str(home_q.event_ticker or away_q.event_ticker or ""),
                market_ticker_home=str(home_q.ticker),
                market_ticker_away=str(away_q.ticker),
                start_time_utc=str(start.isoformat().replace("+00:00", "Z") if start is not None else ""),
                minutes_to_start=minutes_to_start,
                home_yes_bid_cents="" if home_q.yes_bid is None else str(int(home_q.yes_bid)),
                home_yes_ask_cents="" if home_q.yes_ask is None else str(int(home_q.yes_ask)),
                mid_home=_fmt_price(mid_home),
                away_yes_bid_cents="" if away_q.yes_bid is None else str(int(away_q.yes_bid)),
                away_yes_ask_cents="" if away_q.yes_ask is None else str(int(away_q.yes_ask)),
                away_yes_ask_dollars=away_ask_dollars,
                slippage_cents=str(int(slippage_cents)),
                price_away_taker_assumed=price_away_assumed,
            )
        )

    rows.sort(key=lambda r: (r.start_time_utc or "9999", r.matchup))
    return rows


def build_prompt_pack_markdown(*, rows: Sequence[QueueRow]) -> str:
    lines: List[str] = []
    lines.append("# Rule A Research Queue (Manual LLM Runs)")
    lines.append("")
    lines.append("Instructions:")
    lines.append("- You are evaluating ONLY these games (Kalshi home-favored right now).")
    lines.append("- Return ONLY a single JSON object per game with keys: matchup, p_home, confidence.")
    lines.append("- p_home is the probability the HOME team wins (0..1).")
    lines.append("- confidence is 0..1 and should reflect how sure you are.")
    lines.append("")
    for r in rows:
        lines.append(f"## {r.league.upper()} {r.matchup}")
        lines.append(f"- Kalshi home mid: `{r.mid_home}`")
        lines.append(f"- Away YES ask (live): `{r.away_yes_ask_dollars}`")
        lines.append(f"- Assumed taker entry (ask + slippage): `{r.price_away_taker_assumed}`")
        lines.append(f"- Ticker (away YES): `{r.market_ticker_away}`")
        lines.append("")
        lines.append("Return ONLY this JSON (one line):")
        lines.append("```")
        lines.append(f'{{"matchup":"{r.matchup}","p_home":0.50,"confidence":0.50}}')
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_csv(path: Path, rows: Sequence[QueueRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = list(QueueRow.__annotations__.keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].as_dict().keys()))
        w.writeheader()
        w.writerows([r.as_dict() for r in rows])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Write a Rule A research queue (no LLM calls).")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (ledger date).")
    ap.add_argument("--league", required=True, help="nba|nhl|nfl")
    ap.add_argument("--slippage-cents", type=int, default=1, help="Slippage cents added to away ask for assumed taker entry (default: 1).")
    ap.add_argument("--home-mid-min", type=float, default=0.50, help="Minimum home mid to include (default: 0.50).")
    ap.add_argument("--min-minutes-to-start", type=float, default=None, help="If set, include only games with minutes_to_start >= this value.")
    ap.add_argument("--max-minutes-to-start", type=float, default=None, help="If set, include only games with minutes_to_start <= this value.")
    ap.add_argument(
        "--kalshi-public-base",
        default="https://api.elections.kalshi.com/trade-api/v2",
        help="Kalshi public base (default: live trade-api/v2).",
    )
    ap.add_argument(
        "--kalshi-status",
        default="open",
        choices=["open", "settled"],
        help="Kalshi market status to use for quotes (default: open).",
    )
    ap.add_argument("--out-dir", default="reports/execution_logs/rule_a_research_queue", help="Output directory (default under reports/execution_logs/).")
    ap.add_argument("--write-prompts", action="store_true", help="Also write a markdown prompt pack next to the CSV.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    load_env_from_env_list()

    league = _normalize_league(args.league)
    date_iso = str(args.date).strip()
    if not date_iso:
        raise SystemExit("[error] missing --date")

    os.environ["KALSHI_PUBLIC_BASE"] = str(args.kalshi_public_base).strip()
    matchups = _read_matchups_from_daily_ledger(date_iso=date_iso, league=league)
    if not matchups:
        raise SystemExit(f"[error] no slate games found in daily ledger for {league} {date_iso}")

    series = SERIES_TICKER_BY_LEAGUE.get(league)
    if not series:
        raise SystemExit(f"[error] missing series ticker mapping for league: {league}")

    markets = market_linker.fetch_markets(
        league=league,
        series_ticker=series,
        use_private=False,
        status=str(args.kalshi_status),
    )

    rows = build_queue_rows(
        date_iso=date_iso,
        league=league,
        matchups=matchups,
        markets=markets,
        slippage_cents=int(args.slippage_cents),
        min_minutes_to_start=args.min_minutes_to_start,
        max_minutes_to_start=args.max_minutes_to_start,
        home_mid_min=float(args.home_mid_min),
    )

    out_dir = Path(str(args.out_dir)) / date_iso.replace("-", "")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    csv_path = out_dir / f"rule_a_research_queue_{league}_{ts}.csv"
    _write_csv(csv_path, rows)
    print(f"[ok] wrote {len(rows)} games -> {csv_path}")

    if bool(args.write_prompts):
        md_path = out_dir / f"rule_a_research_prompts_{league}_{ts}.md"
        md_path.write_text(build_prompt_pack_markdown(rows=rows), encoding="utf-8")
        print(f"[ok] wrote prompts -> {md_path}")


if __name__ == "__main__":
    main()
