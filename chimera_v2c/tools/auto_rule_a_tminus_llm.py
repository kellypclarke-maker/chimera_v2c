#!/usr/bin/env python
"""
Automated T-minus LLM runner for the Rule A (taker) workflow.

Goal:
  - At (roughly) T-30 before start, run LLMs ONLY for games where Kalshi currently
    favors the HOME team (home mid > 0.50).
  - Write HELIOS-formatted raw specialist report files so the existing ingestion
    tool can safely:
      - write canonical per-game reports, and
      - fill blank/NR daily-ledger model cells (append-only).

Scheduling:
  - This tool is designed to be run frequently (e.g. every 60s) by an external scheduler
    (cron/Task Scheduler). It filters to a narrow minutes_to_start window to behave as
    a trigger around T-30.

Safety:
  - Default is dry-run (no ledger writes).
  - When `--apply` is passed, it calls `ingest_raw_specialist_reports_v2c.py --apply`
    on the generated raw file(s). That ingester is append-only and respects lockfiles
    unless `--force` is explicitly passed (this tool does not pass --force).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

try:
    from google import genai
except Exception:  # pragma: no cover - optional dependency
    genai = None  # type: ignore

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - python<3.9
    ZoneInfo = None  # type: ignore

from chimera_v2c.lib import nhl_scoreboard, team_mapper
from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.src import market_linker


SERIES_TICKER_BY_LEAGUE = {
    "nba": "KXNBAGAME",
    "nhl": "KXNHLGAME",
    "nfl": "KXNFLGAME",
}

MISSING_SENTINEL = "NR"


def _pacific_today_iso() -> str:
    if ZoneInfo is None:
        return datetime.now().date().isoformat()
    return datetime.now(tz=ZoneInfo("America/Los_Angeles")).date().isoformat()


def _normalize_leagues(value: str) -> List[str]:
    parts = [p.strip().lower() for p in str(value or "").split(",") if p.strip()]
    out: List[str] = []
    for p in parts:
        if p not in {"nba", "nhl", "nfl"}:
            raise SystemExit("[error] --leagues must be a comma list of nba,nhl,nfl")
        out.append(p)
    return out


def _normalize_models(value: str) -> List[str]:
    parts = [p.strip().lower() for p in str(value or "").split(",") if p.strip()]
    out: List[str] = []
    for p in parts:
        if p not in {"gpt", "gemini"}:
            raise SystemExit("[error] --models must be a comma list of gpt,gemini")
        out.append(p)
    return out


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


def _read_text_max(path: Path, max_chars: int) -> str:
    if not path.exists() or max_chars <= 0:
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]...\n"


def _daily_ledger_path(date_iso: str) -> Path:
    return Path("reports/daily_ledgers") / f"{date_iso.replace('-', '')}_daily_game_ledger.csv"


def _load_daily_ledger_rows(*, date_iso: str, league: str) -> List[Dict[str, str]]:
    path = _daily_ledger_path(date_iso)
    if not path.exists():
        return []
    import csv

    out: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("date") or "").strip() != date_iso:
                continue
            if str(r.get("league") or "").strip().lower() != league:
                continue
            m = str(r.get("matchup") or "").strip().upper()
            if "@" not in m:
                continue
            out.append({k: str(v) for k, v in r.items()})
    return out


def _prob_cell_is_missing(value: str) -> bool:
    s = str(value or "").strip()
    if not s:
        return True
    if s.upper() == MISSING_SENTINEL:
        return True
    return False


def _parse_prob_cell(value: str) -> Optional[float]:
    s = str(value or "").strip()
    if not s or s.upper() == MISSING_SENTINEL:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _format_prob_cell(p: float) -> str:
    p = max(0.0, min(1.0, float(p)))
    s = f"{p:.2f}"
    if s.startswith("0"):
        s = s[1:]
    return s


def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _coerce_json_object(value: object) -> Optional[Dict[str, object]]:
    """
    Best-effort: coerce a parsed JSON payload into a dict.

    Some providers occasionally wrap an object in a single-element list; accept that
    without crashing the poller.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return None


@dataclass(frozen=True)
class CandidateGame:
    league: str
    date_iso: str
    matchup: str
    away: str
    home: str
    start_time_utc: Optional[str]
    minutes_to_start: Optional[float]
    mid_home: float
    away_yes_ask_cents: Optional[int]
    market_ticker_away: str
    market_ticker_home: str
    v2c_p_home: Optional[float]
    market_proxy_p_home: Optional[float]
    moneypuck_p_home: Optional[float]


def _select_candidate_games(
    *,
    date_iso: str,
    league: str,
    min_minutes_to_start: float,
    max_minutes_to_start: float,
    home_mid_min: float,
    markets: Sequence[market_linker.MarketQuote],
) -> List[CandidateGame]:
    starts = _fetch_start_times_by_matchup(league, date_iso)
    by_matchup = _markets_by_matchup(markets)
    now = datetime.now(timezone.utc)

    rows = _load_daily_ledger_rows(date_iso=date_iso, league=league)
    out: List[CandidateGame] = []
    for r in rows:
        matchup = str(r.get("matchup") or "").strip().upper()
        if "@" not in matchup:
            continue
        away, home = matchup.split("@", 1)
        away = away.strip().upper()
        home = home.strip().upper()

        start = starts.get(matchup)
        if start is None:
            continue
        mts = (start - now).total_seconds() / 60.0
        if mts < float(min_minutes_to_start) or mts > float(max_minutes_to_start):
            continue

        quotes = by_matchup.get(matchup)
        if not quotes:
            continue
        home_q = quotes.get(home)
        away_q = quotes.get(away)
        if home_q is None or away_q is None or home_q.mid is None:
            continue

        mid_home = float(home_q.mid)
        if mid_home <= float(home_mid_min):
            continue

        out.append(
            CandidateGame(
                league=league,
                date_iso=date_iso,
                matchup=matchup,
                away=away,
                home=home,
                start_time_utc=start.isoformat().replace("+00:00", "Z"),
                minutes_to_start=float(mts),
                mid_home=mid_home,
                away_yes_ask_cents=away_q.yes_ask,
                market_ticker_away=str(away_q.ticker),
                market_ticker_home=str(home_q.ticker),
                v2c_p_home=_parse_prob_cell(r.get("v2c", "")),
                market_proxy_p_home=_parse_prob_cell(r.get("market_proxy", "")),
                moneypuck_p_home=_parse_prob_cell(r.get("moneypuck", "")),
            )
        )
    out.sort(key=lambda g: (g.start_time_utc or "9999", g.matchup))
    return out


def render_legacy_helios_block(
    *,
    ts_pacific: str,
    league: str,
    matchup: str,
    model_label: str,
    winner: str,
    p_home: float,
    confidence: Optional[float],
) -> str:
    conf_line = "" if confidence is None else f"\nconfidence: {max(0.0, min(1.0, float(confidence))):.2f}"
    return (
        "HELIOS_PREDICTION_HEADER\n"
        f"timestamp: {ts_pacific}\n"
        f"league: {league.upper()}\n"
        f"matchup: {matchup}\n"
        f"model: {model_label}\n"
        f"winner: {winner}\n"
        f"p_home: {max(0.0, min(1.0, float(p_home))):.4f}"
        f"{conf_line}\n"
        "---\n"
    )


def _call_openai_json(*, prompt: str, model: str, temperature: float) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package not installed")
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    # Best-effort strict JSON; not supported by every model.
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a sports betting Specialist. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a sports betting Specialist. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
        )
        return resp.choices[0].message.content or ""


def _call_gemini_json(*, prompt: str, model: str) -> str:
    if genai is None:
        raise RuntimeError("google-genai not installed")
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=model, contents=prompt)
    text = getattr(resp, "text", None)
    if not text:
        raise RuntimeError("empty Gemini response")
    return str(text)


def _build_prompt(*, g: CandidateGame, extra_context: str) -> str:
    ask_dollars = "" if g.away_yes_ask_cents is None else f"{float(g.away_yes_ask_cents)/100.0:.2f}"
    baseline_bits = []
    if g.v2c_p_home is not None:
        baseline_bits.append(f"v2c_p_home={g.v2c_p_home:.3f}")
    if g.market_proxy_p_home is not None:
        baseline_bits.append(f"market_proxy_p_home={g.market_proxy_p_home:.3f}")
    if g.moneypuck_p_home is not None:
        baseline_bits.append(f"moneypuck_p_home={g.moneypuck_p_home:.3f}")
    baselines = ", ".join(baseline_bits) if baseline_bits else "(none)"

    return (
        "Return ONLY JSON with keys: matchup, p_home, confidence, winner.\n"
        "- matchup must be exactly like AWAY@HOME.\n"
        "- p_home is probability HOME wins (0..1).\n"
        "- confidence is 0..1.\n"
        "- winner must be either the home team code or away team code.\n\n"
        f"League: {g.league.upper()}\n"
        f"Date (Pacific semantics): {g.date_iso}\n"
        f"Matchup: {g.matchup}\n"
        f"Start time (UTC): {g.start_time_utc or ''}\n"
        f"Minutes to start (now): {'' if g.minutes_to_start is None else f'{g.minutes_to_start:.2f}'}\n\n"
        f"Kalshi live home mid: {g.mid_home:.3f}\n"
        f"Kalshi away YES ask (cents): {'' if g.away_yes_ask_cents is None else int(g.away_yes_ask_cents)}\n"
        f"Kalshi away YES ask (dollars): {ask_dollars}\n"
        f"Tickers: home_yes={g.market_ticker_home} away_yes={g.market_ticker_away}\n\n"
        f"Existing baselines from our ledger: {baselines}\n\n"
        f"Context:\n{extra_context}\n"
    )


def _load_extra_context(*, league: str, date_iso: str, max_chars: int) -> str:
    ymd = date_iso.replace("-", "")
    packet_dir = Path("reports/llm_packets") / league / ymd

    candidates: List[Path] = []
    # High-signal text digest (if present)
    candidates.append(packet_dir / "news.txt")
    candidates.append(Path("chimera_v2c/data") / f"news_{date_iso}_{league}.txt")
    # Structured packets (if present)
    if packet_dir.exists():
        for p in sorted(packet_dir.glob("*.csv")):
            candidates.append(p)

    parts: List[str] = []
    used = 0
    for p in candidates:
        if used >= max_chars:
            break
        txt = _read_text_max(p, max_chars=max_chars - used)
        if not txt.strip():
            continue
        chunk = f"[{p}]\n{txt.strip()}\n"
        parts.append(chunk)
        used += len(chunk)

    return "\n".join(parts).strip()


def _write_raw_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ingest_raw_dir(raw_dir: Path, *, apply: bool) -> None:
    cmd = [
        sys.executable,
        "chimera_v2c/tools/ingest_raw_specialist_reports_v2c.py",
        "--raw-dir",
        str(raw_dir),
    ]
    cmd.append("--apply" if apply else "--dry-run")
    env = dict(os.environ)
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Auto-run LLMs at T-minus for Rule A (home-favored only).")
    ap.add_argument("--date", default="", help="Slate date YYYY-MM-DD (default: today in America/Los_Angeles).")
    ap.add_argument("--leagues", default="nba,nhl", help="Comma list of leagues (default: nba,nhl).")
    ap.add_argument("--models", default="gpt", help="Comma list of LLMs to run (default: gpt).")
    ap.add_argument("--min-minutes-to-start", type=float, default=29.0, help="Min minutes_to_start filter (default: 29.0).")
    ap.add_argument("--max-minutes-to-start", type=float, default=31.0, help="Max minutes_to_start filter (default: 31.0).")
    ap.add_argument("--home-mid-min", type=float, default=0.50, help="Only consider games with live home mid > this value (default: 0.50).")
    ap.add_argument(
        "--kalshi-public-base",
        default="https://api.elections.kalshi.com/trade-api/v2",
        help="Kalshi public base (default: live trade-api/v2).",
    )
    ap.add_argument("--kalshi-status", default="open", choices=["open", "settled"], help="Kalshi market status (default: open).")
    ap.add_argument("--gpt-model", default="gpt-5.2-pro", help="OpenAI model name (default: gpt-5.2-pro).")
    ap.add_argument(
        "--gpt-model-fallback",
        default="gpt-4o-mini",
        help="Fallback OpenAI model if the primary model errors (default: gpt-4o-mini).",
    )
    ap.add_argument("--gpt-temperature", type=float, default=0.2, help="OpenAI temperature (default: 0.2).")
    ap.add_argument("--gemini-model", default="gemini-3.0-pro", help="Gemini model name (default: gemini-3.0-pro).")
    ap.add_argument(
        "--gemini-model-fallback",
        default="gemini-2.0-flash",
        help="Fallback Gemini model if the primary model errors (default: gemini-2.0-flash).",
    )
    ap.add_argument("--context-max-chars", type=int, default=6000, help="Max chars of local context to include (default: 6000).")
    ap.add_argument(
        "--raw-out-dir",
        default="reports/specialist_reports/raw_auto",
        help="Directory for generated raw files (default: reports/specialist_reports/raw_auto).",
    )
    ap.add_argument("--apply", action="store_true", help="After writing raw file(s), ingest and apply to write canonicals + fill daily ledger blanks.")
    ap.add_argument(
        "--poll-seconds",
        type=int,
        default=0,
        help="If >0, run in a loop sleeping this many seconds between iterations (default: 0, run once).",
    )
    ap.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="If polling, stop after this many iterations (default: 0, run forever).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    load_env_from_env_list()

    poll_seconds = int(args.poll_seconds or 0)
    max_iters = int(args.max_iterations or 0)
    it = 0
    while True:
        it += 1
        date_iso = str(args.date).strip() or _pacific_today_iso()
        leagues = _normalize_leagues(args.leagues)
        models = _normalize_models(args.models)

        os.environ["KALSHI_PUBLIC_BASE"] = str(args.kalshi_public_base).strip()

        ts_now = datetime.now(timezone.utc)
        if ZoneInfo is not None:
            ts_pacific = ts_now.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PST")
        else:
            ts_pacific = ts_now.strftime("%Y-%m-%d %H:%M UTC")

        raw_out_dir = Path(str(args.raw_out_dir))

        wrote_any = False
        for league in leagues:
            series = SERIES_TICKER_BY_LEAGUE.get(league)
            if not series:
                continue
            try:
                date_obj = datetime.fromisoformat(date_iso).date()
            except ValueError:
                raise SystemExit(f"[error] invalid --date (expected YYYY-MM-DD): {date_iso}")

            markets = market_linker.fetch_markets(
                league=league,
                series_ticker=series,
                use_private=False,
                status=str(args.kalshi_status),
                target_date=date_obj,
            )
            candidates = _select_candidate_games(
                date_iso=date_iso,
                league=league,
                min_minutes_to_start=float(args.min_minutes_to_start),
                max_minutes_to_start=float(args.max_minutes_to_start),
                home_mid_min=float(args.home_mid_min),
                markets=markets,
            )
            if not candidates:
                continue

            ledger_rows = _load_daily_ledger_rows(date_iso=date_iso, league=league)
            ledger_by_matchup = {str(r.get("matchup") or "").strip().upper(): r for r in ledger_rows}

            extra_context = _load_extra_context(league=league, date_iso=date_iso, max_chars=int(args.context_max_chars))
            if not extra_context.strip():
                extra_context = "(no local news context found)"

            for model_name in models:
                blocks: List[str] = []
                for g in candidates:
                    existing_row = ledger_by_matchup.get(g.matchup, {})
                    if model_name in existing_row and not _prob_cell_is_missing(str(existing_row.get(model_name) or "")):
                        continue

                    prompt = _build_prompt(g=g, extra_context=extra_context)
                    raw = ""
                    try:
                        if model_name == "gpt":
                            try:
                                raw = _call_openai_json(
                                    prompt=prompt, model=str(args.gpt_model), temperature=float(args.gpt_temperature)
                                )
                                model_label = str(args.gpt_model)
                            except Exception as exc:
                                fb = str(getattr(args, "gpt_model_fallback", "") or "").strip()
                                if fb and fb != str(args.gpt_model).strip():
                                    print(f"[warn] {league} {g.matchup} gpt primary model failed ({exc}); trying fallback={fb}")
                                    raw = _call_openai_json(prompt=prompt, model=fb, temperature=float(args.gpt_temperature))
                                    model_label = fb
                                else:
                                    raise
                        else:
                            try:
                                raw = _call_gemini_json(prompt=prompt, model=str(args.gemini_model))
                                model_label = str(args.gemini_model)
                            except Exception as exc:
                                fb = str(getattr(args, "gemini_model_fallback", "") or "").strip()
                                if fb and fb != str(args.gemini_model).strip():
                                    print(f"[warn] {league} {g.matchup} gemini primary model failed ({exc}); trying fallback={fb}")
                                    raw = _call_gemini_json(prompt=prompt, model=fb)
                                    model_label = fb
                                else:
                                    raise
                    except Exception as exc:
                        print(f"[warn] {league} {g.matchup} {model_name} call failed: {exc}")
                        continue

                    cleaned = _strip_code_fences(raw)
                    try:
                        parsed_raw = json.loads(cleaned)
                    except Exception:
                        print(f"[warn] {league} {g.matchup} {model_name} invalid JSON; skipping")
                        continue

                    parsed = _coerce_json_object(parsed_raw)
                    if parsed is None:
                        print(f"[warn] {league} {g.matchup} {model_name} JSON was not an object; skipping")
                        continue

                    p_home = parsed.get("p_home")
                    conf = parsed.get("confidence")
                    winner = parsed.get("winner")
                    try:
                        p_home_f = float(p_home)
                    except Exception:
                        print(f"[warn] {league} {g.matchup} {model_name} missing p_home; skipping")
                        continue

                    if not (0.0 <= p_home_f <= 1.0):
                        print(f"[warn] {league} {g.matchup} {model_name} p_home out of range; skipping")
                        continue

                    confidence_f: Optional[float] = None
                    if conf is not None:
                        try:
                            confidence_f = float(conf)
                        except Exception:
                            confidence_f = None

                    winner_s = str(winner or "").strip().upper()
                    if winner_s not in {g.home, g.away}:
                        winner_s = g.home if p_home_f >= 0.5 else g.away

                    blocks.append(
                        render_legacy_helios_block(
                            ts_pacific=ts_pacific,
                            league=league,
                            matchup=g.matchup,
                            model_label=model_label,
                            winner=winner_s,
                            p_home=p_home_f,
                            confidence=confidence_f,
                        )
                    )

                if not blocks:
                    continue

                out_path = raw_out_dir / f"auto_{date_iso.replace('-', '')}_{league}_{model_name}.txt"
                _write_raw_file(out_path, "\n".join(blocks).rstrip() + "\n")
                wrote_any = True
                print(f"[ok] wrote raw {model_name} blocks: {out_path}")

            if wrote_any and bool(args.apply):
                _ingest_raw_dir(raw_out_dir, apply=True)

        if not wrote_any:
            print("[info] no qualifying games in the T-minus window (or all model cells already filled).")

        if poll_seconds <= 0:
            return
        if max_iters > 0 and it >= max_iters:
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
