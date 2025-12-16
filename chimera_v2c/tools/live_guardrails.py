from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure repo imports work even when run directly
import os

sys.path.insert(0, os.getcwd())

from chimera_v2c.lib import kalshi_portfolio
from chimera_v2c.lib import nhl_scoreboard
from chimera_v2c.lib import team_mapper

from chimera_v2c.src import logging as v2c_logging

SENTINEL_PATH = Path("chimera_v2c/data/STOP_TRADING.flag")


def parse_ticker_teams(ticker: str, league: str) -> Optional[Tuple[str, str, str]]:
    """
    Return (home, away, pick) team codes parsed from ticker.
    We assume Kalshi GAME ticker ends with -<PICKTEAM>.
    """
    if not ticker:
        return None
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    pick = parts[-1].upper()
    core = parts[-2] if len(parts) >= 2 else parts[-1]
    teams = team_mapper.match_teams_in_string(core, league)
    if not teams or len(teams) < 2:
        return None
    return teams[0], teams[1], pick


def fetch_scoreboard(league: str, date_str: str) -> Dict:
    league_l = league.lower()
    if league_l == "nba":
        return nhl_scoreboard.fetch_nba_scoreboard(date_str)
    if league_l == "nfl":
        return nhl_scoreboard.fetch_nfl_scoreboard(date_str)
    return nhl_scoreboard.fetch_nhl_scoreboard(date_str)


def score_for_team(sb: Dict, team_code: str) -> Optional[int]:
    games = sb.get("games") or []
    for g in games:
        teams = g.get("teams") or {}
        home = team_mapper.normalize_team_code(teams.get("home", {}).get("alias"), g.get("league", "nba"))
        away = team_mapper.normalize_team_code(teams.get("away", {}).get("alias"), g.get("league", "nba"))
        if home == team_code:
            return g.get("scores", {}).get("home")
        if away == team_code:
            return g.get("scores", {}).get("away")
    return None


def trigger_guardrail(pick: str, opp: str, sb: Dict, margin_thresh: int) -> bool:
    s_pick = score_for_team(sb, pick)
    s_opp = score_for_team(sb, opp)
    if s_pick is None or s_opp is None:
        return False
    if s_opp - s_pick >= margin_thresh:
        return True
    return False


def live_prob_from_score(pick: str, opp: str, sb: Dict, scale: float = 10.0) -> Optional[float]:
    """Heuristic live win prob based on score margin."""
    s_pick = score_for_team(sb, pick)
    s_opp = score_for_team(sb, opp)
    if s_pick is None or s_opp is None:
        return None
    diff = (s_pick or 0) - (s_opp or 0)
    # Logistic on margin: diff ~0 => 0.5; diff ~ +10 => ~0.73
    try:
        import math
        return 1.0 / (1.0 + math.exp(-diff / max(scale, 1.0)))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Live guardrail monitor: halts on thesis breakers.")
    ap.add_argument("--league", default="nba")
    ap.add_argument("--date", help="YYYY-MM-DD (default today)")
    ap.add_argument("--margin", type=int, default=10, help="Trailing margin trigger (points/goals)")
    ap.add_argument("--interval", type=int, default=60, help="Poll interval seconds")
    ap.add_argument("--once", action="store_true", help="Run once and exit")
    ap.add_argument("--no-sentinel", action="store_true", help="Do not write STOP_TRADING.flag (log only)")
    ap.add_argument("--prob-thresh", type=float, default=0.2, help="Trigger if live prob for pick falls below this")
    args = ap.parse_args()

    league = args.league.lower()
    sb_date = args.date or "today"

    while True:
        pf = kalshi_portfolio.load_portfolio()
        positions = pf.get("positions") or []
        sb = fetch_scoreboard(league, sb_date)
        triggered_rows = []
        for pos in positions:
            ticker = pos.get("ticker") or ""
            parsed = parse_ticker_teams(ticker, league)
            if not parsed:
                continue
            home, away, pick = parsed
            opp = away if pick == home else home
            margin_trip = trigger_guardrail(pick, opp, sb, args.margin)
            live_prob = live_prob_from_score(pick, opp, sb)
            prob_trip = live_prob is not None and live_prob <= args.prob_thresh
            if margin_trip or prob_trip:
                reason = []
                if margin_trip:
                    reason.append(f"trailing_by_{args.margin}+")
                if prob_trip:
                    reason.append(f"live_prob_{live_prob:.2f}_<=_{args.prob_thresh}")
                msg = f"{ticker} pick {pick} " + " & ".join(reason)
                print(f"[GUARDRAIL] {msg}")
                triggered_rows.append(
                    {
                        "date": sb_date,
                        "ticker": ticker,
                        "side": "yes",
                        "count": pos.get("count"),
                        "price_cents": pos.get("price"),
                        "edge": "",
                        "stake_fraction": "",
                        "status": "halt",
                        "message": msg,
                    }
                )
        if triggered_rows:
            if not args.no_sentinel:
                SENTINEL_PATH.write_text("halt", encoding="utf-8")
            v2c_logging.append_log(triggered_rows)
            break
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
