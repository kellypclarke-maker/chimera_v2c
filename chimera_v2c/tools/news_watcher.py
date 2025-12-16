"""
Lightweight news watcher to update injury adjustments and per-game halts.

This is keyword-based (no LLM by default) to stay $0 cost. It scans a text feed
(stdin or file) for patterns like "<TEAM> out" and writes Elo-style deltas into
chimera_v2c/data/injury_adjustments.json, and optionally halts games/tickers.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/news_watcher.py --input news.txt --league nba --date YYYY-MM-DD
  # Or stream (tail -f) piped into stdin:
  tail -f news.txt | PYTHONPATH=. python chimera_v2c/tools/news_watcher.py --league nba --date YYYY-MM-DD
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

from chimera_v2c.lib import team_mapper
from chimera_v2c.src import sentinel

INJURY_PATH = Path("chimera_v2c/data/injury_adjustments.json")


def load_injuries() -> Dict:
    if not INJURY_PATH.exists():
        return {}
    try:
        return json.loads(INJURY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_injuries(data: Dict) -> None:
    INJURY_PATH.parent.mkdir(parents=True, exist_ok=True)
    INJURY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_news(text: str, league: str, date_str: str, impact_map: Dict[str, float]) -> List[Dict]:
    """
    Very simple keyword matcher: looks for "<TEAM> out" or "<TEAM> doubtful".
    """
    updates: List[Dict] = []
    league = league.lower()
    patterns = []
    for team in team_mapper.LEAGUE_ALIASES.get(league, []):
        pat = rf"(?i)\b{re.escape(team)}\b\s+(out|doubtful)"
        patterns.append((team, re.compile(pat)))
    for line in text.splitlines():
        for team, pat in patterns:
            m = pat.search(line)
            if not m:
                continue
            status = m.group(1).lower()
            code = team_mapper.normalize_team_code(team, league) or team.upper()
            delta = impact_map.get(status, -3.0 if status == "out" else -1.5)
            updates.append({"team": code, "status": status, "delta": delta})
    return updates


def apply_updates(updates: List[Dict], league: str, date_str: str, halt: bool) -> None:
    if not updates:
        return
    data = load_injuries()
    league_key = league.upper()
    data.setdefault(league_key, {})
    data[league_key].setdefault(date_str, {})
    for u in updates:
        data[league_key][date_str][u["team"]] = u["delta"]
        if halt:
            sentinel.halt_game(u["team"], f"{u['status']} news")
    save_injuries(data)


def main() -> None:
    ap = argparse.ArgumentParser(description="Keyword-based news watcher for injury updates.")
    ap.add_argument("--input", help="Path to news text file (if omitted, read stdin)")
    ap.add_argument("--league", default="nba")
    ap.add_argument("--date", help="Game date YYYY-MM-DD (default: today)")
    ap.add_argument("--halt", action="store_true", help="Halt games/tickers for teams with negative news")
    args = ap.parse_args()

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    else:
        text = sys.stdin.read()
    from datetime import datetime

    date_str = args.date or datetime.utcnow().date().isoformat()

    # Simple impact map; adjust as needed
    impact_map = {"out": -5.0, "doubtful": -2.5}

    updates = parse_news(text, args.league, date_str, impact_map)
    if not updates:
        print("[info] no news updates parsed.")
        return
    apply_updates(updates, args.league, date_str, halt=args.halt)
    print(f"[info] applied {len(updates)} injury updates to {INJURY_PATH}")


if __name__ == "__main__":
    main()
