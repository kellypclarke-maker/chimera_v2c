"""
Build NHL Elo-style ratings from the ESPN scoreboard history.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/elo_builder_nhl.py --start 2024-10-01 --end 2025-04-30 --decay 0.99
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from chimera_v2c.lib.nhl_scoreboard import fetch_nhl_scoreboard
from chimera_v2c.lib import team_mapper

OUT_PATH = Path("chimera_v2c/data/team_ratings_nhl.json")


def daterange(start: datetime, end: datetime):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def load_games(start: str, end: str) -> List[Tuple[str, str, str, int, int]]:
    rows: List[Tuple[str, str, str, int, int]] = []
    for d in daterange(datetime.fromisoformat(start), datetime.fromisoformat(end)):
        sb = fetch_nhl_scoreboard(d.date().isoformat())
        for ev in sb.get("games", []):
            scores = ev.get("scores") or {}
            home_score = scores.get("home")
            away_score = scores.get("away")
            try:
                hs = int(home_score)
                ascore = int(away_score)
            except Exception:
                continue
            home = team_mapper.normalize_team_code(ev["teams"]["home"]["alias"], "nhl")
            away = team_mapper.normalize_team_code(ev["teams"]["away"]["alias"], "nhl")
            if not home or not away:
                continue
            rows.append((d.date().isoformat(), home, away, hs, ascore))
    return rows


def expected_home_prob(r_home: float, r_away: float, home_bonus: float) -> float:
    r_home_adj = r_home + home_bonus
    expo = (r_away - r_home_adj) / 400.0
    return 1.0 / (1.0 + 10 ** expo)


def build_elo(
    rows: List[Tuple[str, str, str, int, int]],
    base: float,
    k_factor: float,
    home_bonus: float,
    decay: float,
) -> Dict[str, float]:
    ratings: Dict[str, float] = {}
    for _, home, away, hs, ascore in rows:
        r_home = ratings.get(home, base)
        r_away = ratings.get(away, base)
        # decay back toward base before applying this game's delta
        if decay < 1.0:
            r_home = base + (r_home - base) * decay
            r_away = base + (r_away - base) * decay
        e_home = expected_home_prob(r_home, r_away, home_bonus)
        result_home = 1.0 if hs > ascore else 0.0 if hs < ascore else 0.5
        delta = k_factor * (result_home - e_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta
    return ratings


def main() -> None:
    ap = argparse.ArgumentParser(description="Build NHL Elo ratings from ESPN scoreboard history")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--out", default=str(OUT_PATH), help="Output path for ratings JSON")
    ap.add_argument("--base", type=float, default=1500.0, help="Base rating")
    ap.add_argument("--k", type=float, default=20.0, help="Elo K factor")
    ap.add_argument("--home-bonus", type=float, default=60.0, help="Home bonus in Elo points")
    ap.add_argument("--decay", type=float, default=1.0, help="Recency decay per game (0<decay<=1)")
    args = ap.parse_args()

    rows = load_games(args.start, args.end)
    if not rows:
        print(f"[error] no games found in range {args.start}..{args.end}")
        return
    ratings = build_elo(rows, base=args.base, k_factor=args.k, home_bonus=args.home_bonus, decay=args.decay)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ratings, indent=2), encoding="utf-8")
    print(f"[info] wrote Elo ratings for {len(ratings)} NHL teams to {out_path}")


if __name__ == "__main__":
    main()
