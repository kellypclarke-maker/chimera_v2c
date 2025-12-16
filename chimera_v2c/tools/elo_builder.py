"""
Build NBA Elo ratings from the local chimera_v2c/data/chimera.db games table.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/elo_builder.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

DB_PATH = Path("chimera_v2c/data/chimera.db")
OUT_PATH = Path("chimera_v2c/data/team_ratings.json")


@dataclass
class EloConfig:
    base: float = 1500.0
    k_factor: float = 20.0
    home_bonus: float = 60.0


def load_games(db_path: Path) -> list[Tuple[str, str, str, int, int]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT game_date, home_team, away_team, home_score, away_score
        FROM games
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY game_date ASC, game_id ASC
        """
    ).fetchall()
    conn.close()
    return rows


def expected_home_prob(r_home: float, r_away: float, home_bonus: float) -> float:
    r_home_adj = r_home + home_bonus
    expo = (r_away - r_home_adj) / 400.0
    return 1.0 / (1.0 + 10 ** expo)


def build_elo(
    rows: list[Tuple[str, str, str, int, int]], cfg: EloConfig, decay: float = 1.0
) -> Dict[str, float]:
    ratings: Dict[str, float] = {}
    for _, home, away, hs, ascore in rows:
        r_home = ratings.get(home, cfg.base)
        r_away = ratings.get(away, cfg.base)

        # Apply recency decay by nudging each rating back toward base before this game,
        # then apply the fresh game delta unscaled.
        if decay < 1.0:
            r_home = cfg.base + (r_home - cfg.base) * decay
            r_away = cfg.base + (r_away - cfg.base) * decay

        e_home = expected_home_prob(r_home, r_away, cfg.home_bonus)
        result_home = 1.0 if hs > ascore else 0.0 if hs < ascore else 0.5
        delta = cfg.k_factor * (result_home - e_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta
    return ratings


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Elo ratings from chimera_v2c/data/chimera.db")
    ap.add_argument("--db", default=DB_PATH, help="Path to chimera_v2c/data/chimera.db")
    ap.add_argument("--out", default=OUT_PATH, help="Path to write team_ratings.json")
    ap.add_argument("--k", type=float, default=20.0, help="Elo K factor")
    ap.add_argument("--home-bonus", type=float, default=60.0, help="Home bonus in Elo points")
    ap.add_argument("--decay", type=float, default=1.0, help="Recency decay factor per game (0<decay<=1, e.g., 0.99).")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[error] DB not found at {db_path}")
        return

    rows = load_games(db_path)
    if not rows:
        print(f"[error] no games found in {db_path}")
        return

    cfg = EloConfig(base=1500.0, k_factor=args.k, home_bonus=args.home_bonus)
    ratings = build_elo(rows, cfg, decay=args.decay)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ratings, indent=2), encoding="utf-8")
    print(f"[info] wrote Elo ratings for {len(ratings)} teams to {out_path}")


if __name__ == "__main__":
    main()
