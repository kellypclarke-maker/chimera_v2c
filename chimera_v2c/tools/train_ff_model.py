"""
Train a Four Factors logistic regression from chimera_v2c/data/chimera.db and save coefficients.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/train_ff_model.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

DB_PATH = Path("chimera_v2c/data/chimera.db")
OUT_PATH = Path("chimera_v2c/data/ff_model.json")

FEATURES = ["efg_diff", "tov_diff", "orb_diff", "ftr_diff"]


def load_game_rows(db_path: Path) -> List[Tuple[float, float, float, float, int]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT
            g.game_date,
            h.efg_pct - a.efg_pct as efg_diff,
            h.tov_pct - a.tov_pct as tov_diff,
            h.orb_pct - a.orb_pct as orb_diff,
            h.ft_rate - a.ft_rate as ftr_diff,
            CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_win
        FROM games g
        JOIN team_stats h ON g.game_id = h.game_id AND g.home_team_id = h.team_id
        JOIN team_stats a ON g.game_id = a.game_id AND g.away_team_id = a.team_id
        WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        ORDER BY g.game_date ASC, g.game_id ASC
        """
    ).fetchall()
    conn.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Four Factors logistic model.")
    ap.add_argument("--db", default=DB_PATH, help="Path to chimera_v2c/data/chimera.db")
    ap.add_argument("--out", default=OUT_PATH, help="Path to write ff_model.json")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[error] DB not found at {db_path}")
        return

    rows = load_game_rows(db_path)
    if not rows:
        print("[error] no game rows found to train on")
        return

    cleaned = []
    targets = []
    for r in rows:
        feats = [r[1], r[2], r[3], r[4]]
        if any(v is None for v in feats):
            continue
        if any(np.isnan(v) for v in feats):
            continue
        cleaned.append(feats)
        targets.append(r[5])
    X = np.array(cleaned, dtype=float)
    y = np.array(targets, dtype=int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    coefs = dict(zip(FEATURES, model.coef_[0].tolist()))
    payload = {
        "features": FEATURES,
        "coef": coefs,
        "intercept": float(model.intercept_[0]),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[info] trained FF logistic; wrote to {out_path}")
    print(f"[info] intercept={payload['intercept']:.4f}, coefs={payload['coef']}")


if __name__ == "__main__":
    main()
