#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from chimera_v2c.src.ledger.outcomes import parse_home_win

MASTER_PATH = Path("reports/master_ledger/master_game_ledger.csv")
MODEL_COLS = ["v2c", "grok", "gemini", "gpt", "kalshi_mid", "market_proxy", "moneypuck"]


def parse_outcome(outcome: object) -> Optional[float]:
    return parse_home_win(outcome)


def analyze() -> None:
    if not MASTER_PATH.exists():
        print(f"No master ledger found at {MASTER_PATH}")
        return

    df = pd.read_csv(MASTER_PATH)
    df["y"] = df["actual_outcome"].apply(parse_outcome)
    finished = df[df["y"].notna()].copy()

    print(f"Total Games: {len(df)}")
    print(f"Finished Games: {len(finished)}")

    print("\nModel Performance (Accuracy & Brier Score):")
    print(f"{'Model':<15} {'Games':<6} {'Acc':<8} {'Brier':<8}")

    for model in MODEL_COLS:
        if model not in finished.columns:
            continue
        p = pd.to_numeric(finished[model], errors="coerce")
        subset = finished[p.notna()].copy()
        if subset.empty:
            continue
        subset["p"] = pd.to_numeric(subset[model], errors="coerce")

        # Accuracy (exclude pushes)
        no_push = subset[subset["y"].isin([0.0, 1.0])]
        if not no_push.empty:
            correct = ((no_push["p"] > 0.5) == (no_push["y"] == 1.0)).sum()
            acc = float(correct) / float(len(no_push))
        else:
            acc = float("nan")

        brier = float(((subset["p"] - subset["y"]) ** 2).mean())
        print(f"{model:<15} {len(subset):<6} {acc:>7.1%} {brier:>8.4f}")


if __name__ == "__main__":
    analyze()
