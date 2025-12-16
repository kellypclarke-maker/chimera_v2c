from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from chimera_v2c.src.ledger.outcomes import parse_home_win as _parse_home_win

MODELS = ["v2c", "gemini", "grok", "gpt", "kalshi_mid", "market_proxy", "moneypuck"]


@dataclass
class ModelStats:
    games: int = 0
    graded: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    brier_sum: float = 0.0
    brier_n: int = 0


def parse_home_win(actual_outcome: str) -> Optional[float]:
    return _parse_home_win(actual_outcome)


def build_wr_summary(
    daily_dir: Path,
    out_path: Path,
) -> None:
    """
    Scan all daily ledgers and compute per-league and overall accuracy/Brier
    for each model/market, ignoring games without a finalized outcome.

    Output schema:
      league,model,games,graded,wins,losses,pushes,acc,brier
    where:
      - league: nba / nhl / nfl / overall
      - acc: accuracy (win rate) over non-push games, rounded to 3 decimals
      - brier: mean squared error on 0/1 outcomes, rounded to 3 decimals
    """
    if not daily_dir.exists():
        raise SystemExit(f"Daily ledgers directory not found: {daily_dir}")

    # league -> model -> stats
    stats: Dict[str, Dict[str, ModelStats]] = defaultdict(
        lambda: {m: ModelStats() for m in MODELS}
    )

    ledger_paths = sorted(daily_dir.glob("*_daily_game_ledger.csv"))

    for path in ledger_paths:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                outcome = row.get("actual_outcome") or ""
                home_win = parse_home_win(outcome)
                if home_win is None:
                    # Skip games without a finalized result.
                    continue

                league_raw = (row.get("league") or "").strip()
                if not league_raw:
                    continue
                league = league_raw.lower()

                for model in MODELS:
                    val = row.get(model)
                    if val is None or str(val).strip() == "":
                        continue
                    try:
                        p_home = float(val)
                    except ValueError:
                        continue

                    # Per-league stats
                    league_stats = stats[league][model]
                    # Overall stats
                    overall_stats = stats["overall"][model]

                    for target in (league_stats, overall_stats):
                        target.games += 1
                        if home_win == 0.5:
                            target.pushes += 1
                            continue

                        target.graded += 1
                        pick_home = p_home >= 0.5
                        correct = (home_win == 1.0 and pick_home) or (
                            home_win == 0.0 and not pick_home
                        )
                        if correct:
                            target.wins += 1
                        else:
                            target.losses += 1

                        # Brier only on 0/1 outcomes
                        target.brier_sum += (p_home - home_win) ** 2
                        target.brier_n += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "league",
            "model",
            "games",
            "graded",
            "wins",
            "losses",
            "pushes",
            "acc",
            "brier",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Write league rows (sorted), then overall at the end.
        leagues = sorted(l for l in stats.keys() if l != "overall")
        if "overall" in stats:
            leagues.append("overall")

        for league in leagues:
            for model in MODELS:
                s = stats[league][model]
                if s.games == 0:
                    continue
                acc = s.wins / s.graded if s.graded > 0 else None
                brier = s.brier_sum / s.brier_n if s.brier_n > 0 else None
                writer.writerow(
                    {
                        "league": league,
                        "model": model,
                        "games": s.games,
                        "graded": s.graded,
                        "wins": s.wins,
                        "losses": s.losses,
                        "pushes": s.pushes,
                        "acc": f"{acc:.3f}" if acc is not None else "",
                        "brier": f"{brier:.3f}" if brier is not None else "",
                    }
                )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build per-league win-rate summary from daily ledgers."
    )
    ap.add_argument(
        "--daily-dir",
        default="reports/daily_ledgers",
        help="Directory containing *_daily_game_ledger.csv files "
        "(default: reports/daily_ledgers)",
    )
    ap.add_argument(
        "--out",
        default="reports/daily_ledgers/model_wr_by_league.csv",
        help="Output CSV path (default: reports/daily_ledgers/model_wr_by_league.csv)",
    )
    args = ap.parse_args()

    daily_dir = Path(args.daily_dir)
    out_path = Path(args.out)
    build_wr_summary(daily_dir, out_path)


if __name__ == "__main__":
    main()
