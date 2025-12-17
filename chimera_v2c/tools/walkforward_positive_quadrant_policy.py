#!/usr/bin/env python
"""
Walk-forward (train->test) evaluation of a "positive quadrant buckets" policy.

This answers the leakage question for the in-sample "positive buckets" approach:
for each test date D, learn which quadrant buckets (A/B/C/D) are positive for each
model using only dates < D, then trade only those buckets on date D.

Policy
------
For a given model and edge threshold t:
  - For each game, compute the quadrant bucket (A/B/C/D) at threshold t.
  - Learn train bucket stats (bets, total_pnl, avg_pnl) vs kalshi_mid.
  - Mark buckets as allowed if bets>=min_bets and avg_pnl>=ev_threshold.
  - On test day, trade 1 unit whenever the model triggers an allowed bucket.

Multi-model stacking
--------------------
We also compute a "COMBINED" policy where each model contributes +1 unit if it
triggers an allowed bucket on that game (so multiple models can stack units).

Safety: read-only on daily ledgers.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.rulebook_quadrants import (
    bucket_letters,
    pnl_buy_away,
    pnl_buy_home,
    trade_side,
)


THESIS_DIR = Path("reports/thesis_summaries")
QUADRANT_BUCKETS = ["A", "B", "C", "D"]


@dataclass(frozen=True)
class Sample:
    date: str
    league: str
    matchup: str
    p_mid: float
    y: int
    probs: Dict[str, float]


@dataclass
class Totals:
    bets: int = 0
    total_pnl: float = 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.bets if self.bets else 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Walk-forward positive quadrant bucket policy (1 unit per model signal; read-only)."
    )
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive (test window).")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive (test window).")
    ap.add_argument("--league", default="all", help="nba|nhl|nfl|all (default: all).")
    ap.add_argument(
        "--pool-leagues",
        action="store_true",
        help="When --league=all, learn buckets pooled across leagues (default: learn per-league).",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["grok", "gemini", "gpt", "market_proxy"],
        help="Models to include (default: grok gemini gpt market_proxy).",
    )
    ap.add_argument("--train-days", type=int, default=0, help="Rolling train window in ledger days (0 = expanding).")
    ap.add_argument("--min-train-days", type=int, default=5, help="Minimum distinct train days required (default: 5).")
    ap.add_argument("--min-bets", type=int, default=10, help="Min bets per (model,bucket) gate (default: 10).")
    ap.add_argument("--ev-threshold", type=float, default=0.0, help="Min avg_pnl per bet for a bucket (default: 0.0).")
    ap.add_argument(
        "--edge-threshold",
        type=float,
        default=0.02,
        help="Fixed edge threshold t for quadrant buckets (default: 0.02).",
    )
    ap.add_argument(
        "--edge-thresholds",
        nargs="+",
        type=float,
        help="Optional threshold candidates; if provided, pick the best t per (model,league) on the train set.",
    )
    ap.add_argument(
        "--threshold-select-mode",
        choices=["max_total_pnl", "max_avg_pnl", "min_edge"],
        default="max_total_pnl",
        help="How to pick t among candidates using train data (default: max_total_pnl).",
    )
    ap.add_argument("--out", default="", help="Optional output CSV path. Default under reports/thesis_summaries/.")
    ap.add_argument("--no-write", action="store_true", help="Print summary only; do not write CSV.")
    return ap.parse_args()


def _normalize_league_arg(league: str) -> Optional[str]:
    v = (league or "").strip().lower()
    if v in {"", "all"}:
        return None
    if v in {"nba", "nhl", "nfl"}:
        return v
    raise SystemExit("[error] --league must be one of: nba, nhl, nfl, all")


def _pnl_for_bucket(*, bucket: str, p_mid: float, y: int) -> float:
    side = trade_side(bucket)
    if side == "home":
        return float(pnl_buy_home(p_mid=p_mid, home_win=float(y)))
    return float(pnl_buy_away(p_mid=p_mid, home_win=float(y)))


def _quadrant_bucket(*, p_mid: float, p_model: float, t: float) -> Optional[str]:
    letters = bucket_letters(p_mid=float(p_mid), p_model=float(p_model), edge_threshold=float(t))
    for b in letters:
        if b in QUADRANT_BUCKETS:
            return b
    return None


def _iter_samples(games: Iterable[GameRow], models: Sequence[str]) -> List[Sample]:
    out: List[Sample] = []
    for g in games:
        if g.home_win not in (0.0, 1.0):
            continue
        if g.kalshi_mid is None:
            continue
        probs: Dict[str, float] = {}
        for m in models:
            pm = g.probs.get(m)
            if pm is None:
                continue
            probs[str(m)] = float(pm)
        out.append(
            Sample(
                date=g.date.strftime("%Y-%m-%d"),
                league=g.league,
                matchup=g.matchup,
                p_mid=float(g.kalshi_mid),
                y=int(g.home_win),
                probs=probs,
            )
        )
    return out


def _train_window(samples_all: List[Sample], *, test_date: str, train_days: int) -> List[Sample]:
    before = [s for s in samples_all if s.date < test_date]
    if train_days <= 0:
        return before
    dates = sorted({s.date for s in before})
    keep = set(dates[-train_days:])
    return [s for s in before if s.date in keep]


def _bucket_stats(
    train: Sequence[Sample],
    *,
    model: str,
    t: float,
) -> Dict[str, Totals]:
    stats: Dict[str, Totals] = {b: Totals() for b in QUADRANT_BUCKETS}
    for s in train:
        p_model = s.probs.get(model)
        if p_model is None:
            continue
        bucket = _quadrant_bucket(p_mid=s.p_mid, p_model=float(p_model), t=t)
        if bucket is None:
            continue
        pnl = _pnl_for_bucket(bucket=bucket, p_mid=s.p_mid, y=s.y)
        tot = stats[bucket]
        tot.bets += 1
        tot.total_pnl += float(pnl)
    return stats


def _allowed_buckets(
    stats: Dict[str, Totals],
    *,
    min_bets: int,
    ev_threshold: float,
) -> Set[str]:
    out: Set[str] = set()
    for b, t in stats.items():
        if t.bets >= int(min_bets) and t.avg_pnl >= float(ev_threshold):
            out.add(b)
    return out


def _pick_threshold_for_model(
    train: Sequence[Sample],
    *,
    model: str,
    candidates: Sequence[float],
    min_bets: int,
    ev_threshold: float,
    mode: str,
) -> float:
    mode_norm = (mode or "").strip().lower()
    best_t = float(candidates[0])
    best_score = None
    for t in candidates:
        tt = float(t)
        stats = _bucket_stats(train, model=model, t=tt)
        allowed = _allowed_buckets(stats, min_bets=min_bets, ev_threshold=ev_threshold)
        bets = sum(stats[b].bets for b in allowed)
        pnl = sum(stats[b].total_pnl for b in allowed)
        avg = (pnl / bets) if bets else float("-inf")
        if mode_norm == "min_edge":
            score = -tt if bets > 0 else float("-inf")
        elif mode_norm == "max_avg_pnl":
            score = avg
        else:
            score = pnl
        if best_score is None or score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and tt < best_t):
            best_score = score
            best_t = tt
    return float(best_t)


def _walkforward_rows(
    *,
    samples_all: Sequence[Sample],
    models: Sequence[str],
    league_filter: Optional[str],
    pool_leagues: bool,
    train_days: int,
    min_train_days: int,
    min_bets: int,
    ev_threshold: float,
    candidates: Sequence[float],
    threshold_select_mode: str,
) -> Tuple[List[Dict[str, object]], Dict[str, Totals], Totals]:
    test_dates = sorted({s.date for s in samples_all})
    if league_filter:
        leagues = [league_filter]
    elif pool_leagues:
        leagues = ["overall"]
    else:
        leagues = sorted({s.league for s in samples_all})

    rows_out: List[Dict[str, object]] = []
    overall_by_model: Dict[str, Totals] = {m: Totals() for m in models}
    overall_combined = Totals()

    for d in test_dates:
        train_all = _train_window(list(samples_all), test_date=d, train_days=int(train_days))
        test_all = [s for s in samples_all if s.date == d]
        if not test_all:
            continue

        for league in leagues:
            if league_filter:
                train = [s for s in train_all if s.league == league_filter]
                test = [s for s in test_all if s.league == league_filter]
            elif pool_leagues:
                train = list(train_all)
                test = list(test_all)
            else:
                train = [s for s in train_all if s.league == league]
                test = [s for s in test_all if s.league == league]

            train_days_list = sorted({s.date for s in train})
            if len(train_days_list) < int(min_train_days):
                continue
            if not test:
                continue

            t_by_model: Dict[str, float] = {}
            allowed_by_model: Dict[str, Set[str]] = {}
            for m in models:
                if len(candidates) > 1:
                    t_m = _pick_threshold_for_model(
                        train,
                        model=m,
                        candidates=candidates,
                        min_bets=int(min_bets),
                        ev_threshold=float(ev_threshold),
                        mode=str(threshold_select_mode),
                    )
                else:
                    t_m = float(candidates[0])
                t_by_model[m] = t_m
                stats = _bucket_stats(train, model=m, t=t_m)
                allowed_by_model[m] = _allowed_buckets(stats, min_bets=int(min_bets), ev_threshold=float(ev_threshold))

            day_row: Dict[str, object] = {
                "test_date": d,
                "league": league if league_filter or pool_leagues else str(league),
                "train_start": train_days_list[0],
                "train_end": train_days_list[-1],
                "train_days": len(train_days_list),
                "train_samples": len(train),
            }

            day_combined = Totals()
            for m in models:
                t_m = t_by_model[m]
                allowed = allowed_by_model[m]
                day_tot = Totals()
                for s in test:
                    p_model = s.probs.get(m)
                    if p_model is None:
                        continue
                    bucket = _quadrant_bucket(p_mid=s.p_mid, p_model=float(p_model), t=t_m)
                    if bucket is None or bucket not in allowed:
                        continue
                    day_tot.bets += 1
                    pnl = _pnl_for_bucket(bucket=bucket, p_mid=s.p_mid, y=s.y)
                    day_tot.total_pnl += float(pnl)

                overall_by_model[m].bets += day_tot.bets
                overall_by_model[m].total_pnl += day_tot.total_pnl
                day_combined.bets += day_tot.bets
                day_combined.total_pnl += day_tot.total_pnl

                day_row[f"t_{m}"] = f"{t_m:.3f}"
                day_row[f"positive_{m}"] = " ".join(sorted(allowed)) if allowed else ""
                day_row[f"bets_{m}"] = day_tot.bets
                day_row[f"pnl_{m}"] = f"{day_tot.total_pnl:.6f}"
                day_row[f"avg_{m}"] = f"{day_tot.avg_pnl:.6f}" if day_tot.bets else ""

            overall_combined.bets += day_combined.bets
            overall_combined.total_pnl += day_combined.total_pnl
            day_row["bets_combined"] = day_combined.bets
            day_row["pnl_combined"] = f"{day_combined.total_pnl:.6f}"
            day_row["avg_combined"] = f"{day_combined.avg_pnl:.6f}" if day_combined.bets else ""
            rows_out.append(day_row)

    return rows_out, overall_by_model, overall_combined


def main() -> None:
    args = parse_args()
    if not LEDGER_DIR.exists():
        raise SystemExit(f"[error] daily ledger directory missing: {LEDGER_DIR}")

    league_filter = _normalize_league_arg(args.league)
    models = [m.strip().lower() for m in (args.models or []) if m.strip()]
    if not models:
        raise SystemExit("[error] --models must be non-empty")

    candidates = list(args.edge_thresholds) if args.edge_thresholds else [float(args.edge_threshold)]
    if any(float(t) <= 0 for t in candidates):
        raise SystemExit("[error] edge thresholds must be > 0")

    games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=league_filter,
        models=models,
    )
    samples_all = _iter_samples(games, models=models)
    if not samples_all:
        raise SystemExit("[error] no graded samples found in selected window")

    rows_out, overall_by_model, overall_combined = _walkforward_rows(
        samples_all=samples_all,
        models=models,
        league_filter=league_filter,
        pool_leagues=bool(args.pool_leagues),
        train_days=int(args.train_days),
        min_train_days=int(args.min_train_days),
        min_bets=int(args.min_bets),
        ev_threshold=float(args.ev_threshold),
        candidates=candidates,
        threshold_select_mode=str(args.threshold_select_mode),
    )

    if not rows_out:
        raise SystemExit("[error] no walk-forward rows produced (check min_train_days/window).")

    print("\n=== Walk-forward positive quadrant policy summary ===")
    print(f"league={args.league} window={args.start_date}..{args.end_date} rows={len(rows_out)}")
    for m in models:
        tot = overall_by_model[m]
        print(f"{m:12s} bets={tot.bets:4d} total_pnl={tot.total_pnl:7.3f} avg_pnl_per_bet={tot.avg_pnl:6.3f}")
    print(
        f"{'COMBINED':12s} bets={overall_combined.bets:4d} total_pnl={overall_combined.total_pnl:7.3f} "
        f"avg_pnl_per_bet={overall_combined.avg_pnl:6.3f}"
    )

    if args.no_write:
        return

    candidates_tag = (
        f"t{float(candidates[0]):.3f}"
        if len(candidates) == 1
        else "t" + "-".join(f"{float(t):.3f}" for t in candidates) + f"_{args.threshold_select_mode}"
    )
    scope_tag = "pooled" if (args.league or "").lower() == "all" and bool(args.pool_leagues) else "per_league"
    models_tag = "-".join([m.strip().lower() for m in (args.models or []) if m.strip()]) or "models"
    out_path = (
        Path(args.out)
        if args.out
        else THESIS_DIR
        / (
            "walkforward_positive_quadrant_policy_"
            f"{(args.league or 'all').lower()}_{scope_tag}_{models_tag}_{candidates_tag}_"
            f"{args.start_date.replace('-','')}_{args.end_date.replace('-','')}.csv"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"[ok] wrote -> {out_path}")


if __name__ == "__main__":
    main()
