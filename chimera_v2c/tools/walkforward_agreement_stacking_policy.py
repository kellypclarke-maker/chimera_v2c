#!/usr/bin/env python
"""
Walk-forward (train->test) "agreement-only stacking" policy on quadrant buckets.

Goal
----
Turn the "positive quadrant buckets" idea into a stricter portfolio rule:
trade only when multiple models agree on the same side (home or away) and no
model signals the opposite side for that game.

For each test date D:
  1) Using only dates < D (optionally rolling), learn each model's allowed
     quadrant buckets (A/B/C/D) given:
       - edge threshold t (fixed, or selected from candidates on train-only)
       - gates: min_bets and ev_threshold on train bucket stats
  2) On day D, each model emits at most one signal (home/away) if it triggers an
     allowed bucket.
  3) If both sides have at least one signal, treat as "disagreement" and skip.
  4) If exactly one side has >= min_agree signals, place a trade on that side
     with units = number of agreeing models (optionally capped).

We also report a baseline "STACK_ALL" policy that simply sums +1 unit per model
signal (i.e., the same multi-model stacking used by walkforward_positive_quadrant_policy.py),
so you can see how much disagreement filtering helps.

Safety: read-only on daily ledgers.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
    units: int = 0
    total_pnl: float = 0.0

    @property
    def avg_pnl_per_bet(self) -> float:
        return self.total_pnl / self.bets if self.bets else 0.0

    @property
    def avg_pnl_per_unit(self) -> float:
        return self.total_pnl / self.units if self.units else 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Walk-forward agreement-only stacking policy on positive quadrant buckets (read-only)."
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
        default=["grok", "gemini", "market_proxy"],
        help="Models to include (default: grok gemini market_proxy).",
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
    ap.add_argument("--min-agree", type=int, default=2, help="Minimum agreeing models (default: 2).")
    ap.add_argument("--max-units", type=int, default=0, help="Optional cap on units per bet (0 = no cap).")
    ap.add_argument(
        "--quadrant-buckets",
        nargs="+",
        default=["A", "B", "C", "D"],
        choices=QUADRANT_BUCKETS,
        help="Which quadrant buckets are allowed at all (default: A B C D).",
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


def _pnl_for_side(*, side: str, p_mid: float, y: int) -> float:
    if side == "home":
        return float(pnl_buy_home(p_mid=float(p_mid), home_win=float(y)))
    return float(pnl_buy_away(p_mid=float(p_mid), home_win=float(y)))


def _bucket_for_side(*, p_mid: float, side: str) -> str:
    pm = float(p_mid)
    if side not in {"home", "away"}:
        raise ValueError("side must be 'home' or 'away'")
    if pm >= 0.5:
        return "B" if side == "home" else "A"
    return "D" if side == "home" else "C"


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


def _bucket_stats(train: Sequence[Sample], *, model: str, t: float) -> Dict[str, Totals]:
    stats: Dict[str, Totals] = {b: Totals() for b in QUADRANT_BUCKETS}
    for s in train:
        p_model = s.probs.get(model)
        if p_model is None:
            continue
        bucket = _quadrant_bucket(p_mid=s.p_mid, p_model=float(p_model), t=t)
        if bucket is None:
            continue
        side = trade_side(bucket)
        pnl = _pnl_for_side(side=side, p_mid=s.p_mid, y=s.y)
        tot = stats[bucket]
        tot.bets += 1
        tot.units += 1
        tot.total_pnl += float(pnl)
    return stats


def _allowed_buckets(
    stats: Dict[str, Totals],
    *,
    active_buckets: Set[str],
    min_bets: int,
    ev_threshold: float,
) -> Set[str]:
    out: Set[str] = set()
    for b, t in stats.items():
        if b not in active_buckets:
            continue
        if t.bets >= int(min_bets) and t.avg_pnl_per_bet >= float(ev_threshold):
            out.add(b)
    return out


def _pick_threshold_for_model(
    train: Sequence[Sample],
    *,
    model: str,
    candidates: Sequence[float],
    active_buckets: Set[str],
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
        allowed = _allowed_buckets(stats, active_buckets=active_buckets, min_bets=min_bets, ev_threshold=ev_threshold)
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


def _signals_for_game(
    *,
    s: Sample,
    models: Sequence[str],
    t_by_model: Dict[str, float],
    allowed_by_model: Dict[str, Set[str]],
) -> Tuple[int, int]:
    home = 0
    away = 0
    for m in models:
        p_model = s.probs.get(m)
        if p_model is None:
            continue
        bucket = _quadrant_bucket(p_mid=s.p_mid, p_model=float(p_model), t=float(t_by_model[m]))
        if bucket is None or bucket not in allowed_by_model[m]:
            continue
        side = trade_side(bucket)
        if side == "home":
            home += 1
        else:
            away += 1
    return home, away


def _units_capped(units: int, *, cap: int) -> int:
    if cap and cap > 0:
        return min(int(units), int(cap))
    return int(units)


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
    min_agree: int,
    max_units: int,
    active_buckets: Set[str],
) -> Tuple[List[Dict[str, object]], Totals, Totals]:
    test_dates = sorted({s.date for s in samples_all})
    if league_filter:
        leagues = [league_filter]
    elif pool_leagues:
        leagues = ["overall"]
    else:
        leagues = sorted({s.league for s in samples_all})

    rows_out: List[Dict[str, object]] = []
    totals_agree = Totals()
    totals_stack = Totals()
    totals_agree_by_bucket: Dict[str, Totals] = {b: Totals() for b in QUADRANT_BUCKETS}
    totals_stack_by_bucket: Dict[str, Totals] = {b: Totals() for b in QUADRANT_BUCKETS}

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
                        active_buckets=active_buckets,
                        min_bets=int(min_bets),
                        ev_threshold=float(ev_threshold),
                        mode=str(threshold_select_mode),
                    )
                else:
                    t_m = float(candidates[0])
                t_by_model[m] = float(t_m)
                stats = _bucket_stats(train, model=m, t=float(t_m))
                allowed_by_model[m] = _allowed_buckets(
                    stats,
                    active_buckets=active_buckets,
                    min_bets=int(min_bets),
                    ev_threshold=float(ev_threshold),
                )

            day_agree = Totals()
            day_stack = Totals()
            disagree_games = 0
            day_agree_by_bucket: Dict[str, Totals] = {b: Totals() for b in QUADRANT_BUCKETS}
            day_stack_by_bucket: Dict[str, Totals] = {b: Totals() for b in QUADRANT_BUCKETS}

            for s in test:
                home_sigs, away_sigs = _signals_for_game(
                    s=s, models=models, t_by_model=t_by_model, allowed_by_model=allowed_by_model
                )

                # Baseline stacking: sum all model units regardless of disagreement.
                if home_sigs + away_sigs > 0:
                    day_stack.bets += 1
                    day_stack.units += home_sigs + away_sigs
                    pnl_home = _pnl_for_side(side="home", p_mid=s.p_mid, y=s.y) if home_sigs else 0.0
                    pnl_away = _pnl_for_side(side="away", p_mid=s.p_mid, y=s.y) if away_sigs else 0.0
                    day_stack.total_pnl += float(home_sigs) * pnl_home + float(away_sigs) * pnl_away

                    if home_sigs:
                        b = _bucket_for_side(p_mid=s.p_mid, side="home")
                        t = day_stack_by_bucket[b]
                        t.units += int(home_sigs)
                        t.total_pnl += float(home_sigs) * float(pnl_home)
                    if away_sigs:
                        b = _bucket_for_side(p_mid=s.p_mid, side="away")
                        t = day_stack_by_bucket[b]
                        t.units += int(away_sigs)
                        t.total_pnl += float(away_sigs) * float(pnl_away)

                # Agreement-only: skip if both sides signal.
                if home_sigs > 0 and away_sigs > 0:
                    disagree_games += 1
                    continue

                if home_sigs >= int(min_agree):
                    units = _units_capped(home_sigs, cap=int(max_units))
                    day_agree.bets += 1
                    day_agree.units += units
                    pnl = float(_pnl_for_side(side="home", p_mid=s.p_mid, y=s.y))
                    day_agree.total_pnl += float(units) * pnl
                    b = _bucket_for_side(p_mid=s.p_mid, side="home")
                    t = day_agree_by_bucket[b]
                    t.units += int(units)
                    t.total_pnl += float(units) * pnl
                elif away_sigs >= int(min_agree):
                    units = _units_capped(away_sigs, cap=int(max_units))
                    day_agree.bets += 1
                    day_agree.units += units
                    pnl = float(_pnl_for_side(side="away", p_mid=s.p_mid, y=s.y))
                    day_agree.total_pnl += float(units) * pnl
                    b = _bucket_for_side(p_mid=s.p_mid, side="away")
                    t = day_agree_by_bucket[b]
                    t.units += int(units)
                    t.total_pnl += float(units) * pnl

            totals_agree.bets += day_agree.bets
            totals_agree.units += day_agree.units
            totals_agree.total_pnl += day_agree.total_pnl

            totals_stack.bets += day_stack.bets
            totals_stack.units += day_stack.units
            totals_stack.total_pnl += day_stack.total_pnl
            for b in QUADRANT_BUCKETS:
                totals_agree_by_bucket[b].units += day_agree_by_bucket[b].units
                totals_agree_by_bucket[b].total_pnl += day_agree_by_bucket[b].total_pnl
                totals_stack_by_bucket[b].units += day_stack_by_bucket[b].units
                totals_stack_by_bucket[b].total_pnl += day_stack_by_bucket[b].total_pnl

            row: Dict[str, object] = {
                "test_date": d,
                "league": league if league_filter or pool_leagues else str(league),
                "train_start": train_days_list[0],
                "train_end": train_days_list[-1],
                "train_days": len(train_days_list),
                "train_samples": len(train),
                "min_agree": int(min_agree),
                "max_units": int(max_units),
                "quadrant_buckets": " ".join(sorted(active_buckets)),
                "disagree_games": disagree_games,
                "STACK_ALL_bets": day_stack.bets,
                "STACK_ALL_units": day_stack.units,
                "STACK_ALL_pnl": f"{day_stack.total_pnl:.6f}",
                "STACK_ALL_avg_pnl_per_unit": f"{day_stack.avg_pnl_per_unit:.6f}" if day_stack.units else "",
                "AGREE_ONLY_bets": day_agree.bets,
                "AGREE_ONLY_units": day_agree.units,
                "AGREE_ONLY_pnl": f"{day_agree.total_pnl:.6f}",
                "AGREE_ONLY_avg_pnl_per_unit": f"{day_agree.avg_pnl_per_unit:.6f}" if day_agree.units else "",
            }
            for b in QUADRANT_BUCKETS:
                row[f"STACK_ALL_units_{b}"] = day_stack_by_bucket[b].units
                row[f"STACK_ALL_pnl_{b}"] = f"{day_stack_by_bucket[b].total_pnl:.6f}"
                row[f"AGREE_ONLY_units_{b}"] = day_agree_by_bucket[b].units
                row[f"AGREE_ONLY_pnl_{b}"] = f"{day_agree_by_bucket[b].total_pnl:.6f}"
            for m in models:
                row[f"t_{m}"] = f"{float(t_by_model[m]):.3f}"
                allowed = allowed_by_model[m]
                row[f"positive_{m}"] = " ".join(sorted(allowed)) if allowed else ""
            rows_out.append(row)

    # Attach overall totals as a synthetic final row (useful for quick bucket slicing in CSV tools).
    overall_row: Dict[str, object] = {
        "test_date": "OVERALL",
        "league": league_filter or ("overall" if pool_leagues else "per_league"),
        "train_start": "",
        "train_end": "",
        "train_days": "",
        "train_samples": "",
        "min_agree": int(min_agree),
        "max_units": int(max_units),
        "quadrant_buckets": " ".join(sorted(active_buckets)),
        "disagree_games": "",
        "STACK_ALL_bets": totals_stack.bets,
        "STACK_ALL_units": totals_stack.units,
        "STACK_ALL_pnl": f"{totals_stack.total_pnl:.6f}",
        "STACK_ALL_avg_pnl_per_unit": f"{totals_stack.avg_pnl_per_unit:.6f}" if totals_stack.units else "",
        "AGREE_ONLY_bets": totals_agree.bets,
        "AGREE_ONLY_units": totals_agree.units,
        "AGREE_ONLY_pnl": f"{totals_agree.total_pnl:.6f}",
        "AGREE_ONLY_avg_pnl_per_unit": f"{totals_agree.avg_pnl_per_unit:.6f}" if totals_agree.units else "",
    }
    for b in QUADRANT_BUCKETS:
        overall_row[f"STACK_ALL_units_{b}"] = totals_stack_by_bucket[b].units
        overall_row[f"STACK_ALL_pnl_{b}"] = f"{totals_stack_by_bucket[b].total_pnl:.6f}"
        overall_row[f"AGREE_ONLY_units_{b}"] = totals_agree_by_bucket[b].units
        overall_row[f"AGREE_ONLY_pnl_{b}"] = f"{totals_agree_by_bucket[b].total_pnl:.6f}"
    for m in models:
        overall_row[f"t_{m}"] = ""
        overall_row[f"positive_{m}"] = ""
    rows_out.append(overall_row)

    return rows_out, totals_agree, totals_stack


def main() -> None:
    args = parse_args()
    if not LEDGER_DIR.exists():
        raise SystemExit(f"[error] daily ledger directory missing: {LEDGER_DIR}")

    league_filter = _normalize_league_arg(args.league)
    models = [m.strip().lower() for m in (args.models or []) if m.strip()]
    if not models:
        raise SystemExit("[error] --models must be non-empty")
    if int(args.min_agree) < 1:
        raise SystemExit("[error] --min-agree must be >= 1")

    active_buckets = {b.strip().upper() for b in (args.quadrant_buckets or []) if b.strip()}
    if not active_buckets:
        raise SystemExit("[error] --quadrant-buckets must be non-empty")
    unknown = active_buckets.difference(set(QUADRANT_BUCKETS))
    if unknown:
        raise SystemExit(f"[error] unknown bucket(s) in --quadrant-buckets: {sorted(unknown)}")

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

    rows_out, totals_agree, totals_stack = _walkforward_rows(
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
        min_agree=int(args.min_agree),
        max_units=int(args.max_units),
        active_buckets=active_buckets,
    )
    if not rows_out:
        raise SystemExit("[error] no walk-forward rows produced (check min_train_days/window).")

    print("\n=== Walk-forward agreement stacking summary ===")
    print(
        f"league={args.league} window={args.start_date}..{args.end_date} rows={len(rows_out)} "
        f"models={','.join(models)} min_agree={args.min_agree} max_units={args.max_units or 0} "
        f"quadrant_buckets={' '.join(sorted(active_buckets))}"
    )
    print(
        f"{'AGREE_ONLY':10s} bets={totals_agree.bets:4d} units={totals_agree.units:4d} total_pnl={totals_agree.total_pnl:7.3f} "
        f"avg_pnl_per_unit={totals_agree.avg_pnl_per_unit:6.3f}"
    )
    print(
        f"{'STACK_ALL':10s} bets={totals_stack.bets:4d} units={totals_stack.units:4d} total_pnl={totals_stack.total_pnl:7.3f} "
        f"avg_pnl_per_unit={totals_stack.avg_pnl_per_unit:6.3f}"
    )
    # Bucket slicing is easiest from the output CSV (includes an OVERALL row with A/B/C/D totals).

    if args.no_write:
        return

    candidates_tag = (
        f"t{float(candidates[0]):.3f}"
        if len(candidates) == 1
        else "t" + "-".join(f"{float(t):.3f}" for t in candidates) + f"_{args.threshold_select_mode}"
    )
    scope_tag = "pooled" if (args.league or "").lower() == "all" and bool(args.pool_leagues) else "per_league"
    models_tag = "-".join(models) or "models"
    buckets_tag = "".join(sorted(active_buckets))
    out_path = (
        Path(args.out)
        if args.out
        else THESIS_DIR
        / (
            "walkforward_agreement_stacking_policy_"
            f"{(args.league or 'all').lower()}_{scope_tag}_{models_tag}_"
            f"agree{int(args.min_agree)}_cap{int(args.max_units)}_b{buckets_tag}_{candidates_tag}_"
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
