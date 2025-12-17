#!/usr/bin/env python
"""
Analyze "positive quadrant buckets" per model vs Kalshi mid, and simulate a simple policy.

Goal (operator-friendly)
------------------------
For each model (e.g., grok/gemini/gpt/market_proxy) and each league:
  1) Compute realized mid-trade PnL by quadrant bucket (A/B/C/D) at a fixed edge threshold t.
  2) Mark buckets as "positive" if they pass allow-gates:
       - bets >= --min-bets
       - avg_pnl_per_bet >= --ev-threshold
  3) Simulate the policy: trade 1 unit whenever the model's bucket is in its positive set.

Multi-model stacking
--------------------
This naturally supports "1 unit per model":
  - If multiple models are positive in the bucket they trigger for a game, each gets 1 unit.
  - Total strategy PnL is the sum of per-model PnL.

Safety: read-only on daily ledgers.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.rulebook_quadrants import DEFAULT_BUCKETS, BucketStats, compute_bucket_stats


OUT_DIR = Path("reports/thesis_summaries")
QUADRANT_BUCKETS = ["A", "B", "C", "D"]


@dataclass(frozen=True)
class PolicySummary:
    league: str
    model: str
    edge_threshold: float
    positive_buckets: Tuple[str, ...]
    bets: int
    total_pnl: float

    @property
    def avg_pnl_per_bet(self) -> float:
        return self.total_pnl / self.bets if self.bets else 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Analyze positive quadrant buckets per model and simulate 1-unit-per-model stacking (read-only)."
    )
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive (ledger filename dates).")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive (ledger filename dates).")
    ap.add_argument("--league", default="all", help="nba|nhl|nfl|all (default: all).")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["grok", "gemini", "gpt", "market_proxy"],
        help="Models to evaluate (default: grok gemini gpt market_proxy).",
    )
    ap.add_argument("--edge-threshold", type=float, default=0.05, help="Quadrant edge threshold t (default: 0.05).")
    ap.add_argument("--min-bets", type=int, default=10, help="Min bets gate for a bucket to count (default: 10).")
    ap.add_argument("--ev-threshold", type=float, default=0.0, help="Min avg_pnl_per_bet for a bucket (default: 0.0).")
    ap.add_argument("--out-prefix", default="", help="Optional output filename prefix under reports/thesis_summaries/.")
    ap.add_argument("--no-write", action="store_true", help="Print summary only; do not write CSVs.")
    return ap.parse_args()


def _normalize_league_arg(league: str) -> Optional[str]:
    v = (league or "").strip().lower()
    if v in {"", "all"}:
        return None
    if v in {"nba", "nhl", "nfl"}:
        return v
    raise SystemExit("[error] --league must be one of: nba, nhl, nfl, all")


def is_positive_bucket(stats: BucketStats, *, min_bets: int, ev_threshold: float) -> bool:
    return int(stats.bets) >= int(min_bets) and float(stats.avg_pnl) >= float(ev_threshold)


def _as_overall(games: Iterable[GameRow]) -> List[GameRow]:
    out: List[GameRow] = []
    for g in games:
        out.append(
            GameRow(
                date=g.date,
                league="overall",
                matchup=g.matchup,
                kalshi_mid=g.kalshi_mid,
                probs=g.probs,
                home_win=g.home_win,
            )
        )
    return out


def _policy_summaries(
    *,
    stats_by_key: Dict[Tuple[str, str, str], BucketStats],
    leagues: Sequence[str],
    models: Sequence[str],
    edge_threshold: float,
    min_bets: int,
    ev_threshold: float,
) -> List[PolicySummary]:
    positive: Dict[Tuple[str, str], Set[str]] = {}
    for league in leagues:
        for model in models:
            bs: Set[str] = set()
            for bucket in QUADRANT_BUCKETS:
                s = stats_by_key.get((league, model, bucket))
                if s is None:
                    continue
                if is_positive_bucket(s, min_bets=min_bets, ev_threshold=ev_threshold):
                    bs.add(bucket)
            positive[(league, model)] = bs

    out: List[PolicySummary] = []
    for league in leagues:
        for model in models:
            pos = tuple(sorted(positive.get((league, model), set())))
            bets = 0
            pnl = 0.0
            for bucket in pos:
                s = stats_by_key.get((league, model, bucket))
                if s is None:
                    continue
                bets += int(s.bets)
                pnl += float(s.total_pnl)
            out.append(
                PolicySummary(
                    league=league,
                    model=model,
                    edge_threshold=float(edge_threshold),
                    positive_buckets=pos,
                    bets=bets,
                    total_pnl=float(pnl),
                )
            )
    return out


def _combined_summary(
    summaries: Sequence[PolicySummary],
    *,
    league: str,
    models: Sequence[str],
) -> Tuple[int, float]:
    by_model = {(s.league, s.model): s for s in summaries}
    bets = 0
    pnl = 0.0
    for m in models:
        s = by_model.get((league, m))
        if s is None:
            continue
        bets += int(s.bets)
        pnl += float(s.total_pnl)
    return bets, float(pnl)


def _write_bucket_stats_csv(
    out_path: Path,
    *,
    stats_by_key: Dict[Tuple[str, str, str], BucketStats],
    leagues: Sequence[str],
    models: Sequence[str],
    min_bets: int,
    ev_threshold: float,
    edge_threshold: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "league",
        "model",
        "bucket",
        "edge_threshold",
        "bets",
        "wins",
        "win_rate",
        "total_pnl",
        "avg_pnl_per_bet",
        "positive",
        "min_bets",
        "ev_threshold",
    ]
    rows: List[Dict[str, object]] = []
    for league in leagues:
        for model in models:
            for bucket in QUADRANT_BUCKETS:
                s = stats_by_key.get((league, model, bucket))
                if s is None:
                    continue
                pos = is_positive_bucket(s, min_bets=min_bets, ev_threshold=ev_threshold)
                rows.append(
                    {
                        "league": league,
                        "model": model,
                        "bucket": bucket,
                        "edge_threshold": f"{float(edge_threshold):.3f}",
                        "bets": int(s.bets),
                        "wins": int(s.wins),
                        "win_rate": f"{float(s.win_rate):.6f}",
                        "total_pnl": f"{float(s.total_pnl):.6f}",
                        "avg_pnl_per_bet": f"{float(s.avg_pnl):.6f}",
                        "positive": str(bool(pos)),
                        "min_bets": int(min_bets),
                        "ev_threshold": f"{float(ev_threshold):.6f}",
                    }
                )
    rows.sort(key=lambda r: (r["league"], r["model"], r["bucket"]))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_policy_summary_csv(out_path: Path, summaries: Sequence[PolicySummary], *, models: Sequence[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "league",
        "model",
        "edge_threshold",
        "positive_buckets",
        "bets",
        "total_pnl",
        "avg_pnl_per_bet",
    ]
    rows: List[Dict[str, object]] = []
    for s in summaries:
        rows.append(
            {
                "league": s.league,
                "model": s.model,
                "edge_threshold": f"{float(s.edge_threshold):.3f}",
                "positive_buckets": " ".join(s.positive_buckets),
                "bets": int(s.bets),
                "total_pnl": f"{float(s.total_pnl):.6f}",
                "avg_pnl_per_bet": f"{float(s.avg_pnl_per_bet):.6f}",
            }
        )

    # Add combined rows (sum of per-model signals).
    leagues = sorted({s.league for s in summaries})
    for league in leagues:
        bets, pnl = _combined_summary(summaries, league=league, models=models)
        rows.append(
            {
                "league": league,
                "model": "COMBINED",
                "edge_threshold": rows[0]["edge_threshold"] if rows else "",
                "positive_buckets": "",
                "bets": int(bets),
                "total_pnl": f"{float(pnl):.6f}",
                "avg_pnl_per_bet": f"{(float(pnl) / float(bets)):.6f}" if bets else "",
            }
        )

    rows.sort(key=lambda r: (r["league"], str(r["model"])))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    if not LEDGER_DIR.exists():
        raise SystemExit(f"[error] daily ledger directory missing: {LEDGER_DIR}")

    if float(args.edge_threshold) <= 0:
        raise SystemExit("[error] --edge-threshold must be > 0")

    league_filter = _normalize_league_arg(args.league)
    models = [m.strip().lower() for m in (args.models or []) if m.strip()]
    if not models:
        raise SystemExit("[error] --models must be non-empty")

    games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=league_filter,
        models=models,
    )
    if not games:
        raise SystemExit("[error] no games found for the selected window")

    # Compute stats for per-league and overall.
    games_for_stats = list(games) + _as_overall(games)
    leagues = sorted({g.league for g in games_for_stats})

    stats_by_key = compute_bucket_stats(
        games_for_stats,
        models=models,
        edge_threshold=float(args.edge_threshold),
        buckets=QUADRANT_BUCKETS,
    )

    summaries = _policy_summaries(
        stats_by_key=stats_by_key,
        leagues=leagues,
        models=models,
        edge_threshold=float(args.edge_threshold),
        min_bets=int(args.min_bets),
        ev_threshold=float(args.ev_threshold),
    )

    print("\n=== Positive quadrant buckets policy (1 unit per model signal) ===")
    print(
        f"window={args.start_date}..{args.end_date} league={args.league} "
        f"t={float(args.edge_threshold):.3f} min_bets={int(args.min_bets)} ev_threshold={float(args.ev_threshold):.3f}"
    )
    by_league = {}
    for s in summaries:
        by_league.setdefault(s.league, []).append(s)
    for league in sorted(by_league):
        print(f"\n[{league}]")
        for s in sorted(by_league[league], key=lambda x: x.model):
            print(
                f"- {s.model:12s} positive={','.join(s.positive_buckets) or '∅':5s} "
                f"bets={s.bets:4d} total_pnl={s.total_pnl:7.3f} avg={s.avg_pnl_per_bet:6.3f}"
            )
        bets, pnl = _combined_summary(summaries, league=league, models=models)
        avg = (pnl / bets) if bets else 0.0
        print(f"- {'COMBINED':12s} bets={bets:4d} total_pnl={pnl:7.3f} avg={avg:6.3f}")

    if args.no_write:
        return

    prefix = (args.out_prefix or "").strip()
    if prefix and not prefix.endswith("_"):
        prefix = prefix + "_"
    tag = f"{(args.league or 'all').lower()}_{args.start_date.replace('-','')}_{args.end_date.replace('-','')}_t{float(args.edge_threshold):.3f}"
    out_stats = OUT_DIR / f"{prefix}positive_quadrant_bucket_stats_{tag}.csv"
    out_policy = OUT_DIR / f"{prefix}positive_quadrant_policy_summary_{tag}.csv"

    _write_bucket_stats_csv(
        out_stats,
        stats_by_key=stats_by_key,
        leagues=leagues,
        models=models,
        min_bets=int(args.min_bets),
        ev_threshold=float(args.ev_threshold),
        edge_threshold=float(args.edge_threshold),
    )
    _write_policy_summary_csv(out_policy, summaries, models=models)
    print(f"\n[ok] wrote -> {out_stats}")
    print(f"[ok] wrote -> {out_policy}")


if __name__ == "__main__":
    main()

