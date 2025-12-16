"""
Analyze Grok-vs-Kalshi quadrant regimes with optional consensus sizing.

This is a read-only analysis tool over canonical daily ledgers.

Concept
-------
Given a baseline market home probability p_mid = kalshi_mid and Grok p_home:

Trade only when |p_grok - p_mid| >= t.

Buckets (consistent with analyze_rulebook_quadrants.py):
  - A: market home-fav (p_mid>=0.5) + Grok says home overpriced (p_grok<=p_mid-t)  -> buy away
  - B: market home-fav (p_mid>=0.5) + Grok says home underpriced (p_grok>=p_mid+t) -> buy home
  - C: market away-fav (p_mid< 0.5) + Grok says home overpriced (p_grok<=p_mid-t)  -> buy away
  - D: market away-fav (p_mid< 0.5) + Grok says home underpriced (p_grok>=p_mid+t) -> buy home

Consensus sizing (optional)
---------------------------
Let Grok determine the trade side (home/away). Count how many confirmers
(default v2c/gemini/gpt) agree with Grok's trade direction, requiring the same
edge threshold t for the confirmer. Size units by (1,3,5) for 1/2/3+ models.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.rulebook_quadrants import pnl_buy_away, pnl_buy_home


DEFAULT_CONFIRMERS = ["v2c", "gemini", "gpt"]
DEFAULT_THRESHOLDS = [0.00, 0.02, 0.05]
DEFAULT_BUCKETS = ["A", "B", "C", "D"]


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
        description="Analyze Grok edge buckets A/B/C/D vs kalshi_mid with optional consensus sizing.",
    )
    ap.add_argument("--days", type=int, default=0, help="Most recent N ledger days (default: all).")
    ap.add_argument("--start-date", help="YYYY-MM-DD inclusive; overrides --days.")
    ap.add_argument("--end-date", help="YYYY-MM-DD inclusive; overrides --days.")
    ap.add_argument(
        "--edge-threshold",
        type=float,
        default=0.02,
        help="Single threshold t where trade requires |p_grok - p_mid| >= t (default: 0.02).",
    )
    ap.add_argument(
        "--edge-thresholds",
        nargs="+",
        type=float,
        help="Optional list of thresholds to sweep; overrides --edge-threshold.",
    )
    ap.add_argument(
        "--preset-thresholds",
        action="store_true",
        help=f"Shortcut for --edge-thresholds {' '.join(str(x) for x in DEFAULT_THRESHOLDS)}.",
    )
    ap.add_argument(
        "--confirmers",
        nargs="+",
        default=DEFAULT_CONFIRMERS,
        help="Models that can confirm Grok direction (default: v2c gemini gpt).",
    )
    ap.add_argument(
        "--confirmer-positive-bucket-only",
        action="store_true",
        help="Only count a confirmer toward consensus sizing if that confirmer's historical PnL in that bucket is positive "
        "for the same league (computed in-sample over the analyzed window; requires --confirmer-min-bets).",
    )
    ap.add_argument(
        "--confirmer-min-bets",
        type=int,
        default=10,
        help="Minimum bets required for a confirmer bucket to be eligible for confirmer-positive filtering (default: 10).",
    )
    ap.add_argument(
        "--confirmer-min-avg-pnl",
        type=float,
        default=0.0,
        help="Minimum avg_pnl_per_bet for a confirmer bucket to be considered positive (default: 0.0).",
    )
    ap.add_argument(
        "--sizing",
        nargs=3,
        type=int,
        default=(1, 3, 5),
        metavar=("U1", "U2", "U3"),
        help="Consensus unit sizing for (1 model / 2 models / 3+ models). Default: 1 3 5.",
    )
    ap.add_argument(
        "--out",
        default="reports/thesis_summaries/grok_quadrant_consensus.csv",
        help="Output CSV path (default: reports/thesis_summaries/grok_quadrant_consensus.csv).",
    )
    ap.add_argument("--no-write", action="store_true", help="Print summary only; do not write CSV.")
    return ap.parse_args()


def _trade_bucket(*, p_mid: float, p_grok: float, t: float) -> Optional[Tuple[str, str]]:
    """
    Return (bucket, trade_side) or None if no trade.
    trade_side is 'home' or 'away' YES contract bought at p_mid.
    """
    if abs(p_grok - p_mid) < t:
        return None
    if p_grok >= p_mid + t:
        # buy home
        return ("B" if p_mid >= 0.5 else "D", "home")
    if p_grok <= p_mid - t:
        # buy away
        return ("A" if p_mid >= 0.5 else "C", "away")
    return None


def _confirmer_agrees(*, p_mid: float, p_model: float, t: float, trade_side: str) -> bool:
    if abs(p_model - p_mid) < t:
        return False
    if trade_side == "home":
        return p_model >= p_mid + t
    return p_model <= p_mid - t


def _units_for_models(n_models: int, sizing: Tuple[int, int, int]) -> int:
    u1, u2, u3 = sizing
    if n_models <= 0:
        return 0
    if n_models == 1:
        return u1
    if n_models == 2:
        return u2
    return u3


def _pnl_for_trade(trade_side: str, p_mid: float, home_win: float) -> float:
    return pnl_buy_home(p_mid=p_mid, home_win=home_win) if trade_side == "home" else pnl_buy_away(
        p_mid=p_mid, home_win=home_win
    )


def _iter_graded_games(games: Iterable[GameRow]) -> Iterable[GameRow]:
    for g in games:
        if g.home_win is None or g.home_win == 0.5:
            continue
        if g.kalshi_mid is None:
            continue
        yield g


def main() -> None:
    args = parse_args()

    thresholds: List[float]
    if args.preset_thresholds:
        thresholds = list(DEFAULT_THRESHOLDS)
    elif args.edge_thresholds:
        thresholds = list(args.edge_thresholds)
    else:
        thresholds = [float(args.edge_threshold)]

    days = None if (args.start_date or args.end_date) else (args.days if args.days and args.days > 0 else None)

    needed_models = sorted(set(["grok"] + list(args.confirmers) + ["market_proxy"]))
    games = load_games(
        daily_dir=LEDGER_DIR,
        days=days,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=None,
        models=needed_models,
    )
    if not games:
        raise SystemExit("[error] no games found for given filters")

    leagues = sorted({g.league for g in games} | {"overall"})

    # stats[(threshold, league, bucket, strategy)] -> Totals
    stats: Dict[Tuple[float, str, str, str], Totals] = {}

    def add_stat(threshold: float, league: str, bucket: str, strategy: str, units: int, pnl: float) -> None:
        key = (threshold, league, bucket, strategy)
        t = stats.setdefault(key, Totals())
        t.bets += 1
        t.units += units
        t.total_pnl += pnl

    for threshold in thresholds:
        t = float(threshold)

        # Optional: compute which confirmers are eligible per (league,bucket) based on their own in-sample
        # performance when they would trigger the same bucket trade.
        confirmer_allowed: Dict[Tuple[str, str, str], bool] = {}
        if args.confirmer_positive_bucket_only:
            confirmer_totals: Dict[Tuple[str, str, str], Totals] = {}
            for g in _iter_graded_games(games):
                p_mid = float(g.kalshi_mid)
                for confirmer in args.confirmers:
                    p_model = g.probs.get(confirmer)
                    if p_model is None:
                        continue
                    trade = _trade_bucket(p_mid=p_mid, p_grok=float(p_model), t=t)
                    if trade is None:
                        continue
                    bucket, trade_side = trade
                    pnl_1u = _pnl_for_trade(trade_side, p_mid=p_mid, home_win=float(g.home_win))
                    for league in (g.league, "overall"):
                        key = (league, confirmer, bucket)
                        tot = confirmer_totals.setdefault(key, Totals())
                        tot.bets += 1
                        tot.units += 1
                        tot.total_pnl += pnl_1u
            for (league, confirmer, bucket), tot in confirmer_totals.items():
                ok = tot.bets >= int(args.confirmer_min_bets) and tot.avg_pnl_per_bet >= float(args.confirmer_min_avg_pnl)
                confirmer_allowed[(league, confirmer, bucket)] = ok

        for g in _iter_graded_games(games):
            p_mid = float(g.kalshi_mid)
            p_grok = g.probs.get("grok")
            if p_grok is None:
                continue
            p_grok = float(p_grok)
            trade = _trade_bucket(p_mid=p_mid, p_grok=p_grok, t=t)
            if trade is None:
                continue
            bucket, trade_side = trade

            # grok-only: 1 unit
            pnl_1u = _pnl_for_trade(trade_side, p_mid=p_mid, home_win=float(g.home_win))
            for league in (g.league, "overall"):
                add_stat(t, league, bucket, "grok_only", 1, pnl_1u)

            # consensus: size by agreement count
            agree = 0
            for m in args.confirmers:
                pm = g.probs.get(m)
                if pm is None:
                    continue
                if not _confirmer_agrees(p_mid=p_mid, p_model=float(pm), t=t, trade_side=trade_side):
                    continue
                if args.confirmer_positive_bucket_only:
                    # Only count confirmers that have positive in-sample performance in this bucket for the same league.
                    if not confirmer_allowed.get((g.league, m, bucket), False):
                        continue
                agree += 1
            n_models = 1 + agree
            units = _units_for_models(n_models, sizing=tuple(args.sizing))
            for league in (g.league, "overall"):
                add_stat(t, league, bucket, "grok_consensus", units, pnl_1u * units)

    out_rows: List[Dict[str, object]] = []
    for (t, league, bucket, strategy), totals in sorted(
        stats.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])
    ):
        out_rows.append(
            {
                "edge_threshold": f"{t:.3f}",
                "league": league,
                "bucket": bucket,
                "strategy": strategy,
                "confirmers": ",".join(args.confirmers),
                "sizing": f"{args.sizing[0]}-{args.sizing[1]}-{args.sizing[2]}",
                "confirmer_positive_bucket_only": str(bool(args.confirmer_positive_bucket_only)),
                "confirmer_min_bets": int(args.confirmer_min_bets),
                "confirmer_min_avg_pnl": f"{float(args.confirmer_min_avg_pnl):.6f}",
                "bets": totals.bets,
                "units": totals.units,
                "total_pnl": f"{totals.total_pnl:.6f}",
                "avg_pnl_per_bet": f"{totals.avg_pnl_per_bet:.6f}",
                "avg_pnl_per_unit": f"{totals.avg_pnl_per_unit:.6f}",
            }
        )

    if not args.no_write:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"[info] wrote {len(out_rows)} rows -> {out_path}")

    # Print quick overall summary for a single threshold (or the first)
    t0 = float(thresholds[0])
    for league in leagues:
        for strategy in ("grok_only", "grok_consensus"):
            totals_by_bucket: Dict[str, Totals] = {}
            for b in DEFAULT_BUCKETS:
                totals_by_bucket[b] = stats.get((t0, league, b, strategy), Totals())
            total_units = sum(v.units for v in totals_by_bucket.values())
            total_pnl = sum(v.total_pnl for v in totals_by_bucket.values())
            if total_units == 0:
                continue
            avg_per_unit = total_pnl / total_units if total_units else 0.0
            print(
                f"[summary] t={t0:.3f} league={league} strategy={strategy} "
                f"units={total_units} pnl={total_pnl:+.3f} avg/unit={avg_per_unit:+.3f}"
            )


if __name__ == "__main__":
    main()
