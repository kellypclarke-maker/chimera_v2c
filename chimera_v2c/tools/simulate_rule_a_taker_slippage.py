#!/usr/bin/env python
"""
Simulate Rule A execution with taker fees and fixed slippage vs mid (read-only).

Rule A:
  - Only consider games where Kalshi mid favors home: p_mid_home > 0.50
  - Fade home (buy away) when a model says home is overpriced: p_model_home < p_mid_home

Execution model:
  - We assume taker execution on the away contract at:
      price_away = clamp((1 - p_mid_home) + slippage)
    where slippage is +$0.01 or +$0.02 (1–2 cents above the away mid).
  - Gross PnL is computed at that execution price (1 contract = $1 payout).
  - Fees use Kalshi's quadratic taker fee schedule (Oct 2025), rounded up per order:
      fee = ceil_to_cent(0.07 * contracts * P * (1-P))
    using P = executed contract price_away.

Strategies:
  - BLIND_FADE_1U: Always fade home favorites (1 contract per qualifying game).
  - RULEA_ALLMODELS_SEPARATE: For each qualifying model trigger, place a separate 1-contract order.
  - BLIND_PLUS_VOTES_AGG: 1 baseline contract + 1 per triggering model, aggregated into a single order per game.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/simulate_rule_a_taker_slippage.py \
    --start-date 2025-12-01 --end-date 2025-12-15 --slippage-cents 1 2
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from chimera_v2c.src.kalshi_fees import taker_fee_dollars
from chimera_v2c.src.ledger_analysis import GameRow, LEDGER_DIR, load_games


MODELS: List[str] = ["v2c", "grok", "gemini", "gpt", "market_proxy", "moneypuck"]


@dataclass
class Totals:
    units: int = 0
    gross_pnl: float = 0.0
    fees: float = 0.0

    @property
    def net_pnl(self) -> float:
        return float(self.gross_pnl - self.fees)

    @property
    def avg_net(self) -> float:
        return 0.0 if self.units == 0 else float(self.net_pnl / self.units)


def _clamp_price(p: float) -> float:
    return max(0.01, min(0.99, float(p)))


def _away_price_from_mid(*, mid_home: float, slippage: float) -> float:
    return _clamp_price((1.0 - float(mid_home)) + float(slippage))


def _away_pnl_per_contract(*, price_away: float, home_win: float) -> float:
    # Buying away YES at price_away:
    # - away wins (home_win==0): + (1 - price_away)
    # - away loses (home_win==1): - price_away
    if home_win == 0.0:
        return 1.0 - float(price_away)
    return -float(price_away)


def _is_qualifying_game(g: GameRow) -> bool:
    if g.home_win is None or g.home_win == 0.5:
        return False
    if g.kalshi_mid is None:
        return False
    return float(g.kalshi_mid) > 0.5


def _trigger_votes(g: GameRow) -> int:
    if g.kalshi_mid is None:
        return 0
    mid = float(g.kalshi_mid)
    votes = 0
    for m in MODELS:
        p = g.probs.get(m)
        if p is None:
            continue
        if float(p) < mid:
            votes += 1
    return votes


def simulate(
    games: Iterable[GameRow],
    *,
    slippage_cents: int,
) -> Dict[str, Totals]:
    slip = float(slippage_cents) / 100.0
    out: Dict[str, Totals] = {
        "BLIND_FADE_1U": Totals(),
        "RULEA_ALLMODELS_SEPARATE": Totals(),
        "BLIND_PLUS_VOTES_AGG": Totals(),
    }
    for g in games:
        if not _is_qualifying_game(g):
            continue
        mid = float(g.kalshi_mid or 0.0)
        price_away = _away_price_from_mid(mid_home=mid, slippage=slip)
        pnl_unit = _away_pnl_per_contract(price_away=price_away, home_win=float(g.home_win or 0.0))

        # Blind baseline: 1 contract per game.
        base = out["BLIND_FADE_1U"]
        base.units += 1
        base.gross_pnl += pnl_unit
        base.fees += taker_fee_dollars(contracts=1, price=price_away)

        # Model votes
        votes = _trigger_votes(g)

        # Rule A models: each model order is a separate 1-contract order (fee rounds per order).
        sep = out["RULEA_ALLMODELS_SEPARATE"]
        for _ in range(votes):
            sep.units += 1
            sep.gross_pnl += pnl_unit
            sep.fees += taker_fee_dollars(contracts=1, price=price_away)

        # Blind + votes: aggregated into a single order per game.
        agg = out["BLIND_PLUS_VOTES_AGG"]
        units = 1 + votes
        agg.units += units
        agg.gross_pnl += units * pnl_unit
        agg.fees += taker_fee_dollars(contracts=int(units), price=price_away)

    return out


def _write_rows(out_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate Rule A with taker fees + fixed slippage vs mid (read-only).")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD (inclusive).")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive).")
    ap.add_argument(
        "--slippage-cents",
        nargs="+",
        type=int,
        default=[1, 2],
        help="One or more slippage values in cents to add to the away mid (default: 1 2).",
    )
    ap.add_argument("--out", help="Optional output CSV path (default under reports/thesis_summaries).")
    args = ap.parse_args()

    games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        models=MODELS + ["kalshi_mid"],
    )
    if not games:
        raise SystemExit("[error] no games loaded from daily ledgers for the given window")

    leagues = sorted({g.league for g in games})
    rows: List[Dict[str, object]] = []

    for cents in args.slippage_cents:
        for league in leagues + ["overall"]:
            subset = games if league == "overall" else [g for g in games if g.league == league]
            if not subset:
                continue
            sims = simulate(subset, slippage_cents=int(cents))
            for strat, tot in sims.items():
                rows.append(
                    {
                        "window_start": args.start_date,
                        "window_end": args.end_date,
                        "league": league,
                        "strategy": strat,
                        "slippage_cents": int(cents),
                        "units": int(tot.units),
                        "gross_pnl": round(float(tot.gross_pnl), 6),
                        "fees": round(float(tot.fees), 6),
                        "net_pnl": round(float(tot.net_pnl), 6),
                        "avg_net_per_unit": round(float(tot.avg_net), 6),
                    }
                )

    out_path = Path(args.out) if args.out else Path(
        f"reports/thesis_summaries/ruleA_taker_slippage_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}.csv"
    )
    _write_rows(out_path, rows)
    print(f"[ok] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()

