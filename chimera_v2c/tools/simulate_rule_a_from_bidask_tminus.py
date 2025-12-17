#!/usr/bin/env python
"""
Simulate Rule A using Kalshi bid/ask snapshots at a fixed T-minus (read-only).

This tool is an execution-realism upgrade vs mid-based backtests:
  - Uses exported candlestick snapshots (e.g. T-30) for BOTH teams' YES markets.
  - Determines whether "Kalshi favors home" from the HOME-YES mid at T-minus.
  - Executes the fade by buying AWAY-YES at the AWAY-YES ask (plus optional slippage).
  - Applies Kalshi taker fees (quadratic schedule, rounded up per order).

Rule A (fade-home only):
  - Only consider games where the market favors home: mid_home > 0.50
  - For a given model, trigger a fade if p_model_home < mid_home

Strategies:
  - BLIND_FADE_1U: Always fade home favorites (1 contract per qualifying game).
  - RULEA_MODEL_SEPARATE: For each qualifying model trigger, place a 1-contract order.
  - BLIND_PLUS_VOTES_AGG: 1 baseline contract + 1 per triggering model, aggregated into 1 order per game.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/simulate_rule_a_from_bidask_tminus.py \
    --start-date 2025-11-19 --end-date 2025-12-15 \
    --bidask-csv reports/market_snapshots/kalshi_bidask_tminus_20251119_20251215_m30.csv \
    --slippage-cents 0 1 2
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from chimera_v2c.lib import team_mapper
from chimera_v2c.src.kalshi_fees import taker_fee_dollars
from chimera_v2c.src.ledger_analysis import GameRow, LEDGER_DIR, load_games


MODELS: List[str] = ["v2c", "grok", "gemini", "gpt", "market_proxy", "moneypuck"]


@dataclass
class Totals:
    bets: int = 0
    units: int = 0
    gross_pnl: float = 0.0
    fees: float = 0.0
    risked: float = 0.0

    @property
    def net_pnl(self) -> float:
        return float(self.gross_pnl - self.fees)

    @property
    def roi_net(self) -> float:
        return 0.0 if self.risked <= 0 else float(self.net_pnl / self.risked)


def _clamp_price(p: float) -> float:
    return max(0.01, min(0.99, float(p)))


def _parse_matchup(*, league: str, matchup: str) -> Optional[Tuple[str, str]]:
    if "@" not in matchup:
        return None
    away_raw, home_raw = matchup.split("@", 1)
    away = team_mapper.normalize_team_code(away_raw.strip(), league)
    home = team_mapper.normalize_team_code(home_raw.strip(), league)
    if not away or not home:
        return None
    return away, home


def _away_pnl_per_contract(*, price_away: float, home_win: float) -> float:
    if home_win == 0.0:
        return 1.0 - float(price_away)
    return -float(price_away)


def _load_bidask_snapshot(path: Path) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
    """
    Keyed by (date, league, matchup, yes_team) -> {ask, bid, mid, spread}.
    """
    df = pd.read_csv(path)
    need = {"date", "league", "matchup", "yes_team", "yes_bid_cents", "yes_ask_cents", "mid", "spread", "ok"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise SystemExit(f"[error] bid/ask csv missing columns: {missing}")

    df = df[df["ok"] == 1].copy()
    out: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    for _, r in df.iterrows():
        date = str(r["date"]).strip()
        league = str(r["league"]).strip().lower()
        matchup = str(r["matchup"]).strip()
        yes_team = str(r["yes_team"]).strip().upper()
        if not date or not league or not matchup or not yes_team:
            continue
        try:
            bid = float(r["yes_bid_cents"]) / 100.0
            ask = float(r["yes_ask_cents"]) / 100.0
            mid = float(r["mid"])
            spread = float(r["spread"])
        except (TypeError, ValueError):
            continue
        out[(date, league, matchup, yes_team)] = {"bid": bid, "ask": ask, "mid": mid, "spread": spread}
    return out


def _is_qualifying_game(*, g: GameRow, mid_home: float) -> bool:
    if g.home_win is None or g.home_win == 0.5:
        return False
    return float(mid_home) > 0.5


def simulate(
    games: Iterable[GameRow],
    *,
    bidask: Dict[Tuple[str, str, str, str], Dict[str, float]],
    slippage_cents: int,
) -> List[Dict[str, object]]:
    slip = float(slippage_cents) / 100.0
    league_set = sorted({g.league for g in games})

    totals: Dict[Tuple[str, str, str], Totals] = {}

    def _tot(league: str, strat: str, model: str) -> Totals:
        key = (league, strat, model)
        if key not in totals:
            totals[key] = Totals()
        return totals[key]

    def _apply(league: str, strat: str, model: str, *, contracts: int, price: float, home_win: float) -> None:
        t = _tot(league, strat, model)
        t.bets += 1
        t.units += int(contracts)
        t.risked += float(price) * float(contracts)
        t.gross_pnl += float(contracts) * _away_pnl_per_contract(price_away=price, home_win=home_win)
        t.fees += taker_fee_dollars(contracts=int(contracts), price=float(price))

    missing_snapshot = 0
    missing_outcome = 0

    for g in games:
        if g.home_win is None:
            missing_outcome += 1
            continue
        parsed = _parse_matchup(league=g.league, matchup=g.matchup)
        if parsed is None:
            continue
        away, home = parsed
        date = g.date.strftime("%Y-%m-%d")

        home_row = bidask.get((date, g.league, f"{away}@{home}", home))
        away_row = bidask.get((date, g.league, f"{away}@{home}", away))
        if home_row is None or away_row is None:
            missing_snapshot += 1
            continue

        mid_home = float(home_row["mid"])
        if not _is_qualifying_game(g=g, mid_home=mid_home):
            continue

        price_away = _clamp_price(float(away_row["ask"]) + slip)
        home_win = float(g.home_win)

        # Blind baseline: 1 contract per qualifying game (league + overall)
        _apply(g.league, "BLIND_FADE_1U", "ALL", contracts=1, price=price_away, home_win=home_win)
        _apply("overall", "BLIND_FADE_1U", "ALL", contracts=1, price=price_away, home_win=home_win)

        votes = 0
        triggering: List[str] = []
        for m in MODELS:
            p = g.probs.get(m)
            if p is None:
                continue
            if float(p) < mid_home:
                votes += 1
                triggering.append(m)

                # Separate per-model order for slicing.
                _apply(g.league, "RULEA_MODEL_SEPARATE", m, contracts=1, price=price_away, home_win=home_win)
                _apply("overall", "RULEA_MODEL_SEPARATE", m, contracts=1, price=price_away, home_win=home_win)

                # All-models-as-separate-orders totals (fee rounds per order).
                _apply(g.league, "RULEA_ALLMODELS_SEPARATE", "ALL", contracts=1, price=price_away, home_win=home_win)
                _apply("overall", "RULEA_ALLMODELS_SEPARATE", "ALL", contracts=1, price=price_away, home_win=home_win)

        # Aggregated: one order per game with (1 + votes) contracts.
        _apply(g.league, "BLIND_PLUS_VOTES_AGG", "ALL", contracts=1 + votes, price=price_away, home_win=home_win)
        _apply("overall", "BLIND_PLUS_VOTES_AGG", "ALL", contracts=1 + votes, price=price_away, home_win=home_win)

    rows: List[Dict[str, object]] = []
    for league in league_set + ["overall"]:
        for strat in ["BLIND_FADE_1U", "RULEA_MODEL_SEPARATE", "RULEA_ALLMODELS_SEPARATE", "BLIND_PLUS_VOTES_AGG"]:
            for model in (MODELS if strat == "RULEA_MODEL_SEPARATE" else ["ALL"]):
                t = totals.get((league, strat, model))
                if t is None:
                    continue
                rows.append(
                    {
                        "league": league,
                        "strategy": strat,
                        "model": model,
                        "slippage_cents": int(slippage_cents),
                        "bets": int(t.bets),
                        "units": int(t.units),
                        "risked": round(float(t.risked), 6),
                        "gross_pnl": round(float(t.gross_pnl), 6),
                        "fees": round(float(t.fees), 6),
                        "net_pnl": round(float(t.net_pnl), 6),
                        "roi_net": round(float(t.roi_net), 6),
                        "missing_snapshot_games": int(missing_snapshot),
                        "missing_outcome_games": int(missing_outcome),
                    }
                )
    return rows


def _write_rows(out_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate Rule A using bid/ask snapshots at T-minus (read-only).")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD (inclusive, ledger filename date).")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive, ledger filename date).")
    ap.add_argument("--bidask-csv", required=True, help="Path to export_kalshi_bidask_tminus.py output CSV.")
    ap.add_argument(
        "--slippage-cents",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Optional extra slippage to add on top of the away ask (default: 0 1 2).",
    )
    ap.add_argument("--out", help="Optional output CSV path (default under reports/thesis_summaries).")
    args = ap.parse_args()

    bidask = _load_bidask_snapshot(Path(args.bidask_csv))
    games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        models=MODELS,
    )
    if not games:
        raise SystemExit("[error] no games loaded from daily ledgers for the given window")

    rows: List[Dict[str, object]] = []
    for cents in args.slippage_cents:
        rows.extend(simulate(games, bidask=bidask, slippage_cents=int(cents)))

    # Add window metadata at write time to avoid duplicating during sim.
    out_rows: List[Dict[str, object]] = []
    for r in rows:
        out_rows.append(
            {
                "window_start": args.start_date,
                "window_end": args.end_date,
                "bidask_csv": str(args.bidask_csv),
                **r,
            }
        )

    out_path = Path(args.out) if args.out else Path(
        f"reports/thesis_summaries/ruleA_bidask_tminus_by_league_model_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}.csv"
    )
    _write_rows(out_path, out_rows)
    print(f"[ok] wrote {out_path} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()

