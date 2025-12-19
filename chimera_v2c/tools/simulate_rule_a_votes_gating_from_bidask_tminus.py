#!/usr/bin/env python
"""
Simulate Rule A vote gating variants using Kalshi bid/ask snapshots at fixed T-minus (read-only).

This tool is intended to answer: can more conservative gating (e.g., I/J-tier gating)
beat the current production baseline "BLIND + VOTES agg" under execution realism?

Execution realism:
  - Qualifying games: home YES mid at anchor > 0.50.
  - Trade execution: buy AWAY YES at the AWAY YES ask (+ optional slippage).
  - Fees: Kalshi taker fees (quadratic, rounded up per order).
  - Aggregation: one order per game for the chosen contract count.

Vote definition (per model m):
  - Requires BOTH:
      delta_ok: (mid_home - p_home_m) >= vote_delta_m
      edge_ok:  expected_edge_net_per_contract(p_home_m, price_away) >= vote_edge_m
  - I-trigger (flip): p_home_m < 0.50 AND edge_ok (delta is ignored for I).

Strategies compared:
  - BLIND_FADE_1U: 1 contract on every qualifying game.
  - BLIND_PLUS_VOTES: 1 + votes contracts on every qualifying game (production-style sizing).
  - BLIND_PLUS_I: 1 + I-flip votes on every qualifying game.
  - BLIND_PLUS_WEIGHTED_FLIP2: 1 + weighted_votes where a flip (I-trigger) counts as 2 units.
  - BLIND_PLUS_J2: 1 + votes on every qualifying game, but only add units when votes >= 2.
  - BLIND_PLUS_IJ_EXTRA: 1 + votes_union on every qualifying game, but only add units when (any I-trigger) OR (votes >= 2).
  - MIN1_VOTE_ONLY: trade only if (votes >= 1 OR any I-trigger); size = 1 + votes_union.
  - IJ_GATED: trade only if (any I-trigger) OR (votes >= 2); size = 1 + votes_union.
  - I_ONLY: trade only if any I-trigger; size = 1 + i_votes.
  - J_ONLY: trade only if votes >= 2; size = 1 + votes.

Notes:
  - votes_union = union(vote_models, i_models) so a flip model contributes a unit even
    if its p_home is not below mid_home.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/simulate_rule_a_votes_gating_from_bidask_tminus.py \
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

from chimera_v2c.src.ledger_analysis import GameRow, LEDGER_DIR, load_games
from chimera_v2c.src.rule_a_unit_policy import (
    expected_edge_net_per_contract,
    iter_rule_a_games,
    load_bidask_csv,
    net_pnl_taker,
)


DEFAULT_MODELS = ["v2c", "grok", "gemini", "gpt", "market_proxy", "moneypuck"]


def _parse_model_thresholds(items: Sequence[str], *, arg_name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for raw in items:
        s = (raw or "").strip()
        if not s:
            continue
        if ":" not in s:
            raise SystemExit(f"[error] invalid {arg_name} '{raw}' (expected model:value)")
        model, val = s.split(":", 1)
        model = model.strip()
        try:
            v = float(val.strip())
        except ValueError as exc:
            raise SystemExit(f"[error] invalid {arg_name} '{raw}' (value must be float)") from exc
        if model:
            out[model] = v
    return out


def _clamp_int(value: int, *, name: str, lo: int, hi: int) -> int:
    x = int(value)
    if x < lo or x > hi:
        raise SystemExit(f"[error] {name} must be in [{lo}, {hi}]")
    return x


@dataclass
class Totals:
    bets: int = 0
    contracts: int = 0
    risked: float = 0.0
    gross_pnl: float = 0.0
    fees: float = 0.0

    @property
    def net_pnl(self) -> float:
        return float(self.gross_pnl - self.fees)

    @property
    def roi_net(self) -> float:
        return 0.0 if self.risked <= 0 else float(self.net_pnl / self.risked)


def _apply_totals(t: Totals, *, contracts: int, price_away: float, home_win: int) -> None:
    if int(contracts) <= 0:
        return
    gross, fees, risked = net_pnl_taker(contracts=int(contracts), price_away=float(price_away), home_win=int(home_win))
    t.bets += 1
    t.contracts += int(contracts)
    t.risked += float(risked)
    t.gross_pnl += float(gross)
    t.fees += float(fees)


def _vote_sets_for_game(
    g,
    *,
    models: Sequence[str],
    vote_delta_default: float,
    vote_edge_default: float,
    vote_delta_by_model: Dict[str, float],
    vote_edge_by_model: Dict[str, float],
) -> Tuple[List[str], List[str], List[str]]:
    vote_models: List[str] = []
    i_models: List[str] = []

    for m in models:
        p = g.probs.get(str(m))
        if p is None:
            continue
        edge_thr = float(vote_edge_by_model.get(str(m), vote_edge_default))
        edge = expected_edge_net_per_contract(p_home=float(p), price_away=float(g.price_away))
        edge_ok = float(edge) >= edge_thr
        if float(p) < 0.5 and edge_ok:
            i_models.append(str(m))

        delta_thr = float(vote_delta_by_model.get(str(m), vote_delta_default))
        delta_ok = (float(g.mid_home) - float(p)) >= delta_thr
        if delta_ok and edge_ok:
            vote_models.append(str(m))

    vote_models = sorted(set(vote_models))
    i_models = sorted(set(i_models))
    votes_union = sorted(set(vote_models) | set(i_models))
    return vote_models, i_models, votes_union


def simulate_window(
    games: Iterable[GameRow],
    *,
    bidask_csv: Path,
    slippage_cents: int,
    models: Sequence[str],
    cap_units: int,
    vote_delta_default: float,
    vote_edge_default: float,
    vote_delta_by_model: Dict[str, float],
    vote_edge_by_model: Dict[str, float],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Returns (summary_rows, trade_rows).
    """
    bidask = load_bidask_csv(bidask_csv)
    ra = iter_rule_a_games(
        games,
        bidask=bidask,
        models=list(models),
        slippage_cents=int(slippage_cents),
        require_outcome=True,
    )
    if not ra:
        raise SystemExit("[error] no Rule-A qualifying graded games found for window + snapshot")

    leagues = sorted({g.league for g in ra})
    cap = int(cap_units)

    totals: Dict[Tuple[str, str], Totals] = {}

    def tot(league: str, strat: str) -> Totals:
        key = (league, strat)
        if key not in totals:
            totals[key] = Totals()
        return totals[key]

    trade_rows: List[Dict[str, object]] = []

    for g in ra:
        if g.home_win is None:
            continue
        vote_models, i_models, votes_union = _vote_sets_for_game(
            g,
            models=models,
            vote_delta_default=float(vote_delta_default),
            vote_edge_default=float(vote_edge_default),
            vote_delta_by_model=vote_delta_by_model,
            vote_edge_by_model=vote_edge_by_model,
        )
        n_votes = int(len(vote_models))
        n_i = int(len(i_models))
        n_union = int(len(votes_union))
        i_set = set(i_models)
        weighted_votes = sum(2 if m in i_set else 1 for m in votes_union)

        # Strategy contract counts
        blind = 1
        blind_votes = min(cap, 1 + n_votes)
        blind_i = min(cap, 1 + n_i)
        blind_weighted_flip2 = min(cap, 1 + int(weighted_votes))
        blind_j2 = min(cap, 1 + n_votes) if (n_votes >= 2) else 1
        blind_ij_extra = min(cap, 1 + n_union) if (n_i >= 1 or n_votes >= 2) else 1

        min1_vote_only = min(cap, 1 + n_union) if (n_union >= 1) else 0
        ij_gated = min(cap, 1 + n_union) if (n_i >= 1 or n_votes >= 2) else 0
        i_only = min(cap, 1 + n_i) if (n_i >= 1) else 0
        j_only = min(cap, 1 + n_votes) if (n_votes >= 2) else 0

        for league in (g.league, "overall"):
            _apply_totals(tot(league, "BLIND_FADE_1U"), contracts=blind, price_away=g.price_away, home_win=int(g.home_win))
            _apply_totals(
                tot(league, "BLIND_PLUS_VOTES"),
                contracts=blind_votes,
                price_away=g.price_away,
                home_win=int(g.home_win),
            )
            _apply_totals(
                tot(league, "BLIND_PLUS_I"),
                contracts=blind_i,
                price_away=g.price_away,
                home_win=int(g.home_win),
            )
            _apply_totals(
                tot(league, "BLIND_PLUS_WEIGHTED_FLIP2"),
                contracts=blind_weighted_flip2,
                price_away=g.price_away,
                home_win=int(g.home_win),
            )
            _apply_totals(
                tot(league, "BLIND_PLUS_J2"),
                contracts=blind_j2,
                price_away=g.price_away,
                home_win=int(g.home_win),
            )
            _apply_totals(
                tot(league, "BLIND_PLUS_IJ_EXTRA"),
                contracts=blind_ij_extra,
                price_away=g.price_away,
                home_win=int(g.home_win),
            )
            _apply_totals(
                tot(league, "MIN1_VOTE_ONLY"),
                contracts=min1_vote_only,
                price_away=g.price_away,
                home_win=int(g.home_win),
            )
            _apply_totals(
                tot(league, "IJ_GATED"),
                contracts=ij_gated,
                price_away=g.price_away,
                home_win=int(g.home_win),
            )
            _apply_totals(
                tot(league, "I_ONLY"),
                contracts=i_only,
                price_away=g.price_away,
                home_win=int(g.home_win),
            )
            _apply_totals(
                tot(league, "J_ONLY"),
                contracts=j_only,
                price_away=g.price_away,
                home_win=int(g.home_win),
            )

        trade_rows.append(
            {
                "date": g.date,
                "league": g.league,
                "matchup": g.matchup,
                "mid_home": round(float(g.mid_home), 6),
                "price_away": round(float(g.price_away), 6),
                "home_win": int(g.home_win),
                "votes": n_votes,
                "i_votes": n_i,
                "votes_union": n_union,
                "weighted_votes": int(weighted_votes),
                "vote_models": ",".join(vote_models),
                "i_models": ",".join(i_models),
                "union_models": ",".join(votes_union),
                "blind_contracts": int(blind),
                "blind_votes_contracts": int(blind_votes),
                "blind_i_contracts": int(blind_i),
                "blind_weighted_flip2_contracts": int(blind_weighted_flip2),
                "blind_j2_contracts": int(blind_j2),
                "blind_ij_extra_contracts": int(blind_ij_extra),
                "min1_vote_only_contracts": int(min1_vote_only),
                "ij_gated_contracts": int(ij_gated),
                "i_only_contracts": int(i_only),
                "j_only_contracts": int(j_only),
            }
        )

    summary_rows: List[Dict[str, object]] = []
    for league in leagues + ["overall"]:
        for strat in [
            "BLIND_FADE_1U",
            "BLIND_PLUS_VOTES",
            "BLIND_PLUS_I",
            "BLIND_PLUS_WEIGHTED_FLIP2",
            "BLIND_PLUS_J2",
            "BLIND_PLUS_IJ_EXTRA",
            "MIN1_VOTE_ONLY",
            "IJ_GATED",
            "I_ONLY",
            "J_ONLY",
        ]:
            t = totals.get((league, strat))
            if t is None:
                continue
            summary_rows.append(
                {
                    "league": league,
                    "strategy": strat,
                    "slippage_cents": int(slippage_cents),
                    "cap_units": int(cap_units),
                    "vote_delta_default": float(vote_delta_default),
                    "vote_edge_default": float(vote_edge_default),
                    "models": ",".join(list(models)),
                    "bets": int(t.bets),
                    "contracts": int(t.contracts),
                    "risked": round(float(t.risked), 6),
                    "fees": round(float(t.fees), 6),
                    "net_pnl": round(float(t.net_pnl), 6),
                    "roi_net": round(float(t.roi_net), 6),
                }
            )

    return summary_rows, trade_rows


def _write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Simulate Rule A vote gating variants using bid/ask snapshots (read-only).")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive.")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive.")
    ap.add_argument("--bidask-csv", required=True, help="Path to export_kalshi_bidask_tminus.py output CSV.")
    ap.add_argument(
        "--slippage-cents",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Extra slippage to add on top of away ask (default: 0 1 2).",
    )
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model columns to use (default: v2c grok gemini gpt market_proxy moneypuck).")
    ap.add_argument("--cap-units", type=int, default=10, help="Max contracts per game (default: 10).")
    ap.add_argument("--vote-delta", type=float, default=0.0, help="Default delta gate (mid_home - p_home) >= this (default: 0).")
    ap.add_argument("--vote-edge", type=float, default=0.0, help="Default edge gate (fee-aware edge_net) >= this (default: 0).")
    ap.add_argument("--model-delta", nargs="*", default=[], help="Per-model delta overrides, e.g. grok:0.03 gpt:0.07.")
    ap.add_argument("--model-edge", nargs="*", default=[], help="Per-model edge overrides, e.g. market_proxy:0.01.")
    ap.add_argument("--out-dir", default="reports/thesis_summaries", help="Output directory (default: reports/thesis_summaries).")
    ap.add_argument("--no-write", action="store_true", help="Print overall summary only; do not write files.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cap = _clamp_int(args.cap_units, name="--cap-units", lo=1, hi=10_000)

    vote_delta_by_model = _parse_model_thresholds(list(args.model_delta), arg_name="--model-delta")
    vote_edge_by_model = _parse_model_thresholds(list(args.model_edge), arg_name="--model-edge")

    bidask_path = Path(args.bidask_csv)
    if not bidask_path.exists():
        raise SystemExit(f"[error] missing bid/ask csv: {bidask_path}")

    games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        models=list(args.models),
    )
    if not games:
        raise SystemExit("[error] no games loaded from daily ledgers for the given window")

    all_summary: List[Dict[str, object]] = []
    all_trades: List[Dict[str, object]] = []
    for cents in args.slippage_cents:
        summary_rows, trade_rows = simulate_window(
            games,
            bidask_csv=Path(args.bidask_csv),
            slippage_cents=int(cents),
            models=list(args.models),
            cap_units=cap,
            vote_delta_default=float(args.vote_delta),
            vote_edge_default=float(args.vote_edge),
            vote_delta_by_model=vote_delta_by_model,
            vote_edge_by_model=vote_edge_by_model,
        )
        all_summary.extend(summary_rows)
        all_trades.extend(trade_rows)

    # Add metadata to output filenames.
    tag = f"{bidask_path.stem}_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}"
    out_dir = Path(args.out_dir)

    if args.no_write:
        for r in all_summary:
            if r["league"] == "overall" and r["slippage_cents"] == 1:
                print(r)
        return

    _write_rows(out_dir / f"ruleA_votes_gating_summary_{tag}.csv", all_summary)
    _write_rows(out_dir / f"ruleA_votes_gating_trades_{tag}.csv", all_trades)
    print(f"[ok] wrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
