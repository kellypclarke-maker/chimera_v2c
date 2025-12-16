#!/usr/bin/env python
"""
Evaluate a "positive-only bucket" trading policy per model vs Kalshi mid.

Idea:
  - Build a per-model rulebook from a training window:
      * which p_yes buckets have positive realized EV/ROI vs Kalshi mid
  - Then simulate a target date where trades are only allowed if the trade's bucket is in the model's
    positive bucket set (and has enough samples in training).

This tool is READ-ONLY on daily ledgers.

Modes:
  - --mode simple (default):
      * 1 contract per eligible signal
      * trade direction is: if p_home_model > kalshi_mid => buy HOME; else buy AWAY
      * bucket is based on p_yes_model for the side you buy (home p or away (1-p))
      * PnL per contract:
          win:  (1 - price)
          loss: -price
  - --mode doctrine:
      * doctrine-style gating + Kelly sizing (from config), priced at mid
      * bucket profitability is measured as ROI on cost (profit / cost)

Notes:
  - This is an offline simulator (no fill model, no maker slippage, mid-based pricing).
  - Training window should end BEFORE target_date to avoid leakage.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/eval_positive_bucket_policy.py \\
      --league nhl --target-date 2025-12-13 --train-start 2025-12-04 --train-end 2025-12-12
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from chimera_v2c.src.config_loader import V2CConfig
from chimera_v2c.src.doctrine import DoctrineConfig, bucket_for_p, doctrine_decide_trade
from chimera_v2c.src.ledger_analysis import GameRow, load_games


DEFAULT_MODELS = ["v2c", "gemini", "grok", "gpt", "market_proxy", "moneypuck"]


def _parse_iso_date(s: str) -> str:
    return date.fromisoformat(str(s).strip()).isoformat()


def _default_config_for_league(league: str) -> str:
    league = league.lower()
    if league == "nhl":
        return "chimera_v2c/config/nhl_defaults.yaml"
    if league == "nfl":
        return "chimera_v2c/config/nfl_defaults.yaml"
    return "chimera_v2c/config/defaults.yaml"


@dataclass
class BucketAgg:
    n: int = 0
    cost: float = 0.0
    profit: float = 0.0

    def roi(self) -> Optional[float]:
        if self.cost <= 0:
            return None
        return self.profit / self.cost


@dataclass
class SimTotals:
    trades: int = 0
    cost: float = 0.0
    profit: float = 0.0
    skipped_bucket: int = 0
    skipped_no_edge: int = 0
    skipped_missing: int = 0

    def roi(self) -> Optional[float]:
        if self.cost <= 0:
            return None
        return self.profit / self.cost


def _safe_price(p: float) -> float:
    return max(0.01, min(0.99, float(p)))


def _roi_per_cost(price: float, yes_win: bool) -> float:
    price = _safe_price(price)
    if yes_win:
        return (1.0 - price) / price
    return -1.0


def _pnl_per_contract(price: float, yes_win: bool) -> float:
    price = _safe_price(price)
    if yes_win:
        return 1.0 - price
    return -price


def _simple_trade_for_game(
    *,
    p_home_model: float,
    p_home_mid: float,
    home_win: float,
) -> Tuple[str, float, float]:
    """
    Simple trade rule (1 contract):
      - if model>mid: buy HOME yes @ mid
      - else:         buy AWAY yes @ (1-mid)

    Returns (bucket, price, pnl_per_contract).
    """
    p_home_model = max(0.0, min(1.0, float(p_home_model)))
    p_home_mid = _safe_price(p_home_mid)

    if p_home_model == p_home_mid:
        raise ValueError("no edge")

    if p_home_model > p_home_mid:
        p_yes = p_home_model
        price = p_home_mid
        yes_win = home_win == 1.0
    else:
        p_yes = 1.0 - p_home_model
        price = _safe_price(1.0 - p_home_mid)
        yes_win = home_win == 0.0
    return bucket_for_p(p_yes), price, _pnl_per_contract(price, bool(yes_win))


def _choose_trade_for_game(
    *,
    p_home_model: float,
    p_home_mid: float,
    home_win: float,
    cfg: DoctrineConfig,
    used_fraction: float,
    daily_cap: float,
) -> Tuple[Optional[str], Optional[float], Optional[float], float]:
    """
    Return (bucket, stake_fraction, profit_fraction, new_used_fraction) for the chosen trade, or (None,...)
    when no trade is eligible.
    """
    p_home_model = max(0.0, min(1.0, float(p_home_model)))
    p_home_mid = _safe_price(p_home_mid)

    best = None  # (edge, stake, price, yes_win, p_yes_model)

    # Home YES
    stake_h, _, _ = doctrine_decide_trade(
        p_model=p_home_model,
        p_market=p_home_mid,
        cfg=cfg,
        used_fraction=used_fraction,
        daily_cap=daily_cap,
    )
    if stake_h is not None and stake_h > 0:
        edge = p_home_model - p_home_mid
        yes_win = home_win == 1.0
        best = (edge, stake_h, p_home_mid, yes_win, p_home_model)

    # Away YES (approximate price as 1 - home_mid)
    p_away_model = 1.0 - p_home_model
    p_away_mid = _safe_price(1.0 - p_home_mid)
    stake_a, _, _ = doctrine_decide_trade(
        p_model=p_away_model,
        p_market=p_away_mid,
        cfg=cfg,
        used_fraction=used_fraction,
        daily_cap=daily_cap,
    )
    if stake_a is not None and stake_a > 0:
        edge = p_away_model - p_away_mid
        yes_win = home_win == 0.0
        cand = (edge, stake_a, p_away_mid, yes_win, p_away_model)
        if best is None or cand[0] > best[0]:
            best = cand

    if best is None:
        return None, None, None, used_fraction

    edge, stake, price, yes_win, p_yes_model = best
    bucket = bucket_for_p(p_yes_model)
    profit = stake * _roi_per_cost(price, bool(yes_win))
    return bucket, stake, profit, used_fraction + stake


def _group_by_date(games: Sequence[GameRow]) -> Dict[str, List[GameRow]]:
    by: Dict[str, List[GameRow]] = defaultdict(list)
    for g in games:
        by[g.date.strftime("%Y-%m-%d")].append(g)
    for d in by:
        by[d].sort(key=lambda x: (x.matchup or ""))
    return dict(by)


def build_positive_bucket_rulebook(
    *,
    games: Sequence[GameRow],
    model: str,
    cfg: DoctrineConfig,
    daily_cap: float,
    min_bets: int,
) -> Tuple[Dict[str, BucketAgg], List[str]]:
    """
    Returns (bucket_stats, positive_buckets) for a model.
    """
    stats: Dict[str, BucketAgg] = defaultdict(BucketAgg)
    by_date = _group_by_date(games)
    for _, day_games in sorted(by_date.items()):
        used = 0.0
        for g in day_games:
            if g.home_win is None or g.home_win == 0.5:
                continue
            if g.kalshi_mid is None:
                continue
            p_home_model = g.probs.get(model)
            if p_home_model is None:
                continue
            bucket, stake, profit, used = _choose_trade_for_game(
                p_home_model=float(p_home_model),
                p_home_mid=float(g.kalshi_mid),
                home_win=float(g.home_win),
                cfg=cfg,
                used_fraction=used,
                daily_cap=daily_cap,
            )
            if bucket is None or stake is None or profit is None:
                continue
            s = stats[bucket]
            s.n += 1
            s.cost += float(stake)
            s.profit += float(profit)

    positive = []
    for bucket, s in stats.items():
        roi = s.roi()
        if roi is None:
            continue
        if s.n >= min_bets and roi > 0:
            positive.append(bucket)
    positive.sort()
    return dict(stats), positive


def build_positive_bucket_rulebook_simple(
    *,
    games: Sequence[GameRow],
    model: str,
    min_bets: int,
) -> Tuple[Dict[str, BucketAgg], List[str]]:
    stats: Dict[str, BucketAgg] = defaultdict(BucketAgg)
    for g in games:
        if g.home_win is None or g.home_win == 0.5:
            continue
        if g.kalshi_mid is None:
            continue
        p_home_model = g.probs.get(model)
        if p_home_model is None:
            continue
        try:
            bucket, price, pnl = _simple_trade_for_game(
                p_home_model=float(p_home_model),
                p_home_mid=float(g.kalshi_mid),
                home_win=float(g.home_win),
            )
        except ValueError:
            continue
        s = stats[bucket]
        s.n += 1
        s.cost += float(price)
        s.profit += float(pnl)

    positive = []
    for bucket, s in stats.items():
        if s.n < min_bets:
            continue
        roi = s.roi()
        if roi is None:
            continue
        if roi > 0:
            positive.append(bucket)
    positive.sort()
    return dict(stats), positive


def simulate_target_date(
    *,
    games: Sequence[GameRow],
    model: str,
    cfg: DoctrineConfig,
    daily_cap: float,
    allowed_buckets: Sequence[str],
) -> SimTotals:
    allowed = set(allowed_buckets)
    totals = SimTotals()

    used = 0.0
    for g in sorted(games, key=lambda x: (x.matchup or "")):
        if g.home_win is None or g.home_win == 0.5:
            totals.skipped_missing += 1
            continue
        if g.kalshi_mid is None:
            totals.skipped_missing += 1
            continue
        p_home_model = g.probs.get(model)
        if p_home_model is None:
            totals.skipped_missing += 1
            continue

        bucket, stake, profit, used_next = _choose_trade_for_game(
            p_home_model=float(p_home_model),
            p_home_mid=float(g.kalshi_mid),
            home_win=float(g.home_win),
            cfg=cfg,
            used_fraction=used,
            daily_cap=daily_cap,
        )
        if bucket is None or stake is None or profit is None:
            totals.skipped_no_edge += 1
            continue
        if bucket not in allowed:
            totals.skipped_bucket += 1
            continue

        used = used_next
        totals.trades += 1
        totals.cost += float(stake)
        totals.profit += float(profit)

    return totals


def simulate_target_date_simple(
    *,
    games: Sequence[GameRow],
    model: str,
    allowed_buckets: Sequence[str],
) -> SimTotals:
    allowed = set(allowed_buckets)
    totals = SimTotals()
    for g in sorted(games, key=lambda x: (x.matchup or "")):
        if g.home_win is None or g.home_win == 0.5:
            totals.skipped_missing += 1
            continue
        if g.kalshi_mid is None:
            totals.skipped_missing += 1
            continue
        p_home_model = g.probs.get(model)
        if p_home_model is None:
            totals.skipped_missing += 1
            continue
        try:
            bucket, price, pnl = _simple_trade_for_game(
                p_home_model=float(p_home_model),
                p_home_mid=float(g.kalshi_mid),
                home_win=float(g.home_win),
            )
        except ValueError:
            totals.skipped_no_edge += 1
            continue
        if bucket not in allowed:
            totals.skipped_bucket += 1
            continue
        totals.trades += 1
        totals.cost += float(price)
        totals.profit += float(pnl)
    return totals


def _format_float(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return ""
    if math.isnan(x):
        return ""
    return f"{x:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate per-model positive-bucket trading policy vs Kalshi mid.")
    ap.add_argument("--league", required=True, help="nba|nhl|nfl")
    ap.add_argument("--target-date", required=True, help="YYYY-MM-DD (must be graded in daily ledgers).")
    ap.add_argument(
        "--train-start",
        help="YYYY-MM-DD (inclusive). Default: 9 days before target-date.",
    )
    ap.add_argument(
        "--train-end",
        help="YYYY-MM-DD (inclusive). Default: day before target-date.",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model columns to evaluate (default: v2c gemini grok gpt market_proxy moneypuck).",
    )
    ap.add_argument(
        "--mode",
        choices=["simple", "doctrine"],
        default="simple",
        help="Simulation mode (default: simple).",
    )
    ap.add_argument("--bucket-min-bets", type=int, default=10, help="Min training trades per bucket (default: 10).")
    ap.add_argument("--config", help="League config YAML (default based on --league).")
    args = ap.parse_args()

    league = args.league.strip().lower()
    target_date = _parse_iso_date(args.target_date)
    target_dt = date.fromisoformat(target_date)
    train_end = _parse_iso_date(args.train_end) if args.train_end else (target_dt - timedelta(days=1)).isoformat()
    train_start = _parse_iso_date(args.train_start) if args.train_start else (target_dt - timedelta(days=9)).isoformat()
    if train_end >= target_date:
        raise SystemExit("[error] training window must end before target-date to avoid leakage")

    cfg_path = args.config or _default_config_for_league(league)
    v2c_cfg = V2CConfig.load(cfg_path)

    # Use doctrine sizing rules but disable bucket guardrails while building rulebooks.
    doc_cfg = DoctrineConfig(
        max_fraction=v2c_cfg.max_fraction,
        fee_buffer=0.02,
        kelly_fraction=0.25,
        target_spread_bp=v2c_cfg.target_spread_bp,
        require_confluence=False,
        enable_bucket_guardrails=False,
        require_positive_roi_buckets=False,
        bucket_guardrails_path="",
        league_min_samples=None,
        paper_mode_enforce=True,
        league=league,
        negative_roi_buckets=v2c_cfg.doctrine_cfg.get("negative_roi_buckets"),
    )
    daily_cap = float(getattr(v2c_cfg, "daily_max_fraction", v2c_cfg.max_fraction * 8))

    models = [m.strip() for m in args.models if m.strip()]
    games_train = load_games(start_date=train_start, end_date=train_end, league_filter=league, models=models)
    games_target = load_games(start_date=target_date, end_date=target_date, league_filter=league, models=models)
    if not games_target:
        raise SystemExit(f"[error] no ledger rows found for {league} {target_date}")

    # Build rulebooks + simulate target.
    results = []
    for model in models:
        if args.mode == "doctrine":
            stats, positive = build_positive_bucket_rulebook(
                games=games_train,
                model=model,
                cfg=doc_cfg,
                daily_cap=daily_cap,
                min_bets=int(args.bucket_min_bets),
            )
            totals = simulate_target_date(
                games=games_target,
                model=model,
                cfg=doc_cfg,
                daily_cap=daily_cap,
                allowed_buckets=positive,
            )
        else:
            stats, positive = build_positive_bucket_rulebook_simple(
                games=games_train,
                model=model,
                min_bets=int(args.bucket_min_bets),
            )
            totals = simulate_target_date_simple(
                games=games_target,
                model=model,
                allowed_buckets=positive,
            )
        results.append(
            {
                "model": model,
                "train_positive_buckets": len(positive),
                "train_buckets_with_stats": len(stats),
                "target_trades": totals.trades,
                "target_cost_frac": totals.cost,
                "target_profit_frac": totals.profit,
                "target_roi": totals.roi(),
                "skipped_bucket": totals.skipped_bucket,
            }
        )

    # Print sorted by profit.
    results.sort(key=lambda r: (r["target_profit_frac"] or 0.0), reverse=True)
    print(
        f"\nPositive-bucket policy eval — mode={args.mode} league={league} target_date={target_date} "
        f"train={train_start}:{train_end} min_bets={args.bucket_min_bets}"
    )
    print("model            pos_buckets  target_trades  spend      profit     roi       skipped_bucket")
    for r in results:
        print(
            f"{r['model']:<15} {r['train_positive_buckets']:<11} {r['target_trades']:<13} "
            f"{_format_float(r.get('target_cost_frac'), 4):<10} {_format_float(r['target_profit_frac'], 4):<10} "
            f"{_format_float(r['target_roi'], 4):<8} {r['skipped_bucket']}"
        )


if __name__ == "__main__":
    main()
