#!/usr/bin/env python
"""
Paper trade sheet: Rule-A I/J tiered policy (fee-aware; read-only).

This generates per-game recommendations for a target date using ONLY
information from dates < target date (walk-forward, no leakage).

Policy recap (home-win probabilities; trade at Kalshi mid):
  - Rule A (home favorite + fade home): p_mid >= 0.5 and p_model <= p_mid - t
    -> buy AWAY YES (price = 1 - p_mid)

Sub-buckets:
  - I: Rule A and p_model < 0.5  (sign flip to away-fav)
  - J: Rule A and p_model >= 0.5 (no flip)

Tiering / gating:
  - If primary triggers I: trade away with units = --units-i
  - Else if primary triggers J AND confirmer triggers J: trade away with units = --units-jj
  - Else: skip

Fees:
  - Optional Kalshi maker/taker fees (Oct 2025 schedule) via chimera_v2c.src.kalshi_fees.

Output:
  - Writes a CSV under reports/trade_sheets/ by default.
  - Never writes to daily ledgers.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chimera_v2c.src.kalshi_fees import maker_fee_dollars, taker_fee_dollars
from chimera_v2c.src.ledger_analysis import GameRow, load_games


DEFAULT_OUT_DIR = Path("reports/trade_sheets")


@dataclass(frozen=True)
class Sample:
    date: str
    league: str
    matchup: str
    p_mid: float
    y: Optional[int]
    probs: Dict[str, float]


def _parse_iso_date(s: str) -> str:
    return date.fromisoformat(str(s).strip()).isoformat()


def _normalize_league(value: str) -> str:
    v = (value or "all").strip().lower()
    if v in {"all", "nba", "nhl", "nfl"}:
        return v
    raise SystemExit("[error] --league must be one of: nba, nhl, nfl, all")


def _fee_dollars(*, mode: str, contracts: int, price: float) -> float:
    m = (mode or "none").strip().lower()
    if m == "none":
        return 0.0
    if m == "taker":
        return taker_fee_dollars(contracts=contracts, price=price)
    return maker_fee_dollars(contracts=contracts, price=price)


def _iter_samples(games: Iterable[GameRow], models: Sequence[str]) -> List[Sample]:
    out: List[Sample] = []
    for g in games:
        if g.kalshi_mid is None:
            continue
        probs: Dict[str, float] = {}
        for m in models:
            pm = g.probs.get(m)
            if pm is None:
                continue
            probs[str(m)] = float(pm)
        y: Optional[int]
        if g.home_win in (0.0, 1.0):
            y = int(g.home_win)
        else:
            y = None
        out.append(
            Sample(
                date=g.date.strftime("%Y-%m-%d"),
                league=g.league,
                matchup=g.matchup,
                p_mid=float(g.kalshi_mid),
                y=y,
                probs=probs,
            )
        )
    return out


def _train_window(samples_all: List[Sample], *, target_date: str, train_days: int) -> List[Sample]:
    before = [s for s in samples_all if s.date < target_date]
    if train_days <= 0:
        return before
    dates = sorted({s.date for s in before})
    keep = set(dates[-train_days:])
    return [s for s in before if s.date in keep]


def _rule_a_subbucket(*, p_mid: float, p_model: float, t: float) -> Optional[str]:
    pmid = float(p_mid)
    if pmid < 0.5:
        return None
    eps = 1e-12
    if float(p_model) > (pmid - float(t) + eps):
        return None
    return "I" if float(p_model) < 0.5 else "J"


def _trade_for_sample(
    s: Sample,
    *,
    primary: str,
    confirmer: str,
    t: float,
    units_i: int,
    units_jj: int,
) -> Tuple[int, Optional[str], Optional[str]]:
    if float(s.p_mid) < 0.5:
        return 0, None, None
    p_primary = s.probs.get(primary)
    p_conf = s.probs.get(confirmer)
    if p_primary is None or p_conf is None:
        return 0, None, None

    sub_p = _rule_a_subbucket(p_mid=s.p_mid, p_model=float(p_primary), t=t)
    if sub_p == "I":
        return int(units_i), "I", None
    if sub_p == "J":
        sub_c = _rule_a_subbucket(p_mid=s.p_mid, p_model=float(p_conf), t=t)
        if sub_c == "J":
            return int(units_jj), "J", "J"
        return 0, "J", sub_c
    return 0, None, None


def _gross_pnl_buy_away(*, p_mid: float, home_win: int, contracts: int) -> float:
    # Realized PnL per contract when buying AWAY YES at price (1 - p_mid):
    # away win (home_win=0) => +p_mid; home win => -(1 - p_mid)
    pmid = float(p_mid)
    per = pmid if int(home_win) == 0 else -(1.0 - pmid)
    return float(contracts) * float(per)


def _train_totals_for_t(
    train: Sequence[Sample],
    *,
    primary: str,
    confirmer: str,
    t: float,
    units_i: int,
    units_jj: int,
    fee_mode: str,
) -> Tuple[int, int, float]:
    """
    Returns (bets, contracts, net_pnl) on TRAIN only.
    """
    bets = 0
    contracts_total = 0
    net = 0.0
    for s in train:
        if s.y is None:
            continue
        contracts, _, _ = _trade_for_sample(
            s,
            primary=primary,
            confirmer=confirmer,
            t=float(t),
            units_i=int(units_i),
            units_jj=int(units_jj),
        )
        if contracts <= 0:
            continue
        bets += 1
        contracts_total += int(contracts)
        gross = _gross_pnl_buy_away(p_mid=s.p_mid, home_win=int(s.y), contracts=int(contracts))
        price = 1.0 - float(s.p_mid)
        fees = _fee_dollars(mode=fee_mode, contracts=int(contracts), price=float(price))
        net += float(gross - fees)
    return bets, contracts_total, float(net)


def _select_threshold(
    train: Sequence[Sample],
    *,
    primary: str,
    confirmer: str,
    thresholds: Sequence[float],
    select_mode: str,
    units_i: int,
    units_jj: int,
    fee_mode: str,
) -> float:
    best_t = float(thresholds[0])
    best_score: Tuple[float, float] = (-1e18, -1e18)  # (net_pnl, avg_net)
    for t in thresholds:
        bets, contracts, net = _train_totals_for_t(
            train,
            primary=primary,
            confirmer=confirmer,
            t=float(t),
            units_i=int(units_i),
            units_jj=int(units_jj),
            fee_mode=fee_mode,
        )
        avg = float(net / contracts) if contracts else -1e18
        score = (float(net), avg)
        if select_mode == "max_avg_net":
            score = (avg, float(net))
        if score > best_score:
            best_score = score
            best_t = float(t)
    return float(best_t)


def plan_trades(
    games: Sequence[GameRow],
    *,
    target_date: str,
    league_for_stats: str,
    primary: str,
    confirmer: str,
    edge_threshold: float,
    edge_thresholds: Optional[Sequence[float]],
    threshold_select_mode: str,
    units_i: int,
    units_jj: int,
    fee_mode: str,
    train_days: int,
    min_train_days: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    target = _parse_iso_date(target_date)
    league_norm = _normalize_league(league_for_stats)
    primary_key = str(primary).strip()
    confirmer_key = str(confirmer).strip()

    models = sorted({primary_key, confirmer_key})
    samples_all = _iter_samples(games, models=models)
    if league_norm != "all":
        samples_all = [s for s in samples_all if s.league == league_norm]

    train = _train_window(samples_all, target_date=target, train_days=int(train_days))
    train_days_distinct = sorted({s.date for s in train if s.y is not None})
    trained = len(train_days_distinct) >= int(min_train_days)

    thresholds = list(edge_thresholds) if edge_thresholds else [float(edge_threshold)]
    thresholds = [float(t) for t in thresholds if float(t) > 0]
    if not thresholds:
        thresholds = [float(edge_threshold)]

    selected_t = float(thresholds[0])
    if trained and len(thresholds) > 1:
        selected_t = _select_threshold(
            train,
            primary=primary_key,
            confirmer=confirmer_key,
            thresholds=thresholds,
            select_mode=str(threshold_select_mode),
            units_i=int(units_i),
            units_jj=int(units_jj),
            fee_mode=str(fee_mode),
        )

    rows: List[Dict[str, object]] = []
    test = [s for s in samples_all if s.date == target]
    for s in test:
        p_primary = s.probs.get(primary_key)
        p_conf = s.probs.get(confirmer_key)
        if p_primary is None or p_conf is None:
            continue

        contracts, sub_p, sub_c = _trade_for_sample(
            s,
            primary=primary_key,
            confirmer=confirmer_key,
            t=float(selected_t),
            units_i=int(units_i),
            units_jj=int(units_jj),
        )
        if contracts <= 0:
            continue

        price_away = 1.0 - float(s.p_mid)
        fee = _fee_dollars(mode=str(fee_mode), contracts=int(contracts), price=float(price_away))
        edge_per = float(s.p_mid - float(p_primary))  # EV/contract for buying away YES at (1-p_mid)
        exp_gross = float(contracts) * edge_per
        exp_net = float(exp_gross - fee)

        rows.append(
            {
                "date": s.date,
                "league": s.league,
                "matchup": s.matchup,
                "side": "away",
                "policy_bucket": "A",
                "policy_subbucket_primary": sub_p or "",
                "policy_subbucket_confirmer": sub_c or "",
                "t_selected": round(float(selected_t), 6),
                "p_mid_home": float(s.p_mid),
                "price_away_yes": float(price_away),
                "p_primary_home": float(p_primary),
                "p_confirmer_home": float(p_conf),
                "contracts": int(contracts),
                "exp_edge_per_contract": float(edge_per),
                "exp_gross_pnl": float(exp_gross),
                "fee_estimate": float(fee),
                "exp_net_pnl": float(exp_net),
            }
        )

    meta: Dict[str, object] = {
        "target_date": target,
        "league": league_norm,
        "primary": primary_key,
        "confirmer": confirmer_key,
        "trained": bool(trained),
        "train_days": int(train_days),
        "min_train_days": int(min_train_days),
        "train_days_distinct": int(len(train_days_distinct)),
        "threshold_select_mode": str(threshold_select_mode),
        "threshold_candidates": [float(t) for t in thresholds],
        "t_selected": float(selected_t),
        "units_i": int(units_i),
        "units_jj": int(units_jj),
        "fee_mode": str(fee_mode),
        "trades": int(len(rows)),
        "contracts": int(sum(int(r["contracts"]) for r in rows)),
        "exp_net_total": float(sum(float(r["exp_net_pnl"]) for r in rows)),
    }
    return rows, meta


def _default_out_path(meta: Dict[str, object]) -> Path:
    date_token = str(meta["target_date"]).replace("-", "")
    league = str(meta["league"])
    primary = str(meta["primary"])
    confirmer = str(meta["confirmer"])
    fee_mode = str(meta["fee_mode"])
    t = float(meta["t_selected"])
    u_i = int(meta["units_i"])
    u_jj = int(meta["units_jj"])
    name = f"{date_token}_rule_a_ij_tiered_{league}_{primary}-{confirmer}_fee_{fee_mode}_t{t:.3f}_uI{u_i}_uJJ{u_jj}.csv"
    return DEFAULT_OUT_DIR / name


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan Rule-A I/J tiered trades for a target date (read-only).")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD.")
    ap.add_argument("--league", default="all", help="nba|nhl|nfl|all (default: all).")
    ap.add_argument("--primary", default="grok", help="Primary model column (default: grok).")
    ap.add_argument("--confirmer", default="market_proxy", help="Confirmer model column (default: market_proxy).")
    ap.add_argument("--train-days", type=int, default=0, help="Rolling train window in ledger days (0 = expanding).")
    ap.add_argument("--min-train-days", type=int, default=5, help="Minimum distinct train days required (default: 5).")
    ap.add_argument("--edge-threshold", type=float, default=0.015, help="Fallback threshold t if no candidates provided.")
    ap.add_argument("--edge-thresholds", nargs="+", type=float, help="Candidate thresholds for train-only selection.")
    ap.add_argument(
        "--threshold-select-mode",
        choices=["max_net_pnl", "max_avg_net"],
        default="max_net_pnl",
        help="How to pick t among candidates (default: max_net_pnl).",
    )
    ap.add_argument("--units-i", type=int, default=3, help="Contracts for sub-bucket I (default: 3).")
    ap.add_argument("--units-jj", type=int, default=1, help="Contracts for sub-bucket J with confirmer (default: 1).")
    ap.add_argument("--fee-mode", choices=["none", "maker", "taker"], default="maker", help="Fee model (default: maker).")
    ap.add_argument("--out", default="", help="Optional output CSV path (default under reports/trade_sheets/).")
    ap.add_argument("--no-write", action="store_true", help="Do not write CSV; print summary only.")
    args = ap.parse_args()

    target = _parse_iso_date(args.date)
    league = _normalize_league(args.league)

    games = load_games(
        days=None,
        start_date=None,
        end_date=target,
        league_filter=None if league == "all" else league,
        models=list(sorted({str(args.primary).strip(), str(args.confirmer).strip()})),
    )

    rows, meta = plan_trades(
        games,
        target_date=target,
        league_for_stats=league,
        primary=str(args.primary),
        confirmer=str(args.confirmer),
        edge_threshold=float(args.edge_threshold),
        edge_thresholds=args.edge_thresholds,
        threshold_select_mode=str(args.threshold_select_mode),
        units_i=int(args.units_i),
        units_jj=int(args.units_jj),
        fee_mode=str(args.fee_mode),
        train_days=int(args.train_days),
        min_train_days=int(args.min_train_days),
    )

    print(
        f"\n=== Rule-A I/J tiered trade sheet ===\n"
        f"date={meta['target_date']} league={meta['league']} primary={meta['primary']} confirmer={meta['confirmer']} fee={meta['fee_mode']}\n"
        f"trained={meta['trained']} train_days_distinct={meta['train_days_distinct']} t_selected={meta['t_selected']:.6f}\n"
        f"trades={meta['trades']} contracts={meta['contracts']} exp_net_total={meta['exp_net_total']:.3f}\n"
    )

    if args.no_write:
        return

    out_path = Path(args.out) if args.out else _default_out_path(meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "league",
        "matchup",
        "side",
        "policy_bucket",
        "policy_subbucket_primary",
        "policy_subbucket_confirmer",
        "t_selected",
        "p_mid_home",
        "price_away_yes",
        "p_primary_home",
        "p_confirmer_home",
        "contracts",
        "exp_edge_per_contract",
        "exp_gross_pnl",
        "fee_estimate",
        "exp_net_pnl",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()

