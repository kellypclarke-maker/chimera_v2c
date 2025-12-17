#!/usr/bin/env python
"""
Walk-forward (train->test) Rule-A I/J tiered policy (maker-only; fee-aware).

Rule A is the "home-rich fade" regime:
  - Market favors home: p_mid >= 0.5
  - Model says home is overpriced: p_model <= p_mid - t
  - Trade: buy AWAY YES

Sub-buckets:
  - I: Rule A and p_model < 0.5 (sign flip to away-fav)
  - J: Rule A and p_model >= 0.5 (no flip)

Tiered policy (default):
  - If primary model triggers I: trade away with units = --units-i
  - Else if primary triggers J AND confirmer triggers J: trade away with units = --units-jj
  - Else: skip

We compare to a baseline:
  - Always buy away when p_mid >= 0.5 (1 contract per game)

Fees:
  - Maker fee schedule (Oct 2025): fee = round_up_to_cent(0.0175 * C * P * (1-P))
  - Taker fee schedule (Oct 2025): fee = round_up_to_cent(0.07   * C * P * (1-P))
    where:
      P = contract price in dollars
      C = contracts executed
  - For AWAY YES, the contract price is P = (1 - p_mid)

This tool is read-only on daily ledgers; it writes derived CSVs under
reports/thesis_summaries/.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chimera_v2c.src.kalshi_fees import FeeBreakdown, maker_fee_dollars, taker_fee_dollars
from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.rulebook_quadrants import pnl_buy_away


THESIS_DIR = Path("reports/thesis_summaries")


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
    contracts: int = 0
    gross_pnl: float = 0.0
    fees: float = 0.0

    @property
    def net_pnl(self) -> float:
        return float(self.gross_pnl - self.fees)

    @property
    def avg_net_per_contract(self) -> float:
        return self.net_pnl / self.contracts if self.contracts else 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Walk-forward Rule-A I/J tiered policy (fee-aware; read-only).")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive (test window).")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive (test window).")
    ap.add_argument("--league", default="all", help="nba|nhl|nfl|all (default: all).")
    ap.add_argument("--primary", default="grok", help="Primary model column for I/J (default: grok).")
    ap.add_argument("--confirmer", default="market_proxy", help="Confirmer model column for J (default: market_proxy).")
    ap.add_argument("--train-days", type=int, default=0, help="Rolling train window in ledger days (0 = expanding).")
    ap.add_argument("--min-train-days", type=int, default=5, help="Minimum distinct train days required (default: 5).")
    ap.add_argument(
        "--edge-threshold",
        type=float,
        default=0.015,
        help="Fixed edge threshold t for Rule A/I/J (default: 0.015).",
    )
    ap.add_argument(
        "--edge-thresholds",
        nargs="+",
        type=float,
        help="Optional threshold candidates; if provided, pick the best t on train-only data.",
    )
    ap.add_argument(
        "--threshold-select-mode",
        choices=["max_net_pnl", "max_avg_net"],
        default="max_net_pnl",
        help="How to pick t among candidates using train data (default: max_net_pnl).",
    )
    ap.add_argument("--units-i", type=int, default=2, help="Contracts to trade when primary is I (default: 2).")
    ap.add_argument(
        "--units-jj",
        type=int,
        default=1,
        help="Contracts to trade when primary is J and confirmer is J (default: 1).",
    )
    ap.add_argument(
        "--fee-mode",
        choices=["none", "maker", "taker"],
        default="maker",
        help="Fee model to apply (default: maker).",
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


def _rule_a_subbucket(*, p_mid: float, p_model: float, t: float) -> Optional[str]:
    pmid = float(p_mid)
    if pmid < 0.5:
        return None
    # Match rulebook_quadrants' drift protections.
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
) -> Tuple[int, Optional[str]]:
    if float(s.p_mid) < 0.5:
        return 0, None
    p_primary = s.probs.get(primary)
    p_conf = s.probs.get(confirmer)
    if p_primary is None or p_conf is None:
        return 0, None

    sub_p = _rule_a_subbucket(p_mid=s.p_mid, p_model=float(p_primary), t=t)
    if sub_p == "I":
        return int(units_i), "I"
    if sub_p == "J":
        sub_c = _rule_a_subbucket(p_mid=s.p_mid, p_model=float(p_conf), t=t)
        if sub_c == "J":
            return int(units_jj), "J"
    return 0, None


def _pnl_and_fee_for_trade(*, p_mid: float, y: int, contracts: int, fee_mode: str) -> FeeBreakdown:
    pnl_per = float(pnl_buy_away(p_mid=float(p_mid), home_win=float(y)))
    gross = float(contracts) * pnl_per
    price = 1.0 - float(p_mid)  # AWAY YES price
    fees = _fee_dollars(mode=fee_mode, contracts=int(contracts), price=float(price))
    return FeeBreakdown(gross_pnl=gross, fees=fees)


def _baseline_fade_homefav(test: Sequence[Sample], *, fee_mode: str) -> Totals:
    out = Totals()
    for s in test:
        if float(s.p_mid) < 0.5:
            continue
        out.bets += 1
        out.contracts += 1
        fb = _pnl_and_fee_for_trade(p_mid=s.p_mid, y=s.y, contracts=1, fee_mode=fee_mode)
        out.gross_pnl += fb.gross_pnl
        out.fees += fb.fees
    return out


def _train_score_for_t(
    train: Sequence[Sample],
    *,
    primary: str,
    confirmer: str,
    t: float,
    units_i: int,
    units_jj: int,
    fee_mode: str,
) -> Totals:
    tot = Totals()
    for s in train:
        contracts, _ = _trade_for_sample(
            s,
            primary=primary,
            confirmer=confirmer,
            t=float(t),
            units_i=int(units_i),
            units_jj=int(units_jj),
        )
        if contracts <= 0:
            continue
        tot.bets += 1
        tot.contracts += int(contracts)
        fb = _pnl_and_fee_for_trade(p_mid=s.p_mid, y=s.y, contracts=int(contracts), fee_mode=fee_mode)
        tot.gross_pnl += fb.gross_pnl
        tot.fees += fb.fees
    return tot


def _pick_threshold(
    train: Sequence[Sample],
    *,
    primary: str,
    confirmer: str,
    candidates: Sequence[float],
    units_i: int,
    units_jj: int,
    fee_mode: str,
    mode: str,
) -> float:
    best_t = float(candidates[0])
    best_score = None
    mode_norm = (mode or "").strip().lower()
    for t in candidates:
        stats = _train_score_for_t(
            train,
            primary=primary,
            confirmer=confirmer,
            t=float(t),
            units_i=int(units_i),
            units_jj=int(units_jj),
            fee_mode=fee_mode,
        )
        if stats.contracts <= 0:
            score = float("-inf")
        elif mode_norm == "max_avg_net":
            score = stats.avg_net_per_contract
        else:
            score = stats.net_pnl
        if best_score is None or score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and float(t) < best_t):
            best_score = score
            best_t = float(t)
    return float(best_t)


def _walkforward(
    samples_all: Sequence[Sample],
    *,
    league_filter: Optional[str],
    primary: str,
    confirmer: str,
    train_days: int,
    min_train_days: int,
    candidates: Sequence[float],
    threshold_select_mode: str,
    units_i: int,
    units_jj: int,
    fee_mode: str,
) -> Tuple[List[Dict[str, object]], Totals, Totals]:
    dates = sorted({s.date for s in samples_all})
    rows: List[Dict[str, object]] = []
    tot_policy = Totals()
    tot_baseline = Totals()

    for d in dates:
        train_all = _train_window(list(samples_all), test_date=d, train_days=int(train_days))
        test_all = [s for s in samples_all if s.date == d]
        if league_filter:
            train = [s for s in train_all if s.league == league_filter]
            test = [s for s in test_all if s.league == league_filter]
        else:
            train = list(train_all)
            test = list(test_all)

        train_days_list = sorted({s.date for s in train})
        if len(train_days_list) < int(min_train_days):
            continue
        if not test:
            continue

        t = _pick_threshold(
            train,
            primary=primary,
            confirmer=confirmer,
            candidates=candidates,
            units_i=int(units_i),
            units_jj=int(units_jj),
            fee_mode=fee_mode,
            mode=str(threshold_select_mode),
        )

        day_policy = Totals()
        day_i = Totals()
        day_j = Totals()
        for s in test:
            contracts, sub = _trade_for_sample(
                s,
                primary=primary,
                confirmer=confirmer,
                t=float(t),
                units_i=int(units_i),
                units_jj=int(units_jj),
            )
            if contracts <= 0:
                continue
            day_policy.bets += 1
            day_policy.contracts += int(contracts)
            fb = _pnl_and_fee_for_trade(p_mid=s.p_mid, y=s.y, contracts=int(contracts), fee_mode=fee_mode)
            day_policy.gross_pnl += fb.gross_pnl
            day_policy.fees += fb.fees
            if sub == "I":
                day_i.bets += 1
                day_i.contracts += int(contracts)
                day_i.gross_pnl += fb.gross_pnl
                day_i.fees += fb.fees
            elif sub == "J":
                day_j.bets += 1
                day_j.contracts += int(contracts)
                day_j.gross_pnl += fb.gross_pnl
                day_j.fees += fb.fees

        day_base = _baseline_fade_homefav(test, fee_mode=fee_mode)

        tot_policy.bets += day_policy.bets
        tot_policy.contracts += day_policy.contracts
        tot_policy.gross_pnl += day_policy.gross_pnl
        tot_policy.fees += day_policy.fees

        tot_baseline.bets += day_base.bets
        tot_baseline.contracts += day_base.contracts
        tot_baseline.gross_pnl += day_base.gross_pnl
        tot_baseline.fees += day_base.fees

        rows.append(
            {
                "test_date": d,
                "league": league_filter or "overall",
                "train_start": train_days_list[0],
                "train_end": train_days_list[-1],
                "train_days": len(train_days_list),
                "t_selected": f"{float(t):.3f}",
                "fee_mode": fee_mode,
                "units_I": int(units_i),
                "units_JJ": int(units_jj),
                "policy_bets": day_policy.bets,
                "policy_contracts": day_policy.contracts,
                "policy_gross_pnl": f"{day_policy.gross_pnl:.6f}",
                "policy_fees": f"{day_policy.fees:.6f}",
                "policy_net_pnl": f"{day_policy.net_pnl:.6f}",
                "policy_avg_net_per_contract": f"{day_policy.avg_net_per_contract:.6f}" if day_policy.contracts else "",
                "policy_I_bets": day_i.bets,
                "policy_I_contracts": day_i.contracts,
                "policy_I_net_pnl": f"{day_i.net_pnl:.6f}",
                "policy_J_bets": day_j.bets,
                "policy_J_contracts": day_j.contracts,
                "policy_J_net_pnl": f"{day_j.net_pnl:.6f}",
                "baseline_bets": day_base.bets,
                "baseline_contracts": day_base.contracts,
                "baseline_gross_pnl": f"{day_base.gross_pnl:.6f}",
                "baseline_fees": f"{day_base.fees:.6f}",
                "baseline_net_pnl": f"{day_base.net_pnl:.6f}",
                "baseline_avg_net_per_contract": f"{(day_base.net_pnl / day_base.contracts):.6f}" if day_base.contracts else "",
            }
        )

    if not rows:
        raise SystemExit("[error] no walk-forward rows produced (check min_train_days/window).")

    rows.append(
        {
            "test_date": "OVERALL",
            "league": league_filter or "overall",
            "train_start": "",
            "train_end": "",
            "train_days": "",
            "t_selected": "",
            "fee_mode": fee_mode,
            "units_I": int(units_i),
            "units_JJ": int(units_jj),
            "policy_bets": tot_policy.bets,
            "policy_contracts": tot_policy.contracts,
            "policy_gross_pnl": f"{tot_policy.gross_pnl:.6f}",
            "policy_fees": f"{tot_policy.fees:.6f}",
            "policy_net_pnl": f"{tot_policy.net_pnl:.6f}",
            "policy_avg_net_per_contract": f"{tot_policy.avg_net_per_contract:.6f}" if tot_policy.contracts else "",
            "policy_I_bets": "",
            "policy_I_contracts": "",
            "policy_I_net_pnl": "",
            "policy_J_bets": "",
            "policy_J_contracts": "",
            "policy_J_net_pnl": "",
            "baseline_bets": tot_baseline.bets,
            "baseline_contracts": tot_baseline.contracts,
            "baseline_gross_pnl": f"{tot_baseline.gross_pnl:.6f}",
            "baseline_fees": f"{tot_baseline.fees:.6f}",
            "baseline_net_pnl": f"{tot_baseline.net_pnl:.6f}",
            "baseline_avg_net_per_contract": f"{(tot_baseline.net_pnl / tot_baseline.contracts):.6f}"
            if tot_baseline.contracts
            else "",
        }
    )

    return rows, tot_policy, tot_baseline


def main() -> None:
    args = parse_args()
    if not LEDGER_DIR.exists():
        raise SystemExit(f"[error] daily ledger directory missing: {LEDGER_DIR}")

    league_filter = _normalize_league_arg(args.league)
    primary = args.primary.strip().lower()
    confirmer = args.confirmer.strip().lower()
    models = sorted({primary, confirmer})

    candidates = list(args.edge_thresholds) if args.edge_thresholds else [float(args.edge_threshold)]
    if any(float(t) <= 0 for t in candidates):
        raise SystemExit("[error] edge thresholds must be > 0")

    if int(args.units_i) <= 0 or int(args.units_jj) < 0:
        raise SystemExit("[error] --units-i must be >0 and --units-jj must be >=0")

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

    rows, tot_policy, tot_base = _walkforward(
        samples_all=samples_all,
        league_filter=league_filter,
        primary=primary,
        confirmer=confirmer,
        train_days=int(args.train_days),
        min_train_days=int(args.min_train_days),
        candidates=candidates,
        threshold_select_mode=str(args.threshold_select_mode),
        units_i=int(args.units_i),
        units_jj=int(args.units_jj),
        fee_mode=str(args.fee_mode),
    )

    print("\n=== Walk-forward Rule-A I/J tiered policy ===")
    print(
        f"league={args.league} window={args.start_date}..{args.end_date} rows={len(rows)-1} "
        f"primary={primary} confirmer={confirmer} fee_mode={args.fee_mode}"
    )
    print(
        f"policy:   bets={tot_policy.bets:4d} contracts={tot_policy.contracts:4d} net_pnl={tot_policy.net_pnl:7.3f} "
        f"avg_net/contract={tot_policy.avg_net_per_contract:6.3f}"
    )
    print(
        f"baseline: bets={tot_base.bets:4d} contracts={tot_base.contracts:4d} net_pnl={tot_base.net_pnl:7.3f} "
        f"avg_net/contract={(tot_base.net_pnl / tot_base.contracts if tot_base.contracts else 0.0):6.3f}"
    )

    if args.no_write:
        return

    candidates_tag = (
        f"t{float(candidates[0]):.3f}"
        if len(candidates) == 1
        else "t" + "-".join(f"{float(t):.3f}" for t in candidates) + f"_{args.threshold_select_mode}"
    )
    out_path = (
        Path(args.out)
        if args.out
        else THESIS_DIR
        / (
            "walkforward_rule_a_ij_tiered_policy_"
            f"{(args.league or 'all').lower()}_{primary}-{confirmer}_fee{args.fee_mode}_"
            f"uI{int(args.units_i)}_uJJ{int(args.units_jj)}_{candidates_tag}_"
            f"{args.start_date.replace('-','')}_{args.end_date.replace('-','')}.csv"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote -> {out_path}")


if __name__ == "__main__":
    main()

