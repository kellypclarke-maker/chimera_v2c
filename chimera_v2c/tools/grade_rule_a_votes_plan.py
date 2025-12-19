#!/usr/bin/env python
"""
Grade a Rule-A votes plan CSV against realized outcomes in daily ledgers (read-only).

This tool expects a CSV produced by `log_rule_a_votes_plan.py` (or the same schema)
with optional fill columns populated:
  - contracts_filled (int)
  - price_away_filled (dollars)
  - fees_filled (dollars)
  - fill_ts_utc (ISO timestamp; optional)

If fill fields are missing/blank, you can choose whether to assume "not filled"
or to grade as if the planned orders were filled (useful for paper/OOS tracking).

If `--fills-csv` is provided, missing fill fields are automatically backfilled from
Kalshi private-account fills exported by `export_kalshi_fills.py` (buy YES on the
away ticker), scoped to the plan's local `date` (using `date_local` if present in
the fills CSV, else `date_utc`).

Outputs:
  - A results CSV next to the input plan by default.

Safety:
  - Read-only on daily ledgers (does not modify any ledger files).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from chimera_v2c.src.kalshi_fees import maker_fee_dollars, taker_fee_dollars
from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games


def _parse_int(value: object) -> Optional[int]:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_float(value: object) -> Optional[float]:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_bool(value: object) -> Optional[bool]:
    s = str(value or "").strip().lower()
    if not s:
        return None
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _clamp_price(p: float) -> float:
    return max(0.01, min(0.99, float(p)))


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


@dataclass
class FillTotals:
    bets: int = 0
    contracts: int = 0
    risked: float = 0.0
    fees: float = 0.0


def _gross_pnl_per_contract(*, price_away: float, home_win: int) -> float:
    price = _clamp_price(float(price_away))
    return (1.0 - price) if int(home_win) == 0 else -price


def _grade_one(*, contracts: int, price_away: float, fees: float, home_win: int) -> Tuple[float, float, float]:
    c = int(contracts)
    if c <= 0:
        return 0.0, 0.0, 0.0
    price = _clamp_price(float(price_away))
    gross = float(c) * _gross_pnl_per_contract(price_away=price, home_win=int(home_win))
    risked = float(c) * price
    return gross, float(fees), risked


def _load_outcomes(*, date_iso: str) -> Dict[Tuple[str, str], int]:
    games = load_games(daily_dir=LEDGER_DIR, start_date=date_iso, end_date=date_iso, models=["kalshi_mid"])
    out: Dict[Tuple[str, str], int] = {}
    for g in games:
        if g.home_win not in (0.0, 1.0):
            continue
        out[(g.league, g.matchup)] = int(g.home_win)
    return out


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("[error] no rows to write")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _load_fills_agg(
    *,
    fills_csv: Path,
    plan_date_iso: str,
    date_slop_days: int,
) -> Dict[str, Tuple[int, float, str, float]]:
    """
    Returns: {ticker: (contracts_yes_equiv, vwap_yes_equiv_price_dollars, last_fill_ts_utc, fees_sum_dollars)}
    """
    if not fills_csv.exists():
        raise SystemExit(f"[error] missing fills csv: {fills_csv}")

    fills_rows = _read_rows(fills_csv)

    plan_day: Optional[date_type] = None
    if plan_date_iso:
        try:
            plan_day = date_type.fromisoformat(plan_date_iso)
        except ValueError:
            plan_day = None

    # We'll build intermediate sums as: ticker -> (contracts_sum, dollars_sum, last_ts_utc, fees_sum)
    sums: Dict[str, Tuple[int, float, str, float]] = {}
    for r in fills_rows:
        date_key = str(r.get("date_local") or r.get("date_utc") or "").strip()
        if date_key and plan_day is not None:
            try:
                key_day = date_type.fromisoformat(date_key)
            except ValueError:
                key_day = None
            if key_day is not None and abs((key_day - plan_day).days) > int(date_slop_days):
                continue

        ticker = str(r.get("ticker") or "").strip().upper()
        if not ticker:
            continue

        action = str(r.get("action") or "").strip().lower()
        side = str(r.get("side") or "").strip().lower()
        # YES-equivalent longs:
        #  - buy YES @ p
        #  - sell NO @ q  (equivalent to buy YES @ (1-q))
        is_buy_yes = action == "buy" and side == "yes"
        is_sell_no = action == "sell" and side == "no"
        if not (is_buy_yes or is_sell_no):
            continue

        contracts = _parse_int(r.get("count"))
        if contracts is None or contracts <= 0:
            continue

        price_cents = _parse_int(r.get("price_cents"))
        if price_cents is None or price_cents <= 0:
            continue

        price_side = float(price_cents) / 100.0
        price_yes_equiv = (1.0 - price_side) if is_sell_no else price_side

        is_taker = _parse_bool(r.get("is_taker"))
        if is_taker is True:
            fee = taker_fee_dollars(contracts=int(contracts), price=_clamp_price(price_side))
        elif is_taker is False:
            fee = maker_fee_dollars(contracts=int(contracts), price=_clamp_price(price_side))
        else:
            fee = taker_fee_dollars(contracts=int(contracts), price=_clamp_price(price_side))

        ts_utc = str(r.get("created_time_utc") or "").strip()
        prev_c, prev_dollars, prev_ts, prev_fees = sums.get(ticker, (0, 0.0, "", 0.0))
        new_c = prev_c + int(contracts)
        new_dollars = prev_dollars + (float(contracts) * float(price_yes_equiv))
        new_ts = ts_utc if ts_utc and (not prev_ts or ts_utc > prev_ts) else prev_ts
        sums[ticker] = (new_c, new_dollars, new_ts, float(prev_fees) + float(fee))

    out: Dict[str, Tuple[int, float, str, float]] = {}
    for ticker, (contracts, dollars_sum, last_ts, fees_sum) in sums.items():
        if contracts <= 0:
            continue
        out[ticker] = (int(contracts), float(dollars_sum) / float(contracts), str(last_ts), float(fees_sum))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Grade a Rule-A votes plan CSV against daily ledger outcomes.")
    ap.add_argument("--plan-csv", required=True, help="Path to a rule_a_votes_plan_*.csv file.")
    ap.add_argument(
        "--fills-csv",
        default="",
        help="Optional Kalshi fills CSV (from export_kalshi_fills.py). Missing fill fields are backfilled per market_ticker_away.",
    )
    ap.add_argument(
        "--execution-log",
        default="reports/execution_logs/rule_a_execution_log.csv",
        help="Optional consolidated Rule-A execution log (used when --fills-csv is omitted).",
    )
    ap.add_argument(
        "--fills-date-slop-days",
        type=int,
        default=1,
        help="Allow fills from plan-date +/- N days when backfilling (default: 1; helps if you pre-buy the night before).",
    )
    ap.add_argument(
        "--assume-filled",
        action="store_true",
        help="If fill columns are blank, grade as if contracts_planned/price_away_planned were filled (default: treat as not filled).",
    )
    ap.add_argument("--out", default="", help="Optional output CSV path (default: next to plan).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = Path(str(args.plan_csv))
    if not plan_path.exists():
        raise SystemExit(f"[error] missing plan csv: {plan_path}")

    plan_rows = _read_rows(plan_path)
    if not plan_rows:
        raise SystemExit("[error] empty plan csv")

    date_iso = str(plan_rows[0].get("date") or "").strip()
    if not date_iso:
        raise SystemExit("[error] plan csv missing 'date' column value on first row")

    outcomes = _load_outcomes(date_iso=date_iso)
    fills_agg: Dict[str, Tuple[int, float, str, float]] = {}
    if str(args.fills_csv).strip():
        fills_agg = _load_fills_agg(
            fills_csv=Path(str(args.fills_csv)),
            plan_date_iso=date_iso,
            date_slop_days=int(args.fills_date_slop_days),
        )
    else:
        exec_path = Path(str(args.execution_log))
        if exec_path.exists():
            fills_agg = _load_fills_agg(
                fills_csv=exec_path,
                plan_date_iso=date_iso,
                date_slop_days=int(args.fills_date_slop_days),
            )

    totals = Totals()
    fill_totals = FillTotals()
    out_rows: List[Dict[str, object]] = []

    for r in plan_rows:
        league = str(r.get("league") or "").strip().lower()
        matchup = str(r.get("matchup") or "").strip().upper()
        home_win = outcomes.get((league, matchup))
        if home_win is None:
            status = "missing_outcome"
        else:
            status = "ok"

        planned_contracts = _parse_int(r.get("contracts_planned"))
        planned_price = _parse_float(r.get("price_away_planned"))
        fill_contracts = _parse_int(r.get("contracts_filled"))
        fill_price = _parse_float(r.get("price_away_filled"))
        fill_fees = _parse_float(r.get("fees_filled"))
        fill_ts_utc = str(r.get("fill_ts_utc") or "").strip()

        market_ticker_away = str(r.get("market_ticker_away") or "").strip().upper()
        fills_contracts = fills_vwap = fills_last_ts = fills_fees_sum = None
        if fills_agg and market_ticker_away and market_ticker_away in fills_agg:
            fills_contracts, fills_vwap, fills_last_ts, fills_fees_sum = fills_agg[market_ticker_away]
            if fill_contracts is None:
                fill_contracts = int(fills_contracts)
            if fill_price is None:
                fill_price = float(fills_vwap)
            if not fill_ts_utc and fills_last_ts:
                fill_ts_utc = str(fills_last_ts)
            if fill_fees is None and fills_fees_sum is not None:
                fill_fees = float(fills_fees_sum)

        if fill_contracts is None:
            if args.assume_filled:
                fill_contracts = int(planned_contracts or 0)
            else:
                fill_contracts = 0
        if fill_price is None:
            fill_price = float(planned_price or 0.0)
        if fill_fees is None:
            fill_fees = taker_fee_dollars(contracts=int(fill_contracts), price=_clamp_price(float(fill_price or 0.0)))

        contracts_i = int(fill_contracts or 0)
        price_used = _clamp_price(float(fill_price or 0.0)) if contracts_i > 0 else 0.0
        fees = float(fill_fees or 0.0) if contracts_i > 0 else 0.0
        risked = float(contracts_i) * float(price_used) if contracts_i > 0 else 0.0

        if contracts_i > 0:
            fill_totals.bets += 1
            fill_totals.contracts += contracts_i
            fill_totals.risked += float(risked)
            fill_totals.fees += float(fees)

        gross = net = roi = 0.0
        if home_win is not None and contracts_i > 0:
            gross = float(contracts_i) * _gross_pnl_per_contract(price_away=float(price_used), home_win=int(home_win))
            net = float(gross - fees)
            roi = 0.0 if risked <= 0 else float(net / risked)

            totals.bets += 1
            totals.contracts += contracts_i
            totals.risked += float(risked)
            totals.gross_pnl += float(gross)
            totals.fees += float(fees)

        out_rows.append(
            {
                **r,
                "home_win": "" if home_win is None else int(home_win),
                "fills_contracts": "" if fills_contracts is None else int(fills_contracts),
                "fills_vwap_price": "" if fills_vwap is None else round(float(fills_vwap), 6),
                "fills_last_ts_utc": "" if fills_last_ts is None else str(fills_last_ts),
                "fills_fees_sum": "" if fills_fees_sum is None else round(float(fills_fees_sum), 6),
                "contracts_used": int(fill_contracts),
                "price_used": "" if contracts_i <= 0 else round(float(price_used), 6),
                "fees_used": "" if contracts_i <= 0 else round(float(fees), 6),
                "fill_ts_used_utc": str(fill_ts_utc),
                "gross_pnl": "" if home_win is None else round(float(gross), 6),
                "net_pnl": "" if home_win is None else round(float(net), 6),
                "risked": round(float(risked), 6),
                "roi_net": "" if home_win is None else round(float(roi), 6),
                "grade_status": status,
            }
        )

    out_path = Path(str(args.out)) if args.out else plan_path.with_name(plan_path.stem + "_results.csv")
    _write_rows(out_path, out_rows)
    print(f"[ok] wrote results: {out_path}")
    print(
        "[summary] "
        + ", ".join(
            [
                f"bets={totals.bets}",
                f"contracts={totals.contracts}",
                f"risked={totals.risked:.2f}",
                f"fees={totals.fees:.2f}",
                f"net_pnl={totals.net_pnl:.2f}",
                f"roi_net={totals.roi_net:.4f}",
            ]
        )
    )
    print(
        "[fills] "
        + ", ".join(
            [
                f"bets={fill_totals.bets}",
                f"contracts={fill_totals.contracts}",
                f"risked={fill_totals.risked:.2f}",
                f"fees={fill_totals.fees:.2f}",
            ]
        )
    )


if __name__ == "__main__":
    main()
