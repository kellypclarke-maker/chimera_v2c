from __future__ import annotations

import argparse
import datetime as dt
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.lib import kalshi_utils
from chimera_v2c.tools.kalshi_place_limit_order import _get_yes_bid_ask_for_ticker


@dataclass(frozen=True)
class SheetOrder:
    ticker: str
    action: str
    side: str
    count: int
    price_cents: int


def _latest_sheet_for_date(date: dt.date) -> Path:
    yyyymmdd = date.strftime("%Y%m%d")
    d = Path("reports/trade_sheets/rule_a_maker_orders") / yyyymmdd
    candidates = sorted(d.glob(f"rule_a_maker_orders_{yyyymmdd}_*.csv"))
    if not candidates:
        raise SystemExit(f"[error] no maker order sheets found under {d}")
    return candidates[-1]


def _parse_sheet_orders(sheet_path: Path) -> list[SheetOrder]:
    df = pd.read_csv(sheet_path)
    if df.empty:
        return []

    required = ["market_ticker_away", "contracts", "maker_limit_price_cents"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] sheet missing columns: {missing}")

    df["maker_limit_price_cents"] = pd.to_numeric(df["maker_limit_price_cents"], errors="coerce")
    df = df[df["maker_limit_price_cents"].notna()].copy()
    if df.empty:
        return []

    orders: list[SheetOrder] = []
    for row in df.to_dict(orient="records"):
        ticker = str(row.get("market_ticker_away") or "").strip().upper()
        count = int(row.get("contracts") or 0)
        price = int(float(row.get("maker_limit_price_cents")))
        if not ticker or count <= 0:
            continue
        if price < 1 or price > 99:
            continue
        orders.append(SheetOrder(ticker=ticker, action="buy", side="yes", count=count, price_cents=price))

    return orders


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch place Rule A maker orders from a maker order sheet (PRIVATE API; defaults to dry-run)."
    )
    p.add_argument("--date", default="", help="YYYY-MM-DD; used to auto-pick latest maker sheet for that date.")
    p.add_argument("--sheet", default="", help="Explicit maker sheet CSV path.")
    p.add_argument(
        "--env-list",
        default="config/env.list",
        help="Env list path for private creds (default: config/env.list).",
    )
    p.add_argument(
        "--require-maker",
        action="store_true",
        help="Abort any order whose YES limit would cross current YES ask (i.e., price >= ask).",
    )
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between orders (default: 0).",
    )
    p.add_argument(
        "--max-orders",
        type=int,
        default=0,
        help="Optional cap on number of orders submitted (0 = no cap).",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Actually place orders (otherwise dry-run).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.sheet:
        sheet_path = Path(str(args.sheet))
    else:
        if not str(args.date).strip():
            raise SystemExit("[error] pass either --sheet or --date")
        date = dt.date.fromisoformat(str(args.date).strip())
        sheet_path = _latest_sheet_for_date(date)

    orders = _parse_sheet_orders(sheet_path)
    if not orders:
        print(f"[ok] no actionable orders in sheet: {sheet_path}")
        return

    load_env_from_env_list(Path(str(args.env_list)))
    if not kalshi_utils.has_private_creds():
        raise SystemExit(
            "[error] Kalshi private creds not configured (need KALSHI_API_KEY_ID + private key). "
            "Check config/env.list and ensure KALSHI_PRIVATE_KEY_PATH points to a real .pem/.key file."
        )

    max_orders = int(args.max_orders)
    if max_orders < 0:
        raise SystemExit("[error] --max-orders must be >= 0")

    print(f"# Source sheet: {sheet_path}")
    print(f"# Orders: {len(orders)} (confirm={bool(args.confirm)} require_maker={bool(args.require_maker)})")

    submitted = 0
    for o in orders:
        if max_orders and submitted >= max_orders:
            print(f"[stop] reached --max-orders={max_orders}")
            break

        if args.require_maker:
            bid, ask = _get_yes_bid_ask_for_ticker(o.ticker)
            if ask is None:
                print(f"[skip] {o.ticker}: require-maker but missing yes_ask")
                continue
            if o.price_cents >= int(ask):
                print(f"[skip] {o.ticker}: maker guard (limit {o.price_cents}c crosses ask {ask}c)")
                continue

        if not args.confirm:
            print(f"[dry-run] place limit: ticker={o.ticker} action={o.action} side={o.side} count={o.count} price={o.price_cents}c")
            submitted += 1
            continue

        try:
            resp = kalshi_utils.place_order(
                ticker=o.ticker,
                side=o.side,
                count=o.count,
                price=o.price_cents,
                action=o.action,
            )
        except Exception as exc:
            print(f"[error] failed {o.ticker}: {exc}")
            continue

        order_id = resp.get("order_id") or resp.get("id") or ""
        print(f"[ok] placed {o.ticker} count={o.count} price={o.price_cents}c order_id={order_id}")
        submitted += 1

        sleep_s = float(args.sleep_seconds or 0.0)
        if sleep_s > 0:
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()

