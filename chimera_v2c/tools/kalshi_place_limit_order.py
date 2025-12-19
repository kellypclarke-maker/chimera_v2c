from __future__ import annotations

import argparse
from typing import Any

from pathlib import Path

from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.lib import kalshi_utils


def _infer_series_ticker(ticker: str) -> str:
    return (ticker.split("-", 1)[0] or "").strip().upper()


def _get_yes_bid_ask_for_ticker(ticker: str) -> tuple[int | None, int | None]:
    series_ticker = _infer_series_ticker(ticker)
    if not series_ticker:
        return None, None

    markets = kalshi_utils.get_markets(limit=500, status="open", series_ticker=series_ticker, get_all=False)
    for m in markets:
        if not isinstance(m, dict):
            continue
        if str(m.get("ticker") or "").strip().upper() != ticker.strip().upper():
            continue
        yes_bid = m.get("yes_bid")
        yes_ask = m.get("yes_ask")
        try:
            bid_cents = None if yes_bid is None else int(yes_bid)
            ask_cents = None if yes_ask is None else int(yes_ask)
        except Exception:
            bid_cents, ask_cents = None, None
        return bid_cents, ask_cents

    return None, None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Place a single Kalshi limit order (PRIVATE API; real trades). Defaults to dry-run."
    )
    p.add_argument("--ticker", required=True, help="Kalshi market ticker (e.g. KXNBAGAME-...).")
    p.add_argument("--side", default="yes", choices=["yes", "no"], help="Contract side (default: yes).")
    p.add_argument("--action", default="buy", choices=["buy", "sell"], help="Action (default: buy).")
    p.add_argument("--count", type=int, required=True, help="Contracts to trade.")
    p.add_argument("--price-cents", type=int, required=True, help="Limit price in cents (1..99).")
    p.add_argument(
        "--env-list",
        default="config/env.list",
        help="Env list path for private creds (default: config/env.list).",
    )
    p.add_argument(
        "--require-maker",
        action="store_true",
        help="Abort if current YES ask is <= your YES limit (guards against crossing the spread).",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Actually place the order (otherwise dry-run).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    ticker = str(args.ticker).strip().upper()
    side = str(args.side).strip().lower()
    action = str(args.action).strip().lower()
    count = int(args.count)
    price_cents = int(args.price_cents)

    if count <= 0:
        raise SystemExit("[error] --count must be > 0")
    if price_cents < 1 or price_cents > 99:
        raise SystemExit("[error] --price-cents must be in [1, 99]")

    load_env_from_env_list(Path(str(args.env_list)))
    if not kalshi_utils.has_private_creds():
        raise SystemExit(
            "[error] Kalshi private creds not configured (need KALSHI_API_KEY_ID + private key). "
            "Check config/env.list and ensure KALSHI_PRIVATE_KEY_PATH points to a real .pem file."
        )

    if args.require_maker and action == "buy" and side == "yes":
        bid, ask = _get_yes_bid_ask_for_ticker(ticker)
        if ask is None:
            raise SystemExit("[error] require-maker requested but could not fetch current yes_ask for ticker")
        if price_cents >= ask:
            raise SystemExit(
                f"[error] maker guard: your yes limit {price_cents}c crosses current yes_ask {ask}c (ticker={ticker})"
            )

    if not args.confirm:
        print(f"[dry-run] place limit order: ticker={ticker} action={action} side={side} count={count} price_cents={price_cents}")
        if args.require_maker:
            bid, ask = _get_yes_bid_ask_for_ticker(ticker)
            print(f"[dry-run] current book: yes_bid={bid} yes_ask={ask}")
        return

    resp: dict[str, Any] = kalshi_utils.place_order(
        ticker=ticker,
        side=side,
        count=count,
        price=price_cents,
        action=action,
    )
    order_id = resp.get("order_id") or resp.get("id") or ""
    print(f"[ok] placed order: ticker={ticker} side={side} action={action} count={count} price_cents={price_cents} order_id={order_id}")


if __name__ == "__main__":
    main()
