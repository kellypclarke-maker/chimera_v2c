"""
Lightweight Kalshi WS listener to keep a local mid cache fresh.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/kalshi_ws_listener.py --league nba --series KXNBAGAME

Behavior:
  - Subscribes to GAME markets via WS (private by default).
  - Updates a local cache file (default: chimera_v2c/data/ws_mids.json) with yes_bid/yes_ask.
  - Planner will overlay these mids if the cache exists.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from chimera_v2c.lib.ws_client import KalshiWSClient
from chimera_v2c.lib import kalshi_utils
from chimera_v2c.lib.env_loader import load_env_from_env_list

from chimera_v2c.src.ws_mid_cache import update_cache_from_event, write_ws_cache, load_ws_cache


async def main_async(args: argparse.Namespace) -> None:
    load_env_from_env_list()
    cache_path = Path(args.cache)
    cache = load_ws_cache(cache_path)
    # Bootstrap tickers via REST for the series/league
    mkts = kalshi_utils.get_markets(series_ticker=args.series, get_all=True)
    tickers = [m["ticker"] for m in mkts if args.series in m.get("ticker", "")]
    if not tickers:
        print(f"[warn] no tickers found for series {args.series}")
        return
    client = KalshiWSClient(use_private=not args.public)
    await client.connect()
    await client.subscribe_markets(tickers, channels=["ticker"])
    print(f"[info] subscribed to {len(tickers)} markets")

    async def on_event(ev):
        update_cache_from_event(cache, ev)
        write_ws_cache(cache, cache_path)

    try:
        await client.run(on_event)
    finally:
        await client.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Kalshi WS listener -> ws_mids cache.")
    ap.add_argument("--league", default="nba", help="League (for future filtering)")
    ap.add_argument("--series", default="KXNBAGAME", help="Series ticker to subscribe")
    ap.add_argument("--cache", default="chimera_v2c/data/ws_mids.json", help="Path to mid cache JSON")
    ap.add_argument("--public", action="store_true", help="Use public WS (demo) instead of private")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
