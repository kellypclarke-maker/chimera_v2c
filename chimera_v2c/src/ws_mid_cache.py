from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


def load_ws_cache(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_ws_cache(cache: Dict[str, Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def update_cache_from_event(cache: Dict[str, Dict], event: Dict) -> None:
    """
    Update cache in-place from a normalized WS event with yes_bid/yes_ask.
    """
    ticker = event.get("market_ticker")
    if not ticker:
        return
    yes_bid = event.get("yes_bid")
    yes_ask = event.get("yes_ask")
    if yes_bid is None and yes_ask is None:
        return
    entry = cache.get(ticker, {})
    if yes_bid is not None:
        entry["yes_bid"] = yes_bid
    if yes_ask is not None:
        entry["yes_ask"] = yes_ask
    entry["ts"] = event.get("ts")
    cache[ticker] = entry
