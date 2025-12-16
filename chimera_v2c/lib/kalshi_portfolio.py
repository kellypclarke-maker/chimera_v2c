"""Unified access layer for Kalshi portfolio (balance + positions)."""
from __future__ import annotations

from typing import Dict, Any

from chimera_v2c.lib.kalshi_client import KalshiClient
from chimera_v2c.lib import kalshi_utils


def load_portfolio() -> Dict[str, Any]:
    """Return a portfolio payload using bearer auth when possible, RSA fallback otherwise."""

    client = KalshiClient()
    errors: Dict[str, Any] = {}
    if client.api_key:
        try:
            balance = client.fetch_portfolio()
        except Exception as exc:
            errors["balance"] = str(exc)
            balance = None
        try:
            positions_resp = client.fetch_positions()
        except Exception as exc:
            errors["positions"] = str(exc)
            positions_resp = None
        if balance or positions_resp:
            positions = []
            if isinstance(positions_resp, dict):
                positions = positions_resp.get("market_positions") or positions_resp.get("positions") or []
            return {
                "balance": balance,
                "positions": positions,
                "errors": errors or None,
            }
    try:
        payload = kalshi_utils.get_portfolio()
    except Exception as exc:
        return {"errors": {"rsa": str(exc)}, "positions": [], "balance": None}
    payload.setdefault("positions", payload.get("open_positions") or [])
    payload.setdefault("balance", {
        "balance": payload.get("balance"),
        "portfolio_value": payload.get("portfolio_value"),
        "updated_ts": payload.get("updated_ts"),
    })
    return payload
