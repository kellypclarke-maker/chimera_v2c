"""Kalshi WebSocket client (market data + optional portfolio channels)."""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

import websockets
from websockets.client import WebSocketClientProtocol

from chimera_v2c.lib import kalshi_utils

WS_PATH = "/trade-api/ws/v2"


def _env_url(key: str, default: str) -> str:
    url = os.getenv(key) or default
    return url.rstrip("/")


def _default_private_url() -> str:
    # docs: wss://api.elections.kalshi.com/trade-api/ws/v2
    base = (
        os.getenv("KALSHI_WS_URL_PRIVATE")
        or os.getenv("KALSHI_API_BASE")
        or os.getenv("KALSHI_BASE")
        or os.getenv("KALSHI_PRIVATE_BASE")
        or os.getenv("KALSHI_BASE_PRIVATE")
        or "wss://api.elections.kalshi.com"
    )
    base = base.rstrip("/")
    if base.endswith(WS_PATH):
        return f"wss://{base.split('://', 1)[-1]}"
    if base.endswith("/trade-api"):
        base = base.rsplit("/trade-api", 1)[0]
    return f"{base}{WS_PATH}"


def _default_public_url() -> str:
    # Demo env
    base = (
        os.getenv("KALSHI_WS_URL_PUBLIC")
        or os.getenv("KALSHI_PUBLIC_BASE")
        or os.getenv("KALSHI_BASE_PUBLIC")
        or "wss://demo-api.kalshi.co"
    )
    base = base.rstrip("/")
    if base.endswith(WS_PATH):
        return f"wss://{base.split('://', 1)[-1]}"
    if base.endswith("/trade-api"):
        base = base.rsplit("/trade-api", 1)[0]
    return f"{base}{WS_PATH}"


def _build_auth_headers(path: str = WS_PATH) -> Dict[str, str]:
    """
    Build Kalshi signing headers for WS handshake (same scheme as REST).
    Falls back to Bearer token if RSA material is missing.
    """
    headers: Dict[str, str] = {}
    api_key = (os.getenv("KALSHI_API_KEY") or "").strip()
    key_id = (os.getenv("KALSHI_API_KEY_ID") or "").strip()
    pem_str = (os.getenv("KALSHI_API_PRIVATE_KEY") or "").strip()
    key_path = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()

    if not pem_str and key_path:
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                pem_str = f.read()
        except OSError:
            pem_str = ""

    if key_id and pem_str:
        private_key = kalshi_utils._load_private_key(pem_str)  # type: ignore[attr-defined]
        ts = str(int(time.time() * 1000))
        msg = f"{ts}GET{path}"
        signature = private_key.sign(
            msg.encode("utf-8"),
            kalshi_utils.padding.PSS(
                mgf=kalshi_utils.padding.MGF1(kalshi_utils.hashes.SHA256()),
                salt_length=kalshi_utils.padding.PSS.DIGEST_LENGTH,
            ),
            kalshi_utils.hashes.SHA256(),
        )
        sig_b64 = kalshi_utils.base64.b64encode(signature).decode("utf-8")
        headers.update(
            {
                "KALSHI-ACCESS-KEY": key_id,
                "KALSHI-ACCESS-SIGNATURE": sig_b64,
                "KALSHI-ACCESS-TIMESTAMP": ts,
            }
        )
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _normalize_event(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize Kalshi WS payload to a simple shape.
    Expected envelope: {"type": "...", "msg": {...}, "sid": <int>}
    """
    msg_type = raw.get("type") or raw.get("msg_type")
    msg = raw.get("msg") or raw.get("data") or raw
    out: Dict[str, Any] = {
        "type": msg_type,
        "sid": raw.get("sid"),
        "market_ticker": None,
        "raw": raw,
    }
    if isinstance(msg, dict):
        out["market_ticker"] = msg.get("market_ticker")
        if msg_type == "ticker":
            out.update(
                {
                    "yes_bid": msg.get("yes_bid"),
                    "yes_ask": msg.get("yes_ask"),
                    "no_bid": msg.get("no_bid"),
                    "no_ask": msg.get("no_ask"),
                    "price": msg.get("price"),
                    "ts": msg.get("ts"),
                }
            )
        elif msg_type in {"orderbook_snapshot", "orderbook_delta"}:
            out.update({"levels": msg.get("levels"), "ts": msg.get("ts")})
        elif msg_type in {"trade", "fill"}:
            out.update({"side": msg.get("side"), "price": msg.get("price"), "count": msg.get("count"), "ts": msg.get("ts")})
        elif msg_type in {"marketPosition", "market_position"}:
            out.update({"position": msg.get("position"), "pnl": msg.get("realized_pnl"), "ts": msg.get("ts")})
    return out


CallbackType = Callable[[Dict[str, Any]], Awaitable[None]] | Callable[[Dict[str, Any]], None]


@dataclass
class KalshiWSClient:
    use_private: bool = True
    channels: List[str] = field(default_factory=lambda: ["ticker"])
    public_url: str = field(default_factory=_default_public_url)
    private_url: str = field(default_factory=_default_private_url)
    _ws: Optional[WebSocketClientProtocol] = field(init=False, default=None)
    _cmd_id: int = field(init=False, default=1)

    async def connect(self) -> None:
        url = self.private_url if self.use_private else self.public_url
        headers = _build_auth_headers(WS_PATH) if self.use_private else {}
        try:
            self._ws = await websockets.connect(
                url,
                additional_headers=headers if headers else None,
                ping_interval=10,
                ping_timeout=20,
            )
        except Exception as exc:
            raise RuntimeError(f"Kalshi WS connect failed for {url}: {exc}") from exc

    def _next_id(self) -> int:
        self._cmd_id += 1
        return self._cmd_id

    async def subscribe_markets(self, tickers: Iterable[str], channels: Optional[List[str]] = None) -> None:
        if not self._ws:
            raise RuntimeError("WebSocket not connected.")
        chans = channels or self.channels or ["ticker"]
        payload = {
            "id": self._next_id(),
            "cmd": "subscribe",
            "params": {
                "channels": chans,
                "market_tickers": list(tickers),
            },
        }
        await self._ws.send(json.dumps(payload))

    async def subscribe_portfolio(self) -> None:
        if not self._ws:
            raise RuntimeError("WebSocket not connected.")
        payload = {"id": self._next_id(), "cmd": "subscribe", "params": {"channels": ["market_positions", "fill"]}}
        await self._ws.send(json.dumps(payload))

    async def run(self, callback: CallbackType) -> None:
        if not self._ws:
            await self.connect()
        assert self._ws is not None
        async for message in self._ws:
            try:
                raw = json.loads(message)
            except Exception:
                continue
            event = _normalize_event(raw) if isinstance(raw, dict) else {"type": "unknown", "raw": raw}
            res = callback(event)
            if asyncio.iscoroutine(res):
                await res

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None
