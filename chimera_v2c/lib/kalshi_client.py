import os
from typing import Any, Dict, Optional

import requests

DEFAULT_PRIVATE_BASE = "https://api.elections.kalshi.com"
DEFAULT_PUBLIC_BASE = "https://api.elections.kalshi.com"
TRADE_PATH = "/trade-api/v2"


def _normalize_base(url: str) -> str:
    """Ensure the base ends with the trade path exactly once."""
    raw = url.strip().rstrip("/")
    if raw.endswith(TRADE_PATH):
        return raw
    cut = raw.lower().find(TRADE_PATH)
    if cut != -1:
        raw = raw[:cut]
    return f"{raw}{TRADE_PATH}".rstrip("/")


class KalshiClient:
    def __init__(
        self,
        base: Optional[str] = None,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        env_private = (
            base
            or os.getenv("KALSHI_API_BASE")
            or os.getenv("KALSHI_BASE")
            or os.getenv("KALSHI_PRIVATE_BASE")
            or os.getenv("KALSHI_BASE_PRIVATE")
            or DEFAULT_PRIVATE_BASE
        )
        env_public = (
            os.getenv("KALSHI_PUBLIC_BASE")
            or os.getenv("KALSHI_BASE_PUBLIC")
            or env_private
            or DEFAULT_PUBLIC_BASE
        )

        self.base_private = _normalize_base(env_private)
        self.base_public = _normalize_base(env_public)

        self.api_key = (api_key or os.getenv("KALSHI_API_KEY", "")).strip()
        self.s = session or requests.Session()
        self.s.headers.update({"User-Agent": "chimera-kalshi/1.0"})
        if self.api_key:
            self.s.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _get(self, path: str, *, base: Optional[str] = None, timeout: float = 5.0, **params: Any) -> Dict[str, Any]:
        url = f"{(base or self.base_private)}{path}"
        resp = self.s.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception as exc:
            raise RuntimeError(f"Kalshi JSON decode error for {url}: {exc}") from exc

    # ---- Public-ish listing; try preferred path then graceful fallback
    def list_markets_public(self, **params: Any) -> Dict[str, Any]:
        """Public markets endpoint (no auth required)."""

        try:
            return self._get("/markets", base=self.base_public, **params)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (404, 400):
                return self._get("/exchange/markets", base=self.base_public, **params)
            raise

    # ---- Auth listing; returns stub when API key missing
    def list_markets_auth(self, **params: Any) -> Dict[str, Any]:
        if not self.api_key:
            try:
                data = self.list_markets_public(**params)
                data.setdefault("status", "ok")
                data.setdefault("reason", "no KALSHI_API_KEY; using public markets.")
                if "markets" not in data:
                    data["markets"] = self._stub_markets()
                return data
            except Exception:
                return {
                    "status": "stub",
                    "reason": "no KALSHI_API_KEY",
                    "markets": self._stub_markets(),
                }
        try:
            data = self.list_markets_public(**params)
            data.setdefault("status", "ok")
            if "markets" not in data:
                data["markets"] = self._stub_markets()
            return data
        except Exception as exc:
            return {
                "status": "stub",
                "reason": f"kalshi request failed: {exc}",
                "markets": self._stub_markets(),
            }

    def fetch_positions(self) -> Dict[str, Any]:
        if not self.api_key:
            return {"status": "stub", "reason": "no KALSHI_API_KEY", "positions": []}
        try:
            return self._get("/portfolio/positions")
        except requests.HTTPError as exc:
            return {"status": "error", "reason": str(exc), "positions": []}

    def fetch_portfolio(self) -> Dict[str, Any]:
        if not self.api_key:
            return {"status": "stub", "reason": "no KALSHI_API_KEY"}
        # Prefer the balance endpoint; if unavailable, fall back to legacy /portfolio.
        try:
            return self._get("/portfolio/balance")
        except requests.HTTPError as exc:
            try:
                return self._get("/portfolio")
            except requests.HTTPError:
                return {"status": "error", "reason": str(exc)}

    def fetch_fills(self, **params: Any) -> Dict[str, Any]:
        """Fetch historical fills (trades). Params: ticker, min_ts, max_ts, limit, cursor."""
        if not self.api_key:
            return {"status": "stub", "reason": "no KALSHI_API_KEY", "fills": []}
        try:
            return self._get("/portfolio/fills", **params)
        except requests.HTTPError as exc:
            return {"status": "error", "reason": str(exc), "fills": []}

    def list_series(self, **params: Any) -> Dict[str, Any]:
        return self._get("/series", **params)

    def list_events(self, **params: Any) -> Dict[str, Any]:
        return self._get("/events", **params)

    # ---- Optional ping endpoint(s), tolerant to 404
    def ping(self) -> Dict[str, Any]:
        for path in ("/status", "/exchange/status", "/ping"):
            try:
                return self._get(path)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code in (404, 400):
                    continue
                raise
        return {"ok": False, "error": "no status endpoint available"}

    def _stub_markets(self) -> list[Dict[str, Any]]:
        return [
            {
                "ticker": "NFL-SNF-TOTAL",
                "title": "Total points in Sunday Night Football",
                "yes_price": 0.57,
                "no_price": 0.43,
            },
            {
                "ticker": "ELEC-PRIMARY-2026",
                "title": "Will Candidate X win the 2026 primary?",
                "yes_price": 0.34,
                "no_price": 0.66,
            },
        ]
