# kalshi_utils.py
import os
import time
import json
import base64
import requests

from typing import Any, Dict, List, Optional

# we need these to sign private calls
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Public/unauth markets (read-only)
KALSHI_PUBLIC_BASE_DEFAULT = "https://demo-api.kalshi.co/trade-api/v2"

# Private/trading API
KALSHI_PRIVATE_BASE_DEFAULT = "https://api.elections.kalshi.com"

def has_private_creds() -> bool:
    """Return True if private key/id appear set in env."""
    key_id = (os.getenv("KALSHI_API_KEY_ID") or "").strip()
    pem_str = (os.getenv("KALSHI_API_PRIVATE_KEY") or "").strip()
    key_path = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()
    return bool(key_id and (pem_str or key_path))


def _normalize_base(url: str) -> str:
    """Ensure the base ends with /trade-api/v2 exactly once."""
    raw = url.strip().rstrip("/")
    if raw.endswith("/trade-api/v2"):
        return raw
    cut = raw.lower().find("/trade-api/v2")
    if cut != -1:
        raw = raw[:cut]
    return f"{raw}/trade-api/v2".rstrip("/")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _load_private_key(pem_str: str):
    return serialization.load_pem_private_key(
        pem_str.encode("utf-8"),
        password=None,
    )


def _signed_request(
    method: str,
    path: str,
    json_body: Optional[Dict[str, Any]] = None,
) -> requests.Response:
    """
    Call a private Kalshi endpoint using the RSA-signing scheme from their docs.
    path should look like: "/trade-api/v2/portfolio/balance" (without the host)
    """
    # Resolve config at call time so env.list loaders that run later are honored.
    key_id = (os.getenv("KALSHI_API_KEY_ID") or "").strip()
    pem_str = (os.getenv("KALSHI_API_PRIVATE_KEY") or "").strip()
    # Ignore placeholder values commonly found in templates
    if pem_str == "YOUR_KALSHI_PRIVATE_KEY" or len(pem_str) < 50:
        pem_str = ""
    key_path = (os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip()

    pem_bytes = None
    if pem_str:
        pem_bytes = pem_str.encode("utf-8")
    elif key_path:
        try:
            with open(key_path, "rb") as f:
                pem_bytes = f.read()
        except OSError:
            pem_bytes = None

    if not key_id or not pem_bytes:
        raise RuntimeError("Kalshi private call requested but KALSHI_API_KEY_ID or private key is not set")

    try:
        private_key = serialization.load_pem_private_key(
            pem_bytes,
            password=None,
        )
    except Exception as e:
        print(f"DEBUG: load_pem_private_key FAILED: {e}")
        # Dump first 20 bytes hex to see if it's garbage
        print(f"DEBUG: First 50 bytes: {pem_bytes[:50]!r}")
        raise

    ts = str(int(time.time() * 1000))
    method_up = method.upper()
    path_no_query = path.split("?", 1)[0]

    msg = f"{ts}{method_up}{path_no_query}"
    body_str = ""
    if json_body is not None:
        body_str = json.dumps(json_body, separators=(",", ":"))

    signature = private_key.sign(
        msg.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )

    sig_b64 = base64.b64encode(signature).decode("utf-8")

    headers = {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": sig_b64,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json",
    }

    raw_base = (
        os.getenv("KALSHI_API_BASE")
        or os.getenv("KALSHI_PRIVATE_BASE")
        or os.getenv("KALSHI_BASE_PRIVATE")
        or os.getenv("KALSHI_BASE")
        or KALSHI_PRIVATE_BASE_DEFAULT
    )
    base = _normalize_base(raw_base)

    if path.startswith("/trade-api"):
        cleaned = path.replace("/trade-api/v2", "", 1)
        url = base + cleaned
    else:
        url = base + path
    resp = requests.request(
        method=method_up,
        url=url,
        headers=headers,
        data=body_str if body_str else None,
        timeout=30,
    )
    return resp


# ---------------------------------------------------------------------------
# PUBLIC (no key) CALLS
# ---------------------------------------------------------------------------

def list_public_markets(limit: int = 50, status: str = "open", series_ticker: Optional[str] = None) -> Dict[str, Any]:
    """Hit the public endpoint to list markets (optionally filtered by series)."""
    params = {
        "limit": limit,
        "status": status,
    }
    if series_ticker:
        params["series_ticker"] = series_ticker
    base = _normalize_base(
        os.getenv("KALSHI_PUBLIC_BASE")
        or os.getenv("KALSHI_BASE_PUBLIC")
        or KALSHI_PUBLIC_BASE_DEFAULT
    )
    r = requests.get(f"{base}/markets", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# PRIVATE CALLS
# ---------------------------------------------------------------------------

def get_portfolio() -> Dict[str, Any]:
    """Return portfolio balance and positions in a single payload."""
    balance_resp = _signed_request("GET", "/trade-api/v2/portfolio/balance")
    if balance_resp.status_code != 200:
        raise RuntimeError(f"Kalshi portfolio balance error {balance_resp.status_code}: {balance_resp.text}")

    positions_resp = _signed_request("GET", "/trade-api/v2/portfolio/positions")
    if positions_resp.status_code != 200:
        raise RuntimeError(f"Kalshi portfolio positions error {positions_resp.status_code}: {positions_resp.text}")

    balance_data = balance_resp.json()
    positions_data = positions_resp.json()

    market_positions = positions_data.get("market_positions") or []
    event_positions = positions_data.get("event_positions") or []

    combined: Dict[str, Any] = {
        "balance": balance_data.get("balance"),
        "portfolio_value": balance_data.get("portfolio_value"),
        "updated_ts": balance_data.get("updated_ts"),
        "market_positions": market_positions,
        "event_positions": event_positions,
    }
    combined["positions"] = market_positions
    combined["open_positions"] = market_positions

    return combined


def place_order(
    ticker: str,
    side: str = "yes",
    count: int = 1,
    price: int = 1,
    action: str = "buy",
) -> Dict[str, Any]:
    """
    Very small helper that buys/sells 1 contract.
    price is in cents (1 = $0.01)
    """
    if os.getenv("KALSHI_DRY_RUN", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("Kalshi place_order disabled (dry run mode)")
    import uuid
    side_norm = side.lower()
    action_norm = action.lower()
    if action_norm not in ("buy", "sell"):
        action_norm = "buy"
    order_data = {
        "ticker": ticker,
        "action": action_norm,
        "side": side_norm,
        "count": count,
        "type": "limit",
        "yes_price": price if side_norm == "yes" else None,
        "no_price": price if side_norm == "no" else None,
        "client_order_id": str(uuid.uuid4()),
    }
    # strip keys that are None
    order_data = {k: v for k, v in order_data.items() if v is not None}

    r = _signed_request("POST", "/trade-api/v2/portfolio/orders", order_data)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Kalshi order error {r.status_code}: {r.text}")
    return r.json()


def close_position(
    ticker: str,
    side: str,
    count: int,
    price: int,
) -> Dict[str, Any]:
    """
    Place a SELL order to reduce/close an existing position.
    `side` is 'yes' or 'no' (the side we're currently long).
    """
    return place_order(
        ticker=ticker,
        side=side,
        count=count,
        price=price,
        action="sell",
    )


def get_fills(ticker: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """Fetch recent fills (trades)."""
    path = "/trade-api/v2/portfolio/fills"
    # Manual query string construction to satisfy signature requirements
    params = [f"limit={limit}"]
    if ticker:
        params.append(f"ticker={ticker}")
    
    if params:
        path += "?" + "&".join(params)
        
    r = _signed_request("GET", path)
    if r.status_code != 200:
        raise RuntimeError(f"Kalshi fills error {r.status_code}: {r.text}")
    return r.json()


def get_orders(status: str = "resting", ticker: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch orders. 
    status options: 'resting' (active), 'canceled', 'executed'.
    """
    path = "/trade-api/v2/portfolio/orders"
    params = []
    if status:
        params.append(f"status={status}")
    if ticker:
        params.append(f"ticker={ticker}")
        
    if params:
        path += "?" + "&".join(params)

    r = _signed_request("GET", path)
    if r.status_code != 200:
        raise RuntimeError(f"Kalshi get_orders error {r.status_code}: {r.text}")
    return r.json()


def cancel_order(order_id: str) -> Dict[str, Any]:
    """Cancel a specific order by ID."""
    path = f"/trade-api/v2/portfolio/orders/{order_id}"
    r = _signed_request("DELETE", path)
    if r.status_code != 200:
        # 200 OK usually returns the canceled order details
        raise RuntimeError(f"Kalshi cancel_order error {r.status_code}: {r.text}")
    return r.json()


def get_markets(limit: int = 500, status: str = "open", series_ticker: Optional[str] = None, get_all: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch list of markets (authenticated).
    Supports cursor-based pagination if get_all=True.
    """
    path = "/trade-api/v2/markets"
    all_markets = []
    cursor = None
    
    while True:
        params = [f"limit={limit}", f"status={status}"]
        if series_ticker:
            params.append(f"series_ticker={series_ticker}")
        if cursor:
            params.append(f"cursor={cursor}")
        
        url_path = path + "?" + "&".join(params)
        r = _signed_request("GET", url_path)
        if r.status_code != 200:
            raise RuntimeError(f"Kalshi get_markets error {r.status_code}: {r.text}")
        
        data = r.json()
        markets = data.get("markets", [])
        all_markets.extend(markets)
        
        cursor = data.get("cursor")
        if not get_all or not cursor:
            break
            
    return all_markets
