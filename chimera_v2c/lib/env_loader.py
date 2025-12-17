from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_env_from_env_list(env_path: Optional[Path] = None) -> None:
    """
    Load key=value pairs from config/env.list into os.environ if not already present.
    Also bridge SPORTRADAR_* -> SR_* for the Sportradar client expectations.
    """
    if env_path is None:
        env_path = Path("config") / "env.list"
    if not env_path.exists():
        return

    text = env_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or value is None:
            continue
        # Force overwrite to ensure file values take precedence
        os.environ[key] = value

    if "SR_API_KEY" not in os.environ and "SPORTRADAR_API_KEY" in os.environ:
        os.environ["SR_API_KEY"] = os.environ["SPORTRADAR_API_KEY"]
    if "SR_PREFER_AUTH" not in os.environ and "SPORTRADAR_AUTH_MODE" in os.environ:
        os.environ["SR_PREFER_AUTH"] = os.environ["SPORTRADAR_AUTH_MODE"]

    # Kalshi public (read-only) base:
    # Many tools rely on `KALSHI_PUBLIC_BASE` for fetching live GAME markets/mids.
    # If it isn't explicitly set, fall back to the configured Kalshi base.
    if "KALSHI_PUBLIC_BASE" not in os.environ:
        fallback = os.environ.get("KALSHI_BASE") or os.environ.get("KALSHI_API_BASE")
        if fallback:
            os.environ["KALSHI_PUBLIC_BASE"] = fallback
