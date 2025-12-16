from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class TavilyError(RuntimeError):
    pass


def tavily_search(
    query: str,
    *,
    api_key: Optional[str] = None,
    max_results: int = 5,
    search_depth: str = "basic",
    timeout_s: float = 12.0,
) -> Dict[str, Any]:
    """
    Minimal Tavily search wrapper (salvaged from legacy tooling).

    Returns the raw JSON response as a dict. Does not fetch page bodies; use Tavily's
    `content`/`snippet` payloads and keep results grounded to avoid hallucinations.
    """
    clean_query = (query or "").strip()
    if not clean_query:
        raise TavilyError("query is required")

    key = (api_key or os.getenv("TAVILY_API_KEY") or "").strip()
    if not key:
        raise TavilyError("TAVILY_API_KEY not set")

    safe_max = max(1, min(int(max_results), 10))
    safe_depth = (search_depth or "basic").strip().lower()
    if safe_depth not in {"basic", "advanced"}:
        safe_depth = "basic"

    payload = {
        "api_key": key,
        "query": clean_query,
        "search_depth": safe_depth,
        "max_results": safe_max,
    }

    try:
        resp = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise TavilyError("unexpected Tavily response shape")
        return data
    except requests.RequestException as exc:
        raise TavilyError(f"Tavily request failed: {exc}") from exc


def format_tavily_results(data: Dict[str, Any], *, max_items: int = 5) -> str:
    """
    Render Tavily results into a compact, LLM-friendly text block.
    """
    results = data.get("results") or []
    if not isinstance(results, list) or not results:
        return "Tavily results: (none)"

    safe_max = max(1, min(int(max_items), 10))
    lines: List[str] = ["Tavily results:"]
    for item in results[:safe_max]:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()
        snippet = content or (item.get("snippet") or "").strip()
        label = title or url or "result"
        tail = f" ({url})" if url else ""
        body = f": {snippet}" if snippet else ""
        lines.append(f"- {label}{tail}{body}")
    return "\n".join(lines)

