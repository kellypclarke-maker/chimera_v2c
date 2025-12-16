from __future__ import annotations

import os
from typing import List, Optional, Sequence

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - optional dependency
    genai = None  # type: ignore
    types = None  # type: ignore


class GeminiEmbeddingError(RuntimeError):
    pass


def embed_texts(
    texts: Sequence[str],
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: Optional[int] = None,
) -> Optional[List[List[float]]]:
    """
    Best-effort Gemini embeddings wrapper (salvaged concept from legacy tooling).

    Returns a list of embedding vectors, or None if embeddings are unavailable (e.g.,
    missing GEMINI_API_KEY or google-genai not installed).
    """
    if genai is None or types is None:
        return None

    clean_texts = [str(t or "").strip() for t in texts if str(t or "").strip()]
    if not clean_texts:
        return None

    key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        return None

    model_name = (model or os.getenv("EMBEDDING_MODEL") or "gemini-embedding-001").strip() or "gemini-embedding-001"
    cfg = types.EmbedContentConfig(task_type=task_type or None, output_dimensionality=output_dimensionality)

    try:
        client = genai.Client(api_key=key)
        resp = client.models.embed_content(model=model_name, contents=clean_texts, config=cfg)
    except Exception:
        return None

    embeddings = getattr(resp, "embeddings", None)
    if not embeddings:
        return None

    out: List[List[float]] = []
    for emb in embeddings:
        values = getattr(emb, "values", None)
        if not values:
            return None
        out.append([float(x) for x in values])
    return out

