# src/embeddings/embedder.py
"""
Embedding utilities for an Enterprise Fraud Intelligence System.

Design goals:
- CI-safe (no external calls at import time)
- Lazy OpenAI client initialization
- Feature-flag controlled (can disable embeddings entirely)
- Graceful degradation with stable dummy vectors
- Backward compatibility for legacy callers
"""

import os
import threading
from typing import List, Optional

from src.observability.cost import add_request_cost, add_unpriced_response
from src.observability.events import emit_event
from src.observability.timing import elapsed_timer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDINGS_ENABLED = os.getenv("EMBEDDINGS_ENABLED", "true").lower() == "true"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Must match the pgvector column size in infra/supabase_schema.sql
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# Stable dummy dimension (used when embeddings are disabled or fail)
DUMMY_VECTOR_DIM = EMBEDDING_DIM

# Per-attempt deadline for an embedding request (the SDK default is 600 s). The
# SDK's own 2 retries are kept here (embedding failures degrade to a dummy
# vector rather than failing the request), so the worst case is roughly
# 3 x EMBEDDING_TIMEOUT_SECONDS plus the SDK's short backoff.
EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "10"))

# USD per 1M input tokens. Exact model-name match only; a model with no entry
# is UNPRICED (cost unknown, never $0.00), the same rule as the LLM price table.
# Verify against the provider's pricing page whenever this table changes.
EMBEDDING_PRICES_PER_1M = {
    "text-embedding-3-small": 0.02,
}


def estimate_embedding_cost(model: str, tokens: Optional[int]) -> Optional[float]:
    """USD for one embedding response, or None when the model is unpriced or usage is missing."""
    price = EMBEDDING_PRICES_PER_1M.get(model)
    if price is None or tokens is None:
        return None
    # 9 decimals, not 6: one query embedding is ~15 tokens (~$0.0000003), which
    # a 6-decimal round would turn into $0.00 and hide the whole cost.
    return round(tokens / 1_000_000 * price, 9)


# =============================================================================
# LAZY OPENAI CLIENT
# =============================================================================

_openai_client = None
_CLIENT_LOCK = threading.Lock()


def _get_openai_client():
    """
    Lazily instantiate OpenAI client.

    Guarantees:
    - No OpenAI dependency at import time
    - API key validated only when embeddings are actually requested
    - Safe for CI and unit tests
    """
    global _openai_client

    if _openai_client is not None:
        return _openai_client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    with _CLIENT_LOCK:
        if _openai_client is None:  # another thread may have created it first
            from openai import OpenAI  # local import → CI safe
            _openai_client = OpenAI(api_key=api_key, timeout=EMBEDDING_TIMEOUT_SECONDS)
        return _openai_client

def _record_embedding_response(response, duration_ms) -> None:
    """
    Telemetry and request cost for one embedding response. Never raises: an
    observability failure must not turn a successful embedding into a dummy vector.
    """
    try:
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", None) if usage is not None else None
        if tokens is None and usage is not None:
            tokens = getattr(usage, "prompt_tokens", None)
        cost = estimate_embedding_cost(EMBEDDING_MODEL, tokens)

        if cost is None:
            add_unpriced_response()      # usage missing or model unpriced: unknown, never $0
        else:
            add_request_cost(cost)

        emit_event(
            "embedding.completed", step="embedding", status="success",
            duration_ms=duration_ms,
            metadata={"model": EMBEDDING_MODEL, "total_tokens": tokens, "estimated_cost_usd": cost},
        )
    except Exception:
        logger.warning("[embedder] Failed to record embedding telemetry", exc_info=True)


def _record_embedding_failure(error: Exception) -> None:
    try:
        emit_event(
            "embedding.failed", step="embedding", status="failure",
            metadata={"model": EMBEDDING_MODEL, "error_type": type(error).__name__},
        )
    except Exception:
        logger.warning("[embedder] Failed to record embedding failure", exc_info=True)


# =============================================================================
# PUBLIC API
# =============================================================================

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.

    CI-safe behavior:
    - If EMBEDDINGS_ENABLED=false → returns stable dummy vectors
    - No OpenAI usage at import time
    - Errors degrade gracefully to dummy vectors

    Returns:
        List[List[float]] with one vector per input text
    """
    if not texts:
        return []

    if not EMBEDDINGS_ENABLED:
        logger.warning("[embedder] Embeddings disabled — returning dummy vectors")
        return [[0.0] * DUMMY_VECTOR_DIM for _ in texts]

    client = _get_openai_client()

    try:
        with elapsed_timer() as elapsed:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                dimensions=EMBEDDING_DIM,
            )
            duration_ms = elapsed()

        embeddings = [item.embedding for item in response.data]

    except Exception as e:
        logger.error(f"[embedder] Embedding generation failed: {e}")
        _record_embedding_failure(e)
        return [[0.0] * DUMMY_VECTOR_DIM for _ in texts]

    _record_embedding_response(response, duration_ms)
    logger.info(f"[embedder] Generated {len(embeddings)} embeddings")
    return embeddings


def embed_text(text: str) -> Optional[List[float]]:
    """
    Convenience wrapper for single-text embedding.
    """
    if not text:
        return None

    vectors = embed_texts([text])
    return vectors[0] if vectors else None

# =============================================================================
# BACKWARD COMPATIBILITY LAYER
# =============================================================================

class _LazyEmbeddingModel:
    """
    Backward-compatible proxy for legacy embedding_model usage.

    Supports:
    - embed_documents(texts)
    - embed_query(text)
    - embed_one(text)

    Used by:
    - reranker
    - scoring
    - legacy pipelines
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        vec = embed_text(text)
        return vec or []

    def embed_one(self, text: str) -> List[float]:
        """
        Legacy method expected by ranking / scoring modules.
        """
        vec = embed_text(text)
        return vec or []

embedding_model = _LazyEmbeddingModel()
