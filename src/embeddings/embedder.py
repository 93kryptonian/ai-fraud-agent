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
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDINGS_ENABLED = os.getenv("EMBEDDINGS_ENABLED", "true").lower() == "true"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Stable dummy dimension (used when embeddings are disabled or fail)
DUMMY_VECTOR_DIM = 384

# =============================================================================
# LAZY OPENAI CLIENT
# =============================================================================

_openai_client = None


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

    from openai import OpenAI  # local import → CI safe
    _openai_client = OpenAI(api_key=api_key)
    return _openai_client

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
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        embeddings = [item.embedding for item in response.data]

        logger.info(f"[embedder] Generated {len(embeddings)} embeddings")
        return embeddings

    except Exception as e:
        logger.error(f"[embedder] Embedding generation failed: {e}")
        return [[0.0] * DUMMY_VECTOR_DIM for _ in texts]


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
