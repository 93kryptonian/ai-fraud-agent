# src/rag/retriever_direct.py
"""
Direct document retriever (Supabase-backed).

Responsibilities:
- Retrieve top-k document chunks for RAG
- Provide CI-safe behavior via feature flags
- Lazily initialize Supabase client (no import-time side effects)

Design principles:
- Fail closed (empty results on error)
- Never break CI or local tests
- Keep retrieval logic isolated from RAG orchestration
"""

import os
from typing import List, Dict, Optional

from src.utils.logger import get_logger
from src.db.supabase_client import get_supabase
from src.embeddings.embedder import embed_text
from src.rag.ranking import rerank_chunks

logger = get_logger(__name__)

# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Allows retriever to be disabled in CI / tests
RETRIEVER_ENABLED = os.getenv("RETRIEVER_ENABLED", "true").lower() == "true"

# =============================================================================
# PUBLIC API
# =============================================================================

def retrieve_top_k(
    query: str,
    top_k: int = 5,
    source_name: Optional[str] = None,
) -> List[Dict]:
    """
    Retrieve top-k document chunks from Supabase.

    Behavior:
    - If RETRIEVER_ENABLED=false → returns empty list
    - Supabase client is created lazily at runtime
    - Errors are logged and swallowed (safe fallback)

    Returns:
        List of dicts containing:
        - content
        - source_name
        - page
        - similarity (pgvector cosine similarity)
    """
    logger.info(
        f"[retriever_direct] query={query!r} | source={source_name} | top_k={top_k}"
    )

    # ------------------------------------------------------------------
    # CI / TEST GUARD
    # ------------------------------------------------------------------
    if not RETRIEVER_ENABLED:
        logger.warning(
            "[retriever_direct] Retriever disabled via feature flag — returning empty result"
        )
        return []

    # ------------------------------------------------------------------
    # LAZY SUPABASE INITIALIZATION
    # ------------------------------------------------------------------
    try:
        supabase = get_supabase()
    except Exception as e:
        logger.error(
            f"[retriever_direct] Failed to initialize Supabase client: {e}"
        )
        return []

    # ------------------------------------------------------------------
    # QUERY EMBEDDING
    # ------------------------------------------------------------------
    query_vec = embed_text(query)
    if not query_vec or not any(query_vec):
        # Disabled / failed embeddings yield zero vectors → meaningless search
        logger.warning(
            "[retriever_direct] No usable query embedding — returning empty result"
        )
        return []

    # ------------------------------------------------------------------
    # VECTOR SEARCH (pgvector via match_documents RPC) + HYBRID RERANK
    # ------------------------------------------------------------------
    try:
        # filter must be NULL (not {}) for "no filter" — see match_documents()
        response = supabase.rpc(
            "match_documents",
            {
                "filter": {"source_name": source_name} if source_name else None,
                "query_embedding": query_vec,
            },
        ).execute()
        candidates = response.data or []

        rows = rerank_chunks(query, candidates, use_llm=False, top_k=top_k)

        logger.info(
            f"[retriever_direct] Retrieved {len(candidates)} candidates → {len(rows)} chunks"
        )
        return rows

    except Exception as e:
        logger.error(
            f"[retriever_direct] Supabase query failed: {e}",
            exc_info=True,
        )
        return []
