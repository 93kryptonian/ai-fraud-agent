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
        - chunk_index
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
    # QUERY EXECUTION
    # ------------------------------------------------------------------
    try:
        query_builder = (
            supabase
            .table("document_embeddings")
            .select("content, source_name, page, chunk_index")
            .limit(top_k)
        )

        if source_name:
            query_builder = query_builder.eq("source_name", source_name)

        response = query_builder.execute()
        rows = response.data or []

        logger.info(
            f"[retriever_direct] Retrieved {len(rows)} chunks"
        )
        return rows

    except Exception as e:
        logger.error(
            f"[retriever_direct] Supabase query failed: {e}",
            exc_info=True,
        )
        return []
