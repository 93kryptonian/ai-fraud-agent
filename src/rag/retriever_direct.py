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
from src.observability.events import emit_event
from src.observability.timing import elapsed_timer

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

    # M6: "retrieval" and "ranking" are two separate observable stages
    # even though they're fused inside this one function — run_rag() never
    # calls them independently, so the boundary has to live here, not in
    # rag_chain.py. Each of the early-exit cases below is a genuinely
    # different reason for an empty result — collapsing them all into a
    # silent [] (as this function did pre-M6) hid that distinction; see
    # docs/observability-contract.md and the M6 design notes.

    # ------------------------------------------------------------------
    # CI / TEST GUARD
    # ------------------------------------------------------------------
    if not RETRIEVER_ENABLED:
        logger.warning(
            "[retriever_direct] Retriever disabled via feature flag — returning empty result"
        )
        emit_event(
            "retrieval.skipped", step="retrieval", status="skipped",
            metadata={
                "retrieval_method": "vector_rpc", "reason": "retriever_disabled",
                "candidate_count": 0, "selected_count": 0, "source_filter": source_name,
            },
        )
        return []

    with elapsed_timer() as elapsed:
        # ------------------------------------------------------------------
        # LAZY SUPABASE INITIALIZATION
        # ------------------------------------------------------------------
        try:
            supabase = get_supabase()
        except Exception as e:
            logger.error(
                f"[retriever_direct] Failed to initialize Supabase client: {e}"
            )
            emit_event(
                "retrieval.failed", step="retrieval", status="failure",
                duration_ms=elapsed(),
                metadata={
                    "retrieval_method": "vector_rpc", "error_type": type(e).__name__,
                    "source_filter": source_name,
                },
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
            emit_event(
                "retrieval.skipped", step="retrieval", status="skipped",
                duration_ms=elapsed(),
                metadata={
                    "retrieval_method": "vector_rpc", "reason": "no_embedding",
                    "candidate_count": 0, "selected_count": 0, "source_filter": source_name,
                },
            )
            return []

        # ------------------------------------------------------------------
        # VECTOR SEARCH (pgvector via match_documents RPC)
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
        except Exception as e:
            logger.error(
                f"[retriever_direct] Supabase query failed: {e}",
                exc_info=True,
            )
            emit_event(
                "retrieval.failed", step="retrieval", status="failure",
                duration_ms=elapsed(),
                metadata={
                    "retrieval_method": "vector_rpc", "error_type": type(e).__name__,
                    "source_filter": source_name,
                },
            )
            return []

        emit_event(
            "retrieval.completed", step="retrieval", status="success",
            duration_ms=elapsed(),
            metadata={
                "retrieval_method": "vector_rpc",
                "candidate_count": len(candidates),
                "source_filter": source_name,
            },
        )

    if not candidates:
        # Retrieval genuinely ran and succeeded; there's simply nothing to
        # rank. Distinct from the cases above where retrieval never ran at
        # all — see docs/observability-contract.md §3 on `skipped`.
        emit_event(
            "ranking.skipped", step="ranking", status="skipped",
            metadata={"candidate_count": 0, "selected_count": 0},
        )
        return []

    # ------------------------------------------------------------------
    # HYBRID RERANK — a separate stage, own try/except, so a bug in
    # ranking is never misattributed to "retrieval" (which already
    # succeeded by this point).
    # ------------------------------------------------------------------
    with elapsed_timer() as rank_elapsed:
        try:
            rows = rerank_chunks(query, candidates, use_llm=False, top_k=top_k)
        except Exception as e:
            logger.error(f"[retriever_direct] Ranking failed: {e}", exc_info=True)
            emit_event(
                "ranking.failed", step="ranking", status="failure",
                duration_ms=rank_elapsed(),
                metadata={"candidate_count": len(candidates), "error_type": type(e).__name__},
            )
            # Fail closed: no ranked results rather than crashing the
            # whole RAG request over a ranking bug.
            return []

        logger.info(
            f"[retriever_direct] Retrieved {len(candidates)} candidates → {len(rows)} chunks"
        )
        emit_event(
            "ranking.completed", step="ranking", status="success",
            duration_ms=rank_elapsed(),
            metadata={
                "candidate_count": len(candidates),
                "selected_count": len(rows),
                "reranker": "hybrid",  # use_llm=False always in production — see ranking.py
            },
        )
        return rows
