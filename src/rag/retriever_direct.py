

# # src/rag/retriever_direct.py

# import json
# from typing import List, Optional
# from src.db.supabase_client import supabase
# from src.utils.logger import get_logger
# from src.embeddings.embedder import embedding_model

# logger = get_logger(__name__)


# def retrieve_top_k(
#     query: str,
#     top_k: int = 10,
#     source_name: Optional[str] = None,
# ) -> List[dict]:
#     """
#     Direct Supabase query using embeddings + RPC match_documents.
#     Returns list of:
#        { id, content, page, source_name, similarity }
#     """

#     logger.info(f"[retriever_direct] Query={query!r} source={source_name}")

#     # Embed query
#     q_emb = embedding_model.embed_one(query)

#     # Build filter JSON
#     filter_json = {"source_name": source_name} if source_name else None

#     # RPC call
#     resp = supabase.rpc(
#         "match_documents",
#         {
#             "filter": json.dumps(filter_json) if filter_json else None,
#             "query_embedding": q_emb,
#         },
#     ).execute()

#     rows = resp.data or []
#     logger.info(f"[retriever_direct] Retrieved {len(rows)} rows (top_k={top_k})")

#     return rows[:top_k]

# src/rag/retriever_direct.py

from typing import List, Dict, Optional
import os

from src.utils.logger import get_logger
from src.db.supabase_client import get_supabase

logger = get_logger(__name__)

# ======================================================
# FEATURE FLAG (CRITICAL FOR CI)
# ======================================================
# Allows retriever to be disabled in CI / tests
RETRIEVER_ENABLED = os.getenv("RETRIEVER_ENABLED", "true").lower() == "true"


# ======================================================
# PUBLIC API
# ======================================================
def retrieve_top_k(
    query: str,
    top_k: int = 5,
    source: Optional[str] = None,
) -> List[Dict]:
    """
    Retrieve top-k document chunks from Supabase.

    CI-safe behavior:
    - If RETRIEVER_ENABLED=false → returns []
    - No Supabase access at import time
    - Supabase client created lazily
    """

    logger.info(f"[retriever_direct] query='{query}' source={source}")

    # --------------------------------------------------
    # CI / TEST GUARD
    # --------------------------------------------------
    if not RETRIEVER_ENABLED:
        logger.warning("[retriever_direct] Retriever disabled — returning empty list")
        return []

    # --------------------------------------------------
    # RUNTIME SUPABASE ACCESS (LAZY)
    # --------------------------------------------------
    supabase = get_supabase()

    try:
        q = (
            supabase
            .table("document_embeddings")
            .select(
                "content, source_name, page, chunk_index"
            )
            .limit(top_k)
        )

        if source:
            q = q.eq("source_name", source)

        response = q.execute()

        data = response.data or []

        logger.info(
            f"[retriever_direct] Retrieved {len(data)} chunks"
        )

        return data

    except Exception as e:
        logger.error(f"[retriever_direct] Supabase query failed: {e}")
        return []
