

# src/rag/retriever_direct.py

import json
from typing import List, Optional
from src.db.supabase_client import supabase
from src.utils.logger import get_logger
from src.embeddings.embedder import embedding_model

logger = get_logger(__name__)


def retrieve_top_k(
    query: str,
    top_k: int = 10,
    source_name: Optional[str] = None,
) -> List[dict]:
    """
    Direct Supabase query using embeddings + RPC match_documents.
    Returns list of:
       { id, content, page, source_name, similarity }
    """

    logger.info(f"[retriever_direct] Query={query!r} source={source_name}")

    # Embed query
    q_emb = embedding_model.embed_one(query)

    # Build filter JSON
    filter_json = {"source_name": source_name} if source_name else None

    # RPC call
    resp = supabase.rpc(
        "match_documents",
        {
            "filter": json.dumps(filter_json) if filter_json else None,
            "query_embedding": q_emb,
        },
    ).execute()

    rows = resp.data or []
    logger.info(f"[retriever_direct] Retrieved {len(rows)} rows (top_k={top_k})")

    return rows[:top_k]

