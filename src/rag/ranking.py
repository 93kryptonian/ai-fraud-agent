"""
Enterprise Hybrid Reranker 

This module implements a 3-stage ranking pipeline:

    1. Embedding-based similarity 
    2. BM25 relevance (keyword overlap)
    3. (optional) LLM cross-encoder reranking

Additional enterprise enhancements:
 - score normalization
 - document-aware boosting (Bhatla vs EBA/ECB)
 - chunk coherence scoring (penalize low-information chunks)
 - redundancy removal (duplicate / overlapping chunks)
 - final hybrid score fusion

This is used by rag_chain.py.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter

from rank_bm25 import BM25Okapi  

from src.embeddings.embedder import embedding_model
from src.llm.llm_client import llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ======================================================
# Utility: cosine similarity
# ======================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b) / denom)


# ======================================================
# Stage 0 — Deduplication
# ======================================================

def _deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove redundant or duplicate chunks.
    """
    seen = set()
    cleaned = []

    for c in chunks:
        key = (c["source_name"], c["page"], c["content"][:60])
        if key not in seen:
            cleaned.append(c)
            seen.add(key)

    if len(cleaned) < len(chunks):
        logger.info(f"[ranking] Dedup removed {len(chunks) - len(cleaned)} chunks")

    return cleaned


# ======================================================
# Stage 1 — Embedding similarity
# ======================================================

def _embedding_scores(query: str, chunks: List[Dict[str, Any]]) -> List[float]:
    query_vec = embedding_model.embed_one(query)
    scores = []

    for c in chunks:
        vec = embedding_model.embed_one(c["content"])
        sim = cosine_similarity(query_vec, vec)
        scores.append(sim)

    return scores


# ======================================================
# Stage 2 — BM25 keyword relevance
# ======================================================

def _bm25_scores(query: str, chunks: List[Dict[str, Any]]) -> List[float]:
    tokenized_docs = [c["content"].lower().split() for c in chunks]
    tokenized_query = query.lower().split()

    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)

    # Normalize BM25 scores
    if scores.max() > 0:
        scores = scores / scores.max()

    return scores.tolist()


# ======================================================
# Chunk coherence scoring
# ======================================================

def _coherence_score(text: str) -> float:
    """
    Heuristic: penalize extremely short or noisy chunks.
    """
    length = len(text.split())
    if length < 20:
        return 0.3
    if length < 40:
        return 0.6
    return 1.0


# ======================================================
# Source-aware boosting
# ======================================================

def _source_boost(query: str, source: str) -> float:
    q = query.lower()
    s = source.lower()

    if "eea" in q or "cross-border" in q or "h1 2023" in q:
        return 1.25 if "eba" in s or "ecb" in s else 1.0

    if any(k in q for k in ["methods", "card-not-present", "lost or stolen", "counterfeit", "typology"]):
        return 1.25 if "bhatla" in s else 1.0

    return 1.0


# ======================================================
# Stage 3 — LLM Cross-encoder reranking (optional)
# ======================================================

LLM_RERANK_PROMPT = """
Rate the relevance of the following chunk to the query.
Score from 0 (irrelevant) to 1 (highly relevant).
Return ONLY the number.

Query:
{query}

Chunk:
{chunk}
"""

def _rerank_by_llm(query: str, chunks: List[Dict[str, Any]], top_n: int) -> List[float]:
    scores = []

    for c in chunks[:top_n]:
        px = LLM_RERANK_PROMPT.format(query=query, chunk=c["content"])
        resp = llm.run(px, temperature=0.0)

        try:
            s = float(resp.strip())
        except:
            s = 0.0

        scores.append(s)

    # pad scores for remaining chunks
    if len(chunks) > top_n:
        scores.extend([0.0] * (len(chunks) - top_n))

    return scores


# ======================================================
# Hybrid Fusion
# ======================================================

def _fuse_scores(emb: List[float], bm25: List[float], coh: List[float], boost: List[float],
                 llm_scores: Optional[List[float]]) -> List[float]:
    fused = []

    for i in range(len(emb)):
        score = (
            0.45 * emb[i] +
            0.35 * bm25[i] +
            0.10 * coh[i] +
            0.10 * boost[i]
        )

        if llm_scores:
            score = 0.7 * score + 0.3 * llm_scores[i]

        fused.append(score)

    return fused


# ======================================================
# MAIN API — used by rag_chain.py
# ======================================================

def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    use_llm: bool = False,
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:

    if not chunks:
        return []

    # Step 0 — De-duplicate
    chunks = _deduplicate_chunks(chunks)

    # Step 1 — Emb sim
    emb = _embedding_scores(query, chunks)

    # Step 2 — BM25
    bm25 = _bm25_scores(query, chunks)

    # Coherence
    coh = [_coherence_score(c["content"]) for c in chunks]

    # Source boosts
    boost = [_source_boost(query, c["source_name"]) for c in chunks]

    # Step 3 (optional) — LLM cross-encoder
    llm_scores = None
    if use_llm:
        logger.info("[ranking] Running expensive LLM reranking on top 8 chunks...")
        llm_scores = _rerank_by_llm(query, chunks, top_n=8)

    # Hybrid fusion
    hybrid = _fuse_scores(emb, bm25, coh, boost, llm_scores)

    # Attach & sort
    for i, c in enumerate(chunks):
        c["rerank_score"] = hybrid[i]

    ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    # Top-K cut
    if top_k is not None:
        ranked = ranked[:top_k]

    # Clean internal fields
    for c in ranked:
        c.pop("rerank_score", None)

    return ranked
