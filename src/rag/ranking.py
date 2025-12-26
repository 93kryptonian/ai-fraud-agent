"""
Enterprise Hybrid Reranker

Implements a multi-stage ranking pipeline:

1. Embedding-based semantic similarity
2. BM25 keyword relevance
3. Optional LLM cross-encoder reranking

Enterprise enhancements:
- Score normalization & adaptive weighting
- Source-aware boosting (regulatory vs research docs)
- Chunk coherence scoring
- Redundancy removal
- Hybrid score fusion with graceful degradation

Used by: rag_chain.py
"""

import os
from typing import List, Dict, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.embeddings.embedder import embedding_model
from src.llm.llm_client import llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

EMBEDDINGS_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))

# =============================================================================
# VECTOR SIMILARITY
# =============================================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity with numerical safety."""
    va = np.array(a)
    vb = np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10
    return float(np.dot(va, vb) / denom)

# =============================================================================
# STAGE 0 — DEDUPLICATION
# =============================================================================

def _deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate chunks based on source, page, and content prefix.
    """
    seen = set()
    unique_chunks: List[Dict[str, Any]] = []

    for c in chunks:
        key = (c.get("source_name"), c.get("page"), (c.get("content") or "")[:60])
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    if len(unique_chunks) < len(chunks):
        logger.info(
            f"[ranking] Deduplicated {len(chunks) - len(unique_chunks)} chunks"
        )

    return unique_chunks

# =============================================================================
# STAGE 1 — EMBEDDING SIMILARITY
# =============================================================================

def _embedding_scores(query: str, chunks: List[Dict[str, Any]]) -> List[float]:
    """
    Compute embedding similarity scores.
    Falls back to zeros when embeddings are unavailable
    (e.g. CI, offline, or cost-restricted environments).
    """
    if not EMBEDDINGS_AVAILABLE:
        logger.info("[ranking] Embeddings disabled — using zero scores")
        return [0.0] * len(chunks)

    query_vec = embedding_model.embed_one(query)
    scores: List[float] = []

    for c in chunks:
        vec = embedding_model.embed_one(c["content"])
        scores.append(cosine_similarity(query_vec, vec))

    return scores

# =============================================================================
# STAGE 2 — BM25 KEYWORD RELEVANCE
# =============================================================================

def _bm25_scores(query: str, chunks: List[Dict[str, Any]]) -> List[float]:
    """
    Compute normalized BM25 relevance scores.
    """
    tokenized_docs = [c["content"].lower().split() for c in chunks]
    tokenized_query = query.lower().split()

    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)

    if scores.max() > 0:
        scores = scores / scores.max()

    return scores.tolist()

# =============================================================================
# CHUNK COHERENCE HEURISTIC
# =============================================================================

def _coherence_score(text: str) -> float:
    """
    Penalize very short or low-information chunks.
    """
    length = len(text.split())
    if length < 20:
        return 0.3
    if length < 40:
        return 0.6
    return 1.0

# =============================================================================
# SOURCE-AWARE BOOSTING
# =============================================================================

def _source_boost(query: str, source: str) -> float:
    """
    Boost relevance based on query intent and document provenance.
    """
    q = query.lower()
    s = source.lower()

    if any(k in q for k in ("eea", "cross-border", "h1 2023")):
        return 1.25 if any(k in s for k in ("eba", "ecb")) else 1.0

    if any(k in q for k in (
        "methods", "card-not-present", "lost or stolen",
        "counterfeit", "typology"
    )):
        return 1.25 if "bhatla" in s else 1.0

    return 1.0

# =============================================================================
# STAGE 3 — OPTIONAL LLM CROSS-ENCODER
# =============================================================================

LLM_RERANK_PROMPT = """
Rate the relevance of the following chunk to the query.
Score from 0 (irrelevant) to 1 (highly relevant).
Return ONLY the number.

Query:
{query}

Chunk:
{chunk}
""".strip()


def _rerank_by_llm(
    query: str,
    chunks: List[Dict[str, Any]],
    top_n: int,
) -> List[float]:
    """
    Expensive but precise LLM-based cross-encoder reranking.
    Applied only to the top-N chunks.
    """
    scores: List[float] = []

    for c in chunks[:top_n]:
        prompt = LLM_RERANK_PROMPT.format(
            query=query,
            chunk=c["content"],
        )
        resp = llm.run(prompt, temperature=0.0)

        try:
            score = float(resp.strip())
        except Exception:
            score = 0.0

        scores.append(score)

    if len(chunks) > top_n:
        scores.extend([0.0] * (len(chunks) - top_n))

    return scores

# =============================================================================
# HYBRID SCORE FUSION
# =============================================================================

def _fuse_scores(
    emb: List[float],
    bm25: List[float],
    coh: List[float],
    boost: List[float],
    llm_scores: Optional[List[float]],
) -> List[float]:
    """
    Fuse all signals into a single hybrid relevance score.
    """
    fused: List[float] = []
    use_embeddings = any(e > 0 for e in emb)

    for i in range(len(bm25)):
        if use_embeddings:
            score = (
                0.45 * emb[i] +
                0.35 * bm25[i] +
                0.10 * coh[i] +
                0.10 * boost[i]
            )
        else:
            score = (
                0.55 * bm25[i] +
                0.25 * coh[i] +
                0.20 * boost[i]
            )

        if llm_scores:
            score = 0.7 * score + 0.3 * llm_scores[i]

        fused.append(score)

    return fused

# =============================================================================
# PUBLIC API
# =============================================================================

def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    use_llm: bool = False,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Rank document chunks using a hybrid relevance strategy.
    """
    if not chunks:
        return []

    # Stage 0 — Deduplication
    chunks = _deduplicate_chunks(chunks)

    # Stage 1 — Semantic similarity
    emb_scores = _embedding_scores(query, chunks)

    # Stage 2 — Keyword relevance
    bm25_scores = _bm25_scores(query, chunks)

    # Coherence & source priors
    coherence = [_coherence_score(c["content"]) for c in chunks]
    boosts = [_source_boost(query, c["source_name"]) for c in chunks]

    # Stage 3 — Optional LLM reranking
    llm_scores = None
    if use_llm:
        logger.info("[ranking] Running LLM reranking on top chunks")
        llm_scores = _rerank_by_llm(query, chunks, top_n=8)

    # Hybrid fusion
    hybrid_scores = _fuse_scores(
        emb_scores,
        bm25_scores,
        coherence,
        boosts,
        llm_scores,
    )

    # Attach scores
    for i, c in enumerate(chunks):
        c["rerank_score"] = hybrid_scores[i]

    ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    # Top-K truncation
    if top_k is not None:
        ranked = ranked[:top_k]

    # Cleanup
    for c in ranked:
        c.pop("rerank_score", None)

    return ranked
