"""
Hybrid answer scoring for an Enterprise Fraud Intelligence System.

This module implements a multi-signal evaluation pipeline combining:
1. Heuristic relevance & density checks
2. Embedding-based semantic similarity
3. Numeric consistency verification
4. Optional LLM-as-judge (Phoenix / Arize-style)

Design goals:
- Avoid expensive LLM judging for low-quality answers
- Provide deterministic, explainable scores
- Gracefully degrade when embeddings or judges fail
"""

import re
import random
from typing import Dict, List
from statistics import mean

import numpy as np

from src.monitoring.phoenix_client import run_phoenix_llm_judge
from src.embeddings.embedder import embedding_model

# =============================================================================
# CONFIGURATION
# =============================================================================

LOW_HEURISTIC_SKIP = 0.25        # Below this → immediately reject
MID_HEURISTIC_TRIGGER = 0.55     # Borderline → consider LLM judge
RANDOM_SAMPLE_RATE = 0.03        # Periodic sampling for calibration
MAX_CONTEXT_CHARS = 6000

# Blending weights (must sum ≤ 1.0 without LLM)
W_HEURISTIC = 0.45
W_EMBED = 0.35
W_NUMERIC = 0.10
W_LLM = 0.10

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_numbers(text: str) -> List[float]:
    """
    Extract numeric values from text for consistency checking.
    """
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    return [float(n) for n in nums]


def cosine_sim(v1, v2) -> float:
    """
    Safe cosine similarity with zero-vector protection.
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    a = np.array(v1)
    b = np.array(v2)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)

# =============================================================================
# SCORING COMPONENTS
# =============================================================================

def heuristic_score(question: str, answer: str, chunks: List[Dict]) -> float:
    """
    Fast heuristic score based on:
    - context coverage
    - lexical overlap
    - information density
    """
    if not answer or "not provide enough information" in answer.lower():
        return 0.0

    total_ctx_len = sum(len(c["content"]) for c in chunks)
    ctx_score = min(total_ctx_len / 6000, 1.0)

    q_terms = set(question.lower().split())
    a_terms = set(answer.lower().split())
    overlap = len(q_terms & a_terms) / (len(q_terms) + 1)

    density = min(total_ctx_len / 8000, 1.0)

    score = (0.5 * ctx_score) + (0.3 * overlap) + (0.2 * density)
    return round(score, 3)


def numeric_consistency(answer: str, chunks: List[Dict]) -> float:
    """
    Verify whether numeric values in the answer
    are supported by numbers in the context.
    """
    ans_nums = extract_numbers(answer)
    ctx_nums: List[float] = []

    for c in chunks:
        ctx_nums.extend(extract_numbers(c["content"]))

    if not ans_nums or not ctx_nums:
        return 1.0  # Neutral when no numeric comparison is possible

    mismatches = sum(1 for n in ans_nums if n not in ctx_nums)
    ratio = mismatches / len(ans_nums)

    return round(1.0 - ratio, 3)


def embedding_similarity(question: str, answer: str, chunks: List[Dict]) -> float:
    """
    Semantic similarity using embeddings:
    - question ↔ answer
    - answer ↔ each context chunk
    """
    try:
        q_emb = embedding_model.embed(question)
        a_emb = embedding_model.embed(answer)

        qa_sim = cosine_sim(q_emb, a_emb)

        chunk_sims = []
        for c in chunks:
            emb = embedding_model.embed(c["content"][:500])
            chunk_sims.append(cosine_sim(a_emb, emb))

        return round(mean([qa_sim] + chunk_sims), 3)

    except Exception:
        return 0.0

# =============================================================================
# MAIN HYBRID SCORER
# =============================================================================

def score_answer(
    question: str,
    answer: str,
    chunks: List[Dict],
    use_llm: bool = False,
) -> Dict[str, float]:
    """
    Score a generated answer using a weighted ensemble.

    Returns:
        { "final_score": float }
    """

    # ---------------------------------------------------------
    # 1. Heuristic gate (cheap, deterministic)
    # ---------------------------------------------------------
    h_score = heuristic_score(question, answer, chunks)

    # Early exit for clearly low-quality answers
    if h_score <= LOW_HEURISTIC_SKIP:
        return {"final_score": round(h_score, 3)}

    # ---------------------------------------------------------
    # 2. Embedding similarity
    # ---------------------------------------------------------
    embed_score = embedding_similarity(question, answer, chunks)

    # ---------------------------------------------------------
    # 3. Numeric consistency
    # ---------------------------------------------------------
    num_score = numeric_consistency(answer, chunks)

    # ---------------------------------------------------------
    # 4. Optional LLM-as-judge
    # ---------------------------------------------------------
    llm_score = None

    if use_llm:
        llm_needed = (
            h_score <= MID_HEURISTIC_TRIGGER or
            random.random() < RANDOM_SAMPLE_RATE
        )

        if llm_needed:
            context = "\n".join(
                c["content"][:800] for c in chunks
            )[:MAX_CONTEXT_CHARS]

            judge = run_phoenix_llm_judge(
                question=question,
                answer=answer,
                context=context,
            )

            if judge:
                llm_vals = [
                    judge.get("relevance", 0),
                    judge.get("groundedness", 0),
                    judge.get("correctness", 0),
                    judge.get("coherence", 0),
                ]
                llm_score = mean(llm_vals)

    # ---------------------------------------------------------
    # 5. Final weighted fusion
    # ---------------------------------------------------------
    final = (
        W_HEURISTIC * h_score +
        W_EMBED * embed_score +
        W_NUMERIC * num_score
    )

    if llm_score is not None:
        final += W_LLM * llm_score

    return {"final_score": round(float(final), 3)}
