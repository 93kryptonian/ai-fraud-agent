

import re
import random
from typing import Dict, List
from statistics import mean

from src.monitoring.phoenix_client import run_phoenix_llm_judge
from src.embeddings.embedder import embedding_model


# ============================================================
# CONFIGURATION
# ============================================================

LOW_HEURISTIC_SKIP = 0.25     
MID_HEURISTIC_TRIGGER = 0.55  # Borderline → evaluate with LLM
RANDOM_SAMPLE_RATE = 0.03     
MAX_CONTEXT_CHARS = 6000

# Blending weights 
W_HEURISTIC = 0.45
W_EMBED = 0.35
W_NUMERIC = 0.10
W_LLM = 0.10


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def extract_numbers(text: str) -> List[float]:
    """Extract numeric values for consistency check."""
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    return [float(n) for n in nums]


def cosine_sim(v1, v2):
    import numpy as np
    v1 = np.array(v1)
    v2 = np.array(v2)
    if len(v1) != len(v2):
        return 0.0
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


# ============================================================
# SCORING COMPONENTS
# ============================================================

def heuristic_score(question: str, answer: str, chunks: List[Dict]) -> float:
    """Improved heuristic v2."""
    if not answer or "not provide enough information" in answer.lower():
        return 0.0

    ctx_len = sum(len(c["content"]) for c in chunks)
    ctx_score = min(ctx_len / 6000, 1.0)

    q_terms = set(question.lower().split())
    a_terms = set(answer.lower().split())
    overlap = len(q_terms & a_terms) / (len(q_terms) + 1)

    density = min(sum(len(c["content"]) for c in chunks) / 8000, 1.0)

    score = (0.5 * ctx_score) + (0.3 * overlap) + (0.2 * density)
    return round(score, 3)


def numeric_consistency(answer: str, chunks: List[Dict]) -> float:
    """Checks if numeric values in answer match numbers in context."""
    ans_nums = extract_numbers(answer)
    ctx_nums = []
    for c in chunks:
        ctx_nums.extend(extract_numbers(c["content"]))

    if not ans_nums or not ctx_nums:
        return 1.0  # No numbers → can't judge → neutral

    mismatches = sum(1 for n in ans_nums if n not in ctx_nums)
    ratio = mismatches / len(ans_nums)

    return round(1.0 - ratio, 3)


def embedding_similarity(question: str, answer: str, chunks: List[Dict]) -> float:
    """Embedding-based semantic similarity."""
    try:
        q_emb = embedding_model.embed(question)
        a_emb = embedding_model.embed(answer)

        # similarity question ↔ answer
        qa_sim = cosine_sim(q_emb, a_emb)

        # similarity answer ↔ each chunk
        chunk_sims = []
        for c in chunks:
            emb = embedding_model.embed(c["content"][:500])
            chunk_sims.append(cosine_sim(a_emb, emb))

        return round(mean([qa_sim] + chunk_sims), 3)
    except Exception:
        return 0.0


# ============================================================
# MAIN HYBRID SCORER
# ============================================================

def score_answer(question: str, answer: str, chunks: List[Dict], use_llm=False):
    """

    Return format :
        {"final_score": float}

    
    """

    # ——————————————
    # 1. HEURISTIC
    # ——————————————
    h_score = heuristic_score(question, answer, chunks)

    # Skip LLM for garbage output
    if h_score <= LOW_HEURISTIC_SKIP:
        return {"final_score": round(h_score, 3)}

    # ——————————————
    # 2. EMBEDDINGS
   # ——————————————
    embed_score = embedding_similarity(question, answer, chunks)

    # ——————————————
    # 3. NUMERIC CONSISTENCY
   # ——————————————
    num_score = numeric_consistency(answer, chunks)

    # ——————————————
    # 4. DECIDE IF LLM JUDGE NEEDED
   # ——————————————
    llm_score = None

    if use_llm:
        llm_needed = (
            h_score <= MID_HEURISTIC_TRIGGER or
            random.random() < RANDOM_SAMPLE_RATE
        )
        if llm_needed:
            context = "\n".join(c["content"][:800] for c in chunks)[:MAX_CONTEXT_CHARS]

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

    # ——————————————
    # 5. FINAL SCORE — weighted ensemble
    # ——————————————
    final = (
        W_HEURISTIC * h_score +
        W_EMBED * embed_score +
        W_NUMERIC * num_score
    )

    if llm_score is not None:
        final += W_LLM * llm_score

    return {"final_score": round(float(final), 3)}
