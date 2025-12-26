# src/orchestrator.py
"""
Enterprise AI Orchestrator.

Responsibilities:
- Sanitize user input
- Detect language and intent
- Route queries to Analytics or RAG pipelines
- Apply scoring, confidence gating, and fallbacks
- Produce safe, multilingual, explainable outputs

This module coordinates workflows but contains no domain logic itself.
"""

import json
import re
from typing import Any, Dict, Tuple

from src.llm.llm_client import llm
from src.llm.prompts import INTENT_CLASSIFICATION_PROMPT
from src.llm.scoring import score_answer

from src.analytics.fraud_analytics import run_analytics
from src.rag.rag_chain import run_rag
from src.rag.question_rewrite import (
    detect_language,
    process_query,
    translate_en_to_id,
)
from src.rag.insight_layer import generate_insight
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# INPUT SANITIZATION
# =============================================================================

def sanitize_input(raw: str) -> str:
    """
    Clean UI artifacts and normalize user input.
    """
    if not raw:
        return ""

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""

    query = lines[-1]

    query = re.sub(r"^(You:|User:)\s*", "", query, flags=re.I)
    query = re.sub(r"[\x00-\x1f]+", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    if re.search(r"\b(who|what|when|where|why|how|which)\b", query, re.I) and not query.endswith("?"):
        query += "?"

    return query

# =============================================================================
# INTENT DETECTION
# =============================================================================

def detect_intent_heuristic(query: str) -> Tuple[str, float]:
    """
    Fast heuristic intent classifier.

    Returns:
        intent: "rag" | "analytics"
        confidence: [0,1]
    """
    q = query.lower()

    timeseries_kw = (
        "daily", "monthly", "harian", "bulanan",
        "yearly", "trend", "over time",
        "last 12 months", "last 24 months",
        "fluctuate", "increase", "decrease",
        "per day", "per month", "per year",
        "timeline", "evolution", "periode",
    )

    if any(k in q for k in timeseries_kw):
        return "analytics", 0.95

    conceptual_kw = (
        "what are", "explain", "describe", "definition",
        "jelaskan", "apa itu",
        "merchant", "merchants", "mcc", "kategori merchant",
        "eea", "eba", "ecb", "psd2", "cross-border",
    )

    if any(k in q for k in conceptual_kw):
        return "rag", 0.90

    return "rag", 0.40


def detect_intent_llm(query: str) -> Tuple[str, str]:
    """
    LLM-based intent & language classification.

    Returns:
        intent: "rag" | "analytics" | "reject"
        lang: "en" | "id"
    """
    prompt = INTENT_CLASSIFICATION_PROMPT.format(q=query)
    resp = llm.run(prompt, temperature=0.0)

    try:
        data = json.loads(resp)
        intent = data.get("intent", "rag")
        lang = data.get("language", "en")
    except Exception as e:
        logger.error(
            f"[intent_llm] Failed to parse response: {e} | resp={resp!r}"
        )
        intent, lang = "rag", "en"

    logger.info(f"[intent_llm] intent={intent} lang={lang}")
    return intent, lang


def detect_intent(query: str, detected_lang: str) -> Tuple[str, str]:
    """
    Hybrid intent detection strategy.

    1. Heuristic detection
    2. High confidence → accept
    3. Low confidence → defer to LLM
    """
    intent_h, conf = detect_intent_heuristic(query)
    logger.info(f"[intent] heuristic intent={intent_h} conf={conf:.2f}")

    if conf >= 0.80:
        return intent_h, detected_lang

    intent_l, lang_l = detect_intent_llm(query)
    return intent_l, lang_l or detected_lang

# =============================================================================
# MAIN ORCHESTRATION PIPELINE
# =============================================================================

def run_query(raw_query: str, detected_lang: str = None) -> Dict[str, Any]:
    """
    Main entrypoint for all user queries.
    """
    # ------------------------------------------------------------------
    # 1. Input sanitization
    # ------------------------------------------------------------------
    query = sanitize_input(raw_query)
    if not query:
        return {
            "query": raw_query,
            "intent": None,
            "result": None,
            "error": "Empty query.",
        }

    # ------------------------------------------------------------------
    # 2. Language detection
    # ------------------------------------------------------------------
    try:
        user_lang = detect_language(query)
    except Exception:
        user_lang = "en"

    # ------------------------------------------------------------------
    # 3. Intent detection
    # ------------------------------------------------------------------
    intent, user_lang = detect_intent(query, user_lang)
    logger.info(
        f"[orchestrator] intent={intent} | lang={user_lang} | query={query!r}"
    )

    # ------------------------------------------------------------------
    # 4. Reject handling
    # ------------------------------------------------------------------
    if intent == "reject":
        message = (
            "Maaf, permintaan ini berada di luar domain atau berpotensi berbahaya."
            if user_lang == "id"
            else "Sorry, this request is out of scope or potentially unsafe."
        )

        return {
            "query": query,
            "intent": "reject",
            "error": None,
            "result": {"type": "reject", "message": message},
        }

    # ------------------------------------------------------------------
    # 5. Analytics pipeline
    # ------------------------------------------------------------------
    if intent == "analytics":
        try:
            analytics_res = run_analytics(query, user_lang)
            return {
                "query": query,
                "intent": "analytics",
                "error": None,
                "result": {"type": "analytics", "analytics": analytics_res},
            }
        except Exception as e:
            logger.error("[orchestrator] analytics failed", exc_info=True)
            return {
                "query": query,
                "intent": "analytics",
                "result": None,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # 6. RAG pipeline
    # ------------------------------------------------------------------
    try:
        rewritten_query, meta = process_query(query)
        logger.info(
            f"[orchestrator] rewritten_query={rewritten_query!r} | meta={meta}"
        )

        final_query_en = meta["final_query_en"]
        user_lang = meta.get("lang") or user_lang

        rag_res = run_rag(query_en=final_query_en, user_lang=user_lang)

        answer_en = rag_res.get("answer", "")
        chunks = rag_res.get("chunks", [])
        context_text = rag_res.get("context_text", "")

        scoring = score_answer(
            question=final_query_en,
            answer=answer_en,
            chunks=chunks,
            use_llm=False,
        )

        final_score = scoring.get("final_score", 0.0)
        logger.info(f"[orchestrator] RAG score={final_score:.3f}")

        # Low-confidence fallback
        if final_score < 0.12:
            fallback_msg = (
                "Maaf, dokumen yang tersedia tidak cukup untuk menjawab pertanyaan."
                if user_lang == "id"
                else "Sorry, the documents do not contain enough information."
            )

            rag_res.update(
                {"answer": fallback_msg, "insight": None, "score": scoring}
            )

            return {
                "query": query,
                "intent": "rag",
                "error": None,
                "result": {"type": "rag", **rag_res},
            }

        insight_en = generate_insight(
            answer_en, context_text, user_lang="en"
        )

        if user_lang == "id":
            answer = translate_en_to_id(answer_en)
            insight = (
                translate_en_to_id(insight_en) if insight_en else None
            )
        else:
            answer = answer_en
            insight = insight_en

        rag_res.update(
            {"answer": answer, "insight": insight, "score": scoring}
        )

        return {
            "query": query,
            "intent": "rag",
            "error": None,
            "result": {"type": "rag", **rag_res},
        }

    except Exception as e:
        logger.error("[orchestrator] RAG pipeline failed", exc_info=True)
        return {
            "query": query,
            "intent": "rag",
            "result": None,
            "error": str(e),
        }
