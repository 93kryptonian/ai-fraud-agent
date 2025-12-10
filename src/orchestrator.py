# src/orchestrator.py
"""
Enterprise Orchestrator:
 - sanitize input
 - detect language
 - multilingual query handling
 - safe question rewrite
 - analytics or RAG routing
 - scoring + insight generation
 - low-confidence fallback
"""

import re
from typing import Any, Dict
import json
from src.llm.llm_client import llm
from src.llm.prompts import INTENT_CLASSIFICATION_PROMPT 


from src.rag.rag_chain import run_rag
from src.analytics.fraud_analytics import run_analytics
from src.rag.question_rewrite import (
    detect_language,
    process_query,
    translate_en_to_id,
)
from src.llm.scoring import score_answer
from src.utils.logger import get_logger
from src.rag.insight_layer import generate_insight

logger = get_logger(__name__)


# =====================================================================
# Input Sanitizer
# =====================================================================

def sanitize_input(raw: str) -> str:
    """Clean UI history noise & enforce question format."""
    if not raw:
        return ""

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""

    last = lines[-1]

    # Remove UI prefixes
    last = re.sub(r"^(You:|User:)\s*", "", last, flags=re.I)

    # Strip control chars
    last = re.sub(r"[\x00-\x1f]+", " ", last)

    # Collapse spaces
    last = re.sub(r"\s+", " ", last).strip()

    # Ensure question mark when relevant
    if re.search(r"\b(who|what|when|where|why|how|which)\b", last, re.I) and not last.endswith("?"):
        last += "?"

    return last


# =====================================================================
# Intent Classifier 
# =====================================================================

def detect_intent_heuristic(query: str) -> (str, float):
    """
    Heuristic intent detector.
    Returns:
        intent: "rag" or "analytics"
        confidence: float in [0,1]
    """
    q = query.lower()

    timeseries_kw = [
        "daily", "monthly", "harian", "bulanan",
        "yearly", "trend", "over time",
        "last 12 months", "last 24 months",
        "fluctuate", "increase", "decrease",
        "per day", "per month", "per year",
        "timeline", "evolution", "periode"
    ]
    if any(k in q for k in timeseries_kw):
        return "analytics", 0.95

    regulatory_kw = ["eea", "eba", "ecb", "psd2", "cross-border", "cross border"]
    merchant_kw = ["merchant", "merchants", "mcc", "kategori merchant"]
    conceptual_kw = ["what are", "explain", "describe", "definition", "jelaskan", "apa itu"]

    if any(k in q for k in regulatory_kw + merchant_kw + conceptual_kw):
        return "rag", 0.90

    # Default bias: RAG, but low confidence
    return "rag", 0.40

def detect_intent_llm(query: str) -> (str, str):
    """
    LLM-based intent classifier using INTENT_CLASSIFICATION_PROMPT.

    Returns:
        intent: "rag" | "analytics" | "reject"
        language: "en" | "id"
    """
    prompt = INTENT_CLASSIFICATION_PROMPT.format(q=query)
    resp = llm.run(prompt, temperature=0.0)

    try:
        data = json.loads(resp)
        intent = data.get("intent", "rag")
        lang = data.get("language", "en")
    except Exception as e:
        logger.error(f"[intent_llm] Failed to parse LLM intent response: {e} | resp={resp!r}")
        intent, lang = "rag", "en"

    logger.info(f"[intent_llm] intent={intent} lang={lang}")
    return intent, lang

def detect_intent_hybrid(query: str, current_lang: str) -> (str, str):
    """
    Hybrid intent detection:
      1) Run heuristic, get (intent, confidence)
      2) If confidence >= 0.80 → use heuristic, keep current_lang
      3) Else → fallback to LLM classifier (intent + language)

    Returns:
        intent: "rag" | "analytics" | "reject"
        lang:   "en" | "id"
    """
    intent_h, conf = detect_intent_heuristic(query)
    logger.info(f"[intent] heuristic intent={intent_h} conf={conf:.2f}")

    if conf >= 0.80:
        # High confidence → no need to call LLM
        return intent_h, current_lang

    # Low confidence → ask LLM
    intent_l, lang_l = detect_intent_llm(query)
    logger.info(f"[intent] LLM override intent={intent_l} lang={lang_l}")
    return intent_l, lang_l or current_lang

# =====================================================================
# MAIN ORCHESTRATOR FUNCTION
# =====================================================================

def run_query(raw_query: str, detected_lang: str = None) -> Dict[str, Any]:

    # ---------------------------------------------------------
    # 1. Clean input
    # ---------------------------------------------------------
    q = sanitize_input(raw_query)
    if not q:
        return {
            "query": raw_query,
            "intent": None,
            "result": None,
            "error": "Empty query."
        }

    # ---------------------------------------------------------
    # 2. Detect language
    # ---------------------------------------------------------
    try:
        user_lang = detect_language(q)
    except Exception:
        user_lang = "en"

    # ---------------------------------------------------------
    # 3. Detect intent (analytics or rag)
    # ---------------------------------------------------------
    intent, user_lang = detect_intent_hybrid(q, user_lang)
    logger.info(f"[orchestrator] intent={intent} | lang={user_lang} | query={q!r}")
    # ---------------------------------------------------------
    # Reject (Optional)
    # ---------------------------------------------------------
    if intent == "reject":
        logger.warning(f"[orchestrator] Rejecting unsafe/out-of-domain query: {q}")
        if user_lang == "id":
            reject_msg = (
                "Maaf, permintaan ini tidak dapat diproses karena berada di luar domain "
                "atau berpotensi berbahaya."
            )
        else:
            reject_msg = (
                "Sorry, this request cannot be processed because it is out of scope "
                "or potentially unsafe."
            )

        return {
            "query": q,
            "intent": "reject",
            "error": None,
            "result": {
                "type": "reject",
                "message": reject_msg
            }
        }


    # ---------------------------------------------------------------------
    # ANALYTICS PIPELINE — no rewriting, no language rewriting needed
    # ---------------------------------------------------------------------
    if intent == "analytics":
        try:
            analytics_res = run_analytics(q, user_lang)
            return {
                "query": q,
                "intent": "analytics",
                "error": None,
                "result": {
                    "type": "analytics",
                    "analytics": analytics_res
                }
            }
        except Exception as e:
            logger.error(f"[orchestrator] analytics error: {e}", exc_info=True)
            return {
                "query": q,
                "intent": "analytics",
                "result": None,
                "error": str(e)
            }

    # ---------------------------------------------------------------------
    # RAG PIPELINE
    # ---------------------------------------------------------------------
    try:
        # Rewrite & translate 
        rewritten_query, meta = process_query(q)
        logger.info(f"[orchestrator] rewritten_query={rewritten_query!r} | meta={meta}")

        # English question for retrieval
        final_query_en = meta["final_query_en"]
        user_lang = meta["lang"] or initial_lang

        # ---------------------------------------------------------
        # Run the RAG pipeline
        # ---------------------------------------------------------
        rag_res = run_rag(query_en=final_query_en, user_lang=user_lang)
        answer_en = rag_res.get("answer", "")
        chunks = rag_res.get("chunks", [])
        context_text = rag_res.get("context_text", "")

        # ---------------------------------------------------------
        # Enterprise scoring 
        # ---------------------------------------------------------
        scoring = score_answer(
            question=final_query_en,
            answer=answer_en,
            chunks=chunks,
            use_llm=False
        )

        final_score = scoring.get("final_score", 0.0)
        logger.info(f"[orchestrator] RAG score={final_score:.3f}")

        # ---------------------------------------------------------
        # Low-confidence fallback
        # ---------------------------------------------------------
        if final_score < 0.12:
            logger.warning("[orchestrator] Low confidence → fallback answer.")

            if user_lang == "id":
                fallback_msg = (
                    "Maaf, dokumen yang tersedia tidak memiliki informasi yang cukup "
                    "untuk menjawab pertanyaan tersebut."
                )
            else:
                fallback_msg = (
                    "Sorry, the available documents do not provide enough information "
                    "to answer your question."
                )

            rag_res.update({
                "answer": fallback_msg,
                "insight": None,
                "score": scoring
            })

            return {
                "query": q,
                "intent": "rag",
                "error": None,
                "result": {
                    "type": "rag",
                    **rag_res
                }
            }

        # ---------------------------------------------------------
        # High-confidence → generate insight
        # ---------------------------------------------------------
        insight_en = generate_insight(answer_en, context_text, user_lang="en")

        # ---------------------------------------------------------
        # Multilingual output formatting
        # ---------------------------------------------------------
        if user_lang == "id":
            answer = translate_en_to_id(answer_en)
            insight = translate_en_to_id(insight_en) if insight_en else None
        else:
            answer = answer_en
            insight = insight_en

        rag_res.update({
            "answer": answer,
            "insight": insight,
            "score": scoring
        })

        return {
            "query": q,
            "intent": "rag",
            "error": None,
            "result": {
                "type": "rag",
                **rag_res
            }
        }

    except Exception as e:
        logger.error(f"[orchestrator] rag error: {e}", exc_info=True)
        return {
            "query": q,
            "intent": "rag",
            "result": None,
            "error": str(e)
        }
