# src/rag/rag_chain.py
"""
Retrieval-Augmented Generation (RAG) pipeline.

Responsibilities:
- Retrieve relevant document chunks
- Construct bounded, traceable context
- Build safe, grounded prompts
- Handle special inference modes when chunk-based RAG is insufficient

Design principles:
- Context-first, not LLM-first
- Hard character limits to prevent overflow
- Explicit fallback instructions
- Special inference modes are isolated and intentional
"""

import re
from typing import Dict, Any, List

from src.rag.retriever_direct import retrieve_top_k
from src.llm.llm_client import llm
from src.db.supabase_client import DB
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# CONTEXT BUILDERS
# =============================================================================

def build_context(chunks: List[dict], max_chars: int = 12_000) -> str:
    """
    Build a bounded context string from retrieved chunks.
    """
    context_blocks: List[str] = []
    total_chars = 0

    for ch in chunks:
        text = (ch.get("content") or "").strip()
        block = (
            f"[source={ch.get('source_name')} page={ch.get('page')}]\n"
            f"{text}\n\n"
        )

        if total_chars + len(block) > max_chars:
            break

        context_blocks.append(block)
        total_chars += len(block)

    context = "".join(context_blocks)
    logger.info(
        f"[RAG] Context built | chars={len(context)} | chunks={len(context_blocks)}"
    )
    return context


def build_citations(chunks: List[dict], max_preview_chars: int = 180) -> List[dict]:
    """
    Build lightweight citation metadata for UI display.
    """
    citations: List[dict] = []

    for ch in chunks:
        citations.append({
            "source": ch.get("source_name", "Unknown"),
            "page": ch.get("page", "N/A"),
            "preview": (ch.get("content") or "")[:max_preview_chars],
        })

    return citations

# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def build_prompt(question_en: str, context_text: str, user_lang: str) -> str:
    """
    Build the final grounded RAG prompt.
    """
    language_instruction = (
        "Respond in Indonesian."
        if user_lang == "id"
        else "Respond in English."
    )

    return f"""
You are an expert regulatory fraud intelligence assistant.

{language_instruction}

Use ONLY the information from the context below.
If the answer cannot be found, use the fallback answer exactly as written.

User Question (English):
{question_en}

Context:
{context_text}

Fallback answer:
"Sorry, the available documents do not provide enough information to answer your question."
""".strip()

# =============================================================================
# MERCHANT INFERENCE MODE (SPECIAL CASE)
# =============================================================================

MERCHANT_Q_PATTERNS = (
    "which merchants",
    "merchant categories",
    "highest incidence of fraudulent transactions",
    "highest fraud incidence",
    "which merchant categories exhibit",
)


def is_merchant_incidence_question(text: str) -> bool:
    """
    Detect questions that require holistic merchant-level inference.
    """
    t = text.lower()
    return any(p in t for p in MERCHANT_Q_PATTERNS)


def merchant_inference_mode(
    query_en: str,
    db_client,
    llm_client,
    user_lang: str = "en",
) -> Dict[str, Any]:
    """
    Holistic inference mode for merchant fraud questions.

    This bypasses top-k retrieval and instead loads all
    merchant-related pages to enable document-level reasoning.
    """
    keyword_filter = (
        "merchant",
        "merchants",
        "merchant fraud",
        "merchant related",
    )

    # ---------------------------------------------------------
    # Step 1 — Load all merchant-related pages
    # ---------------------------------------------------------
    sql = """
        SELECT d.id, d.content, d.page, d.source_name
        FROM documents d
        WHERE LOWER(d.content) ~ ANY(%s)
        ORDER BY d.page ASC;
    """

    patterns = [f".*{re.escape(k)}.*" for k in keyword_filter]
    rows = db_client.sql(sql, (patterns,))

    if not rows:
        return {
            "type": "rag",
            "answer": None,
            "context_text": "",
            "chunks": [],
            "citations": [],
            "inference_used": True,
        }

    # ---------------------------------------------------------
    # Step 2 — Build long-range context
    # ---------------------------------------------------------
    full_context = ""
    max_chars = 15_000
    total_chars = 0

    chunks: List[dict] = []

    for r in rows:
        snippet = (
            f"[Page {r['page']} - {r['source_name']}]\n"
            f"{r['content']}\n\n"
        )
        total_chars += len(snippet)
        if total_chars > max_chars:
            break

        full_context += snippet
        chunks.append(r)

    # ---------------------------------------------------------
    # Step 3 — Inference prompt
    # ---------------------------------------------------------
    prompt = f"""
You are a fraud intelligence expert.

Below are extracted report pages related to merchant fraud.
Read them holistically (not chunk-by-chunk) and infer patterns.

QUESTION:
{query_en}

DOCUMENT EXTRACTS:
{full_context}

INSTRUCTIONS:
- Summarize fraud patterns by merchant or category
- Identify entities with the highest fraud incidence
- Base conclusions strictly on the documents
- List page numbers used
""".strip()

    # ---------------------------------------------------------
    # Step 4 — LLM inference
    # ---------------------------------------------------------
    answer = llm_client.run(prompt, temperature=0.0)

    return {
        "type": "rag",
        "answer": answer,
        "chunks": chunks,
        "context_text": full_context,
        "citations": build_citations(chunks),
        "inference_used": True,
    }

# =============================================================================
# MAIN RAG ENTRYPOINT
# =============================================================================

def run_rag(query_en: str, user_lang: str) -> Dict[str, Any]:
    """
    Execute the RAG pipeline for a single query.
    """
    # Special inference path
    if is_merchant_incidence_question(query_en):
        logger.info("[RAG] Merchant inference mode activated.")
        return merchant_inference_mode(
            query_en=query_en,
            db_client=DB,
            llm_client=llm,
            user_lang=user_lang,
        )

    # ---------------------------------------------------------
    # Standard chunk-based RAG
    # ---------------------------------------------------------
    chunks = retrieve_top_k(query_en, top_k=10, source_name=None)

    context_text = build_context(chunks)
    prompt = build_prompt(query_en, context_text, user_lang)

    answer = llm.run(prompt, temperature=0.0)

    return {
        "type": "rag",
        "answer": answer,
        "chunks": chunks,
        "context_text": context_text,
        "citations": build_citations(chunks),
    }
