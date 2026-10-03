# src/rag/rag_chain.py
"""
Retrieval-Augmented Generation (RAG) pipeline.

Responsibilities:
- Retrieve relevant document chunks
- Construct bounded, traceable context
- Build safe, grounded prompts

Design principles:
- Context-first, not LLM-first
- Hard character limits to prevent overflow
- Explicit fallback instructions
"""

from typing import Dict, Any, List

from src.rag.retriever_direct import retrieve_top_k_with_status
from src.llm.llm_client import llm
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
# MAIN RAG ENTRYPOINT
# =============================================================================

def run_rag(query_en: str, user_lang: str) -> Dict[str, Any]:
    """
    Execute the RAG pipeline for a single query.
    """
    # ---------------------------------------------------------
    # Standard chunk-based RAG
    # ---------------------------------------------------------
    chunks, retrieval_status = retrieve_top_k_with_status(
        query_en, top_k=10, source_name=None
    )

    if not chunks:
        # No context: don't pay for an LLM call that can only produce the
        # fallback, and tell the caller *why* (outage vs. nothing relevant).
        logger.warning(f"[RAG] No chunks | retrieval_status={retrieval_status}")
        if retrieval_status == "unavailable":
            answer = (
                "Sorry, the document search is temporarily unavailable. "
                "Please try again later."
            )
        else:
            answer = (
                "Sorry, the available documents do not provide enough "
                "information to answer your question."
            )
        return {
            "type": "rag",
            "answer": answer,
            "chunks": [],
            "context_text": "",
            "citations": [],
            "retrieval_status": retrieval_status,
        }

    context_text = build_context(chunks)
    prompt = build_prompt(query_en, context_text, user_lang)

    answer = llm.run(prompt, temperature=0.0, purpose="rag_answer")

    return {
        "type": "rag",
        "answer": answer,
        "chunks": chunks,
        "context_text": context_text,
        "citations": build_citations(chunks),
        "retrieval_status": retrieval_status,
    }
