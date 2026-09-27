"""
Question Rewrite Module
-----------------------
Responsibilities:
- Detect user language (EN / ID)
- Translate queries to English for retrieval
- Prevent rewriting of domain-specific (fraud / regulatory) questions
- Rewrite only vague questions
- Return metadata explaining every transformation

Design principles:
- Conservative by default
- Never add facts
- Never change intent
- Rewrite only when it improves retrieval
"""

import re
from typing import Tuple, Dict

from src.llm.llm_client import llm

# =============================================================================
# LANGUAGE DETECTION
# =============================================================================

def detect_language(text: str) -> str:
    """
    Detect query language.

    Strategy:
    1. Cheap keyword heuristic
    2. LLM fallback only if ambiguous

    Returns:
        "id" or "en"
    """
    t = text.strip().lower()

    indo_keywords = (
        "yang", "apa", "bagaimana", "mengapa",
        "siapa", "berapa", "dengan", "pada",
        "tidak", "adalah",
    )
    if any(k in t for k in indo_keywords):
        return "id"

    eng_keywords = (
        "what", "how", "which", "who",
        "when", "why", "the", "is",
    )
    if any(k in t for k in eng_keywords):
        return "en"

    # Fallback: LLM detection
    lang = llm.run(
        "Detect if this text is Indonesian or English. "
        "Reply only with 'id' or 'en'.\n\n"
        f"{text}",
        temperature=0.0,
        purpose="language_detection",
    ).strip().lower()

    return "id" if "id" in lang else "en"

# =============================================================================
# DOMAIN GUARDRAILS
# =============================================================================

DOMAIN_KEYWORDS = (
    "merchant", "merchants",
    "category", "categories",
    "fraud rate", "fraud rates",
    "fraudulent",
    "card-not-present", "card not present",
    "cross-border", "cross border",
    "eea", "eba", "ecb", "psd2",
    "transaction counterpart",
    "domestic", "international",
)


def is_domain_specific(query: str) -> bool:
    """
    Domain-specific queries are assumed to be precise
    and must NOT be rewritten.
    """
    q = query.lower()
    return any(k in q for k in DOMAIN_KEYWORDS)

# =============================================================================
# TRANSLATION UTILITIES
# =============================================================================

def translate_id_to_en(text: str) -> str:
    """Translate Indonesian → English without altering meaning."""
    return llm.run(
        "Translate the following text to English without adding new meaning:\n"
        f"{text}",
        temperature=0.0,
        purpose="translation",
    ).strip()


def translate_en_to_id(text: str) -> str:
    """Translate English → Indonesian clearly and professionally."""
    return llm.run(
        "Translate the following text to Indonesian clearly and professionally:\n"
        f"{text}",
        temperature=0.0,
        purpose="translation",
    ).strip()

# =============================================================================
# REWRITE LOGIC
# =============================================================================

REWRITE_PROMPT = """
Rewrite the user query to be clearer and more retrieval-friendly.

Rules:
- Do NOT add facts
- Do NOT assume missing information
- Do NOT introduce new entities
- Preserve the original meaning EXACTLY

User query:
{q}

Return only the rewritten query.
""".strip()


def rewrite_query(query: str) -> str:
    """
    Rewrite a vague query using LLM with strict constraints.
    """
    rewritten = llm.run(
        REWRITE_PROMPT.format(q=query),
        temperature=0.0,
        purpose="query_rewrite",
    ).strip()

    # Remove accidental quotation artifacts
    rewritten = re.sub(r'^"|"$', "", rewritten)
    return rewritten

# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

def process_query(original_query: str) -> Tuple[str, Dict]:
    """
    Process a user query for retrieval.

    Flow:
    1. Detect language
    2. Translate to English if needed
    3. Block rewrite for domain-specific queries
    4. Rewrite ONLY vague questions
    5. Return final query + metadata
    """
    query = original_query.strip()

    # P1 note: llm.run() now raises LLMExhaustedRetriesError instead of
    # returning a fake-success sentinel string once retries are exhausted
    # (src/llm/llm_client.py). Every call in this function that reaches the
    # LLM is therefore wrapped locally so this module stays self-contained
    # and CI-safe on its own — matching this module's own stated design
    # principle ("Conservative by default") — rather than relying on every
    # caller to happen to wrap it, which P1 showed wasn't actually true
    # (a direct call to process_query(), or the Indonesian-translation
    # branch specifically, wasn't protected before this fix).
    try:
        lang = detect_language(query)
    except Exception:
        lang = "en"

    meta: Dict = {
        "lang": lang,
        "rewritten": False,
        "translated": False,
        "original": original_query,
        "final_query_en": None,
    }

    # Step 1 — translation (retriever operates in English)
    if lang == "id":
        try:
            query_en = translate_id_to_en(query)
            meta["translated"] = True
        except Exception:
            # Fail closed to the original text rather than crashing the
            # whole pipeline over a translation failure.
            query_en = query
    else:
        query_en = query

    # Step 2 — block rewrite for precise domain questions
    if is_domain_specific(query_en):
        meta["final_query_en"] = query_en
        return query_en, meta

    # Step 3 — rewrite only vague queries
    vague_markers = (
        "explain", "describe", "tell me about",
        "what is", "how does",
    )
    if not any(k in query_en.lower() for k in vague_markers):
        meta["final_query_en"] = query_en
        return query_en, meta

    # Step 4 — controlled rewrite
    try:
        rewritten = rewrite_query(query_en)
        meta["rewritten"] = True
        meta["final_query_en"] = rewritten
        return rewritten, meta
    except Exception:
        # Conservative by default: if rewriting fails, use the
        # (unrewritten) query rather than crashing the pipeline.
        meta["final_query_en"] = query_en
        return query_en, meta
