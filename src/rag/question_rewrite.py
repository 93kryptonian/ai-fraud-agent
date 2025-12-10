
"""
Question Rewrite Module 
-------------------------------------
- Prevents aggressive rewrite for domain-specific finance/regulatory/fraud questions.
- Only rewrites ambiguous questions.
- Handles multilingual translation (ID <-> EN).
- Returns metadata explaining whether rewrite happened.
"""

import re
from typing import Tuple, Dict

from src.llm.llm_client import llm


# =====================================================================
# LANGUAGE DETECTION (cheap heuristic + LLM fallback)
# =====================================================================

def detect_language(text: str) -> str:
    """Return 'id' or 'en'."""
    t = text.strip().lower()

    # Quick heuristic for Indonesian
    indo_keywords = ["yang", "apa", "bagaimana", "mengapa", "siapa", "berapa", "dengan", "pada", "tidak", "adalah"]
    if any(k in t for k in indo_keywords):
        return "id"

    # If contains “the”, “is”, “which”, assume English
    eng_keywords = ["what", "how", "which", "who", "when", "why", "the", "is"]
    if any(k in t for k in eng_keywords):
        return "en"

    # Fallback to LLM detection
    lang = llm.run(
        f"Detect if this text is Indonesian or English. Reply only 'id' or 'en':\n\n{text}",
        temperature=0.0
    ).strip().lower()

    return "id" if "id" in lang else "en"


# =====================================================================
# SAFE DOMAIN RULES
# Never rewrite fraud/merchant/regulatory questions — they are precise.
# =====================================================================

DOMAIN_KEYWORDS = [
    "merchant", "merchants",
    "category", "categories",

    "fraud rate", "fraud rates",
    "fraudulent",

    "card-not-present", "card not present",
    "cross-border", "cross border",

    "eea", "eba", "ecb", "psd2",

    "transaction counterpart",
    "domestic", "international",
]


def is_domain_specific(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in DOMAIN_KEYWORDS)


# =====================================================================
# TRANSLATION UTILITIES
# =====================================================================

def translate_id_to_en(text: str) -> str:
    """Translate Indonesian → English, preserving meaning."""
    return llm.run(
        f"Translate this to English without adding new meaning:\n{text}",
        temperature=0.0
    ).strip()


def translate_en_to_id(text: str) -> str:
    """Translate English → Indonesian."""
    return llm.run(
        f"Translate this to Indonesian clearly and professionally:\n{text}",
        temperature=0.0
    ).strip()


# =====================================================================
# REWRITE LOGIC
# =====================================================================

REWRITE_PROMPT = """
Rewrite the user query to be clearer and more retrieval-friendly,
but DO NOT add or assume facts. DO NOT introduce new entities.
Keep meaning EXACTLY the same.

User query:
{q}

Return only the rewritten query.
"""


def rewrite_query(q: str) -> str:
    """Rewrite using LLM with strict instructions."""
    out = llm.run(
        REWRITE_PROMPT.format(q=q),
        temperature=0.0
    ).strip()

    # Clean artifacts
    out = re.sub(r"^\"|\"$", "", out)
    return out


# =====================================================================
# MAIN ENTRYPOINT
# =====================================================================

def process_query(original_query: str) -> Tuple[str, Dict]:
    """
    Main function:
      - Detect language
      - Translate to English if needed
      - Skip rewrite for domain-specific queries
      - Rewrite only vague questions
      - Return rewritten_query + metadata
    """
    q = original_query.strip()
    lang = detect_language(q)

    meta = {
        "lang": lang,
        "rewritten": False,
        "translated": False,
        "original": original_query,
        "final_query_en": None,
    }

    # Translate Indonesian → English (retriever uses English)
    if lang == "id":
        q_en = translate_id_to_en(q)
        meta["translated"] = True
    else:
        q_en = q

    # 1. BLOCK rewrite for precise domain questions
    if is_domain_specific(q_en):
        meta["final_query_en"] = q_en
        return q_en, meta

    # 2. Rewrite ONLY if vague (heuristic)
    vague_keywords = ["explain", "describe", "tell me about", "what is", "how does"]
    if not any(k in q_en.lower() for k in vague_keywords):
        # Not vague → no rewrite
        meta["final_query_en"] = q_en
        return q_en, meta

    # 3. Perform rewrite
    rewritten = rewrite_query(q_en)
    meta["rewritten"] = True
    meta["final_query_en"] = rewritten

    return rewritten, meta
