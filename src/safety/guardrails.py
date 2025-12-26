"""
INPUT GUARDRAILS FOR an Enterprise Fraud Intelligence System
----------------------------------

This module enforces strict safety and domain constraints on user input.

Responsibilities:
- Basic sanitization & normalization
- Prompt-injection detection
- Noise / junk query rejection
- Domain enforcement (fraud-only)
- Language detection (EN / ID)

Design principles:
- Deterministic (no LLM calls)
- Fast (regex + heuristics only)
- CI-safe
- Fail-closed (reject by default)
"""

import re
from typing import Optional, Tuple

from src.utils.logger import get_logger
from src.rag.question_rewrite import detect_language

logger = get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_QUERY_LENGTH = 3000
MIN_QUERY_LENGTH = 2

# Common prompt-injection & jailbreak patterns
FORBIDDEN_PATTERNS = [
    r"ignore\s+all\s+previous\s+instructions",
    r"forget\s+.*system\s+prompt",
    r"jailbreak",
    r"simulate\s+.*assistant",
    r"bypass\s+.*security",
    r"act\s+as\s+.*system",
]

# Domain allowlist (English + Indonesian)
DOMAIN_KEYWORDS = [
    # English
    r"fraud",
    r"financial crime",
    r"aml",
    r"money laundering",
    r"card fraud",
    r"credit card",
    r"payment fraud",
    r"chargeback",
    r"risk scoring",
    r"identity theft",
    r"authentication",
    r"psd2",
    r"strong customer authentication",
    r"eba",
    r"ecb",
    r"cross[-\s]?border fraud",

    # Indonesian
    r"penipuan",
    r"fraud kartu",
    r"kejahatan finansial",
    r"pencurian identitas",
    r"kejahatan keuangan",
    r"pemalsuan kartu",
    r"transaksi lintas negara",
]

# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def clean_whitespace(text: str) -> str:
    """
    Normalize whitespace and remove control characters.
    """
    text = text.replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def trim_overlong(text: str) -> str:
    """
    Enforce maximum query length.
    """
    if len(text) > MAX_QUERY_LENGTH:
        logger.warning(
            f"[guardrails] Query trimmed from {len(text)} to {MAX_QUERY_LENGTH} chars"
        )
        return text[:MAX_QUERY_LENGTH]
    return text

# =============================================================================
# PROMPT-INJECTION DETECTION
# =============================================================================

def detect_prompt_injection(text: str) -> bool:
    """
    Detect common prompt-injection or jailbreak attempts.
    """
    lowered = text.lower()

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, lowered):
            logger.warning(f"[guardrails] Prompt injection detected: {pattern}")
            return True

    # Structural red flags
    if any(tok in lowered for tok in ["```", "<system>", "<assistant>", "### system"]):
        logger.warning("[guardrails] Structural injection marker detected")
        return True

    return False

# =============================================================================
# BASIC SEMANTIC CHECKS
# =============================================================================

def too_short(text: str) -> bool:
    return len(text.strip()) < MIN_QUERY_LENGTH


def contains_only_noise(text: str) -> bool:
    """
    Reject inputs that contain only symbols / punctuation.
    """
    return bool(re.fullmatch(r"[\W_]+", text.strip()))

# =============================================================================
# DOMAIN ENFORCEMENT
# =============================================================================

def is_domain_related(query: str) -> bool:
    """
    Check whether the query is related to fraud / financial crime.
    """
    lowered = query.lower()

    for kw in DOMAIN_KEYWORDS:
        if re.search(kw, lowered):
            return True

    return False

# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

def validate_query(text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate user input for safety and domain relevance.

    Returns:
        (is_valid, cleaned_text_or_error_message, detected_language)

    detected_language ∈ {"en", "id"}
    """
    lang = detect_language(text)

    # -------------------------------------------------
    # 1. Empty / meaningless input
    # -------------------------------------------------
    if not text or too_short(text):
        msg = "Query too short." if lang == "en" else "Pertanyaan terlalu pendek."
        return False, msg, lang

    if contains_only_noise(text):
        msg = (
            "Query contains no meaningful content."
            if lang == "en"
            else "Pertanyaan tidak memiliki konteks yang jelas."
        )
        return False, msg, lang

    # -------------------------------------------------
    # 2. Prompt-injection attempts
    # -------------------------------------------------
    if detect_prompt_injection(text):
        msg = (
            "Potential prompt-injection attempt detected."
            if lang == "en"
            else "Terdeteksi upaya prompt injection."
        )
        return False, msg, lang

    # -------------------------------------------------
    # 3. Domain restriction (fraud-only)
    # -------------------------------------------------
    if not is_domain_related(text):
        msg = (
            "Sorry, I can only answer questions related to fraud, financial crime, "
            "or the supported documents."
            if lang == "en"
            else
            "Maaf, saya hanya dapat menjawab pertanyaan terkait fraud, kejahatan finansial, "
            "atau dokumen yang tersedia."
        )
        return False, msg, lang

    # -------------------------------------------------
    # 4. Sanitization
    # -------------------------------------------------
    cleaned = clean_whitespace(text)
    cleaned = trim_overlong(cleaned)

    return True, cleaned, lang
