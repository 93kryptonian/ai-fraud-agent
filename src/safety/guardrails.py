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
from typing import Literal, Optional, Tuple

from src.utils.logger import get_logger
from src.rag.question_rewrite import detect_language, detect_language_heuristic

logger = get_logger(__name__)

# =============================================================================
# MACHINE-READABLE REJECTION REASONS (M4)
# =============================================================================
# Stable codes for the observability layer. Deliberately separate from the
# human-facing message strings below: the two are independent
# representations of the same decision, and the event layer must never
# infer a reason by parsing a translated sentence. See
# docs/observability-contract.md §4/§6.
GuardrailReason = Literal["too_short", "noise", "injection", "out_of_domain"]

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
    # Indonesian equivalents. \b keeps "mengabaikan transaksi ..." (to ignore transactions) from matching.
    r"\babaikan\s+(?:semua\s+)?(?:instruksi|perintah|aturan)\s+(?:sebelumnya|di\s+atas)",
    r"\blupakan\s+(?:semua\s+)?(?:instruksi|perintah|aturan)",
    r"\blupakan\s+.*system\s+prompt",
    r"\b(?:tampilkan|ungkapkan|bocorkan|tunjukkan)\s+.*system\s+prompt",
    r"\b(?:lewati|terobos)\s+.*keamanan",
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
    # Dataset vocabulary: analytics questions like "monthly trend of
    # transactions" or "top merchants" are in scope even without the word
    # "fraud" (the analytics/intent stages still decide how to answer).
    r"transactions?",
    r"merchants?",
    r"merchant categor(?:y|ies)",

    # Indonesian
    r"penipuan",
    r"fraud kartu",
    r"kejahatan finansial",
    r"pencurian identitas",
    r"kejahatan keuangan",
    r"pemalsuan kartu",
    r"transaksi lintas negara",
    r"transaksi",
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

# Short acronyms must match as whole words ("eba" must not match "member");
# longer terms only need a leading boundary so "fraud" still matches
# "fraudulent"/"fraudster".
_ACRONYMS = {"aml", "eba", "ecb", "psd2"}


def _with_word_boundary(kw: str) -> str:
    return rf"\b{kw}\b" if kw in _ACRONYMS else rf"\b{kw}"


def is_domain_related(query: str) -> bool:
    """
    Check whether the query is related to fraud / financial crime.
    """
    lowered = query.lower()

    for kw in DOMAIN_KEYWORDS:
        if re.search(_with_word_boundary(kw), lowered):
            return True

    return False

_INDONESIAN_HINTS = re.compile(
    r"\b(abaikan|lupakan|tampilkan|ungkapkan|bocorkan|tunjukkan|lewati|terobos|"
    r"instruksi|perintah|aturan|keamanan|sebelumnya|tentang|jelaskan|untuk|dari|dan)\b"
)


def _guess_message_language(text: str) -> str:
    """No-LLM language guess used only to word a rejection when the keyword rules found nothing."""
    return "id" if _INDONESIAN_HINTS.search(text.lower()) else "en"


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

def validate_query(
    text: str,
) -> Tuple[bool, Optional[str], Optional[str], Optional[GuardrailReason]]:
    """
    Validate user input for safety and domain relevance.

    Returns:
        (is_valid, cleaned_text_or_error_message, detected_language, reason)

    detected_language ∈ {"en", "id"}
    reason is None when is_valid is True, otherwise one of GuardrailReason —
    a stable, machine-readable code for the observability layer (M4). It is
    intentionally independent of the human-facing message: callers building
    telemetry must use `reason`, never infer one by parsing the message.
    """
    # A rejected input must not cost an LLM call. Wording a rejection needs only
    # a cheap, LLM-free language guess; the authoritative detection (which falls
    # back to the LLM for text the keyword rules do not recognise) runs once, at
    # the end, for a query that was accepted.
    lang = detect_language_heuristic(text) or _guess_message_language(text)

    # -------------------------------------------------
    # 1. Empty / meaningless input
    # -------------------------------------------------
    if not text or too_short(text):
        msg = "Query too short." if lang == "en" else "Pertanyaan terlalu pendek."
        return False, msg, lang, "too_short"

    if contains_only_noise(text):
        msg = (
            "Query contains no meaningful content."
            if lang == "en"
            else "Pertanyaan tidak memiliki konteks yang jelas."
        )
        return False, msg, lang, "noise"

    # -------------------------------------------------
    # 2. Prompt-injection attempts
    # -------------------------------------------------
    if detect_prompt_injection(text):
        msg = (
            "Potential prompt-injection attempt detected."
            if lang == "en"
            else "Terdeteksi upaya prompt injection."
        )
        return False, msg, lang, "injection"

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
        return False, msg, lang, "out_of_domain"

    # -------------------------------------------------
    # 4. Sanitization
    # -------------------------------------------------
    cleaned = clean_whitespace(text)
    cleaned = trim_overlong(cleaned)

    # Unlike orchestrator.py's own detect_language() call, this one has no
    # surrounding try/except upstream. llm.run() raises on exhausted retries
    # (P1), so keep the prior fallback to "en" explicitly instead of by accident.
    try:
        lang = detect_language(text)
    except Exception:
        lang = "en"

    return True, cleaned, lang, None
