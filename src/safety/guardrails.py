import re
from typing import Optional, Tuple
from src.utils.logger import get_logger
from src.rag.question_rewrite import detect_language

logger = get_logger(__name__)

# ======================================================
# CONFIG
# ======================================================

MAX_QUERY_LENGTH = 3000
MIN_QUERY_LENGTH = 2

FORBIDDEN_PATTERNS = [
    r"ignore\s+all\s+previous\s+instructions",
    r"forget\s+.*system\s+prompt",
    r"jailbreak",
    r"simulate\s+.*assistant",
    r"bypass\s+.*security",
    r"act\s+as\s+.*system",
]

# Domain-specific allowlist (English + Indonesian)
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
    r"ema report",
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

# ======================================================
# INPUT SANITIZATION
# ======================================================

def clean_whitespace(text: str) -> str:
    text = text.replace("\r", " ").replace("\t", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def trim_overlong(text: str) -> str:
    if len(text) > MAX_QUERY_LENGTH:
        logger.warning(f"Query trimmed from {len(text)} chars to {MAX_QUERY_LENGTH}.")
        return text[:MAX_QUERY_LENGTH]
    return text

# ======================================================
# PROMPT-INJECTION DETECTION
# ======================================================

def detect_prompt_injection(text: str) -> bool:
    lowered = text.lower()

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, lowered):
            logger.warning(f"Prompt injection detected: {pattern}")
            return True

    if any(k in lowered for k in ["```", "<system>", "<assistant>", "### system"]):
        return True

    return False

# ======================================================
# SEMANTIC CHECK
# ======================================================

def too_short(text: str) -> bool:
    return len(text.strip()) < MIN_QUERY_LENGTH

def contains_only_noise(text: str) -> bool:
    return bool(re.fullmatch(r"[\W_]+", text.strip()))

# ======================================================
# DOMAIN CHECKING
# ======================================================

def is_domain_related(query: str) -> bool:
    """Return True if the query is about fraud/financial crime."""
    lowered = query.lower()

    for kw in DOMAIN_KEYWORDS:
        if re.search(kw, lowered):
            return True
    return False


# ======================================================
# MAIN GUARDRAIL FUNCTION
# ======================================================

def validate_query(text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate user input for safety and domain relevance.

    Returns:
        (is_valid, cleaned_text_or_error, detected_lang)

        detected_lang ∈ {"en", "id"}
    """

    lang = detect_language(text)

    # 1. Useless content
    if not text or too_short(text):
        return False, ("Query too short." if lang == "en" else "Pertanyaan terlalu pendek."), lang

    if contains_only_noise(text):
        msg = "Query contains no meaningful content." if lang == "en" else \
              "Pertanyaan tidak memiliki konteks yang jelas."
        return False, msg, lang

    # 2. Prompt-injection attempts
    if detect_prompt_injection(text):
        msg = "Potential prompt-injection attempt detected." if lang == "en" else \
              "Terdeteksi upaya prompt injection."
        return False, msg, lang

    # 3. Domain filtering (your requirement)
    if not is_domain_related(text):
        msg = (
            "Sorry, I can only answer questions related to fraud, financial crime, or the supported documents."
            if lang == "en"
            else
            "Maaf, saya hanya dapat menjawab pertanyaan terkait fraud, kejahatan finansial, atau dokumen yang tersedia."
        )
        return False, msg, lang

    # 4. Sanitize
    cleaned = clean_whitespace(text)
    cleaned = trim_overlong(cleaned)

    return True, cleaned, lang
