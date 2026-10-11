# src/llm/refusal.py
"""
Recognise the RAG pipeline's "the documents do not say" refusal, in English and Indonesian.

The model answers in the user's language, so for an Indonesian question the refusal
arrives in Indonesian. A check that only knows the English sentence scores that
refusal as a confident answer.
"""

REFUSAL_MARKERS = (
    # English
    "not provide enough information",
    "not contain enough information",
    # Indonesian (the model's wording varies: both orders are seen)
    "tidak memberikan cukup informasi",
    "tidak memberikan informasi yang cukup",
    "tidak mengandung cukup informasi",
    "tidak mengandung informasi yang cukup",
)


def is_refusal(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)
