# src/rag/insight_layer.py
"""
Insight generation layer for fraud-focused RAG.

Purpose:
- Generate high-level analytical insights
- STRICTLY grounded in the same RAG context
- Never introduce new facts
- Never contradict the main answer
- Never guess beyond retrieved evidence
"""

from typing import Optional

from src.llm.llm_client import llm


def generate_insight(
    answer: str,
    context_text: str,
    user_lang: str,
) -> Optional[str]:
    """
    Generate a concise analytical insight derived from the RAG answer
    and its supporting context.

    Returns None if:
    - The answer is empty
    - The system already indicates insufficient information
    """
    if not answer:
        return None

    if "not provide enough information" in answer.lower():
        return None

    language_instruction = (
        "Write the insight in Indonesian."
        if user_lang == "id"
        else "Write the insight in English."
    )

    prompt = f"""
Generate a concise analytical insight based strictly on the answer and context below.

{language_instruction}

ANSWER:
{answer}

CONTEXT:
{context_text}

Rules:
- Do NOT introduce new facts
- Do NOT contradict the answer
- Do NOT speculate beyond the context
- Focus on implications, not restatement
- 3-5 sentences only
""".strip()

    return llm.run(prompt, temperature=0.2)
