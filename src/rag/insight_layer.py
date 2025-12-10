# # src/rag/insight_layer.py

# """
# Insight layer for fraud-focused RAG.

# Generates analytic, high-level insights STRICTLY based on the same RAG context.
# Never introduces new facts.
# Never contradicts answer.
# Never guesses.
# """


from src.llm.llm_client import llm

def generate_insight(answer: str, context_text: str, user_lang: str) -> str:
    if not answer or "not provide enough information" in answer.lower():
        return None

    lang_instruction = (
        "Write the insight in Indonesian."
        if user_lang == "id"
        else "Write the insight in English."
    )

    prompt = f"""
Generate a concise analytical insight based on this answer and supporting context.

{lang_instruction}

ANSWER:
{answer}

CONTEXT:
{context_text}

Your insight must:
 - capture the implication of the findings,
 - be 3-5 sentences,
 - be factual and grounded.
""".strip()

    return llm.run(prompt, temperature=0.2)
