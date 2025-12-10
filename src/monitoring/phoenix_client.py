# src/monitoring/phoenix_client.py

import os
from typing import Dict
from dotenv import load_dotenv
from phoenix.evals import LLMEvaluator, OpenAIModel
from src.utils.logger import get_logger


logger = get_logger(__name__)
load_dotenv()

# -------------------------------------------------
# Phoenix LLM Judge Setup
# -------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model used to perform evaluation 
model = OpenAIModel(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
)

# Standard RAG evaluation criteria
evaluator = LLMEvaluator(
       name="rag_eval",
    llm=model,
    prompt_template="RAG_RELEVANCY_PROMPT_TEMPLATE",
)

# -------------------------------------------------
# Evaluation Function
# -------------------------------------------------

def run_phoenix_llm_judge(question: str, answer: str, context: str) -> Dict:
    """
    Evaluate RAG model answers using Phoenix evaluator.
    This version does NOT require Phoenix dashboard or tracing.

    Returns:
        Dict with metric scores (all floats 0-1).
    """

    try:
        result = evaluator.evaluate(
            input=question,
            prediction=answer,
            reference=context,
        )

        metrics = result.to_dict()
        logger.info(f"[phoenix] eval metrics = {metrics}")

        return metrics

    except Exception as e:
        logger.error(f"[phoenix] evaluation failed: {e}")
        return {"error": str(e)}
