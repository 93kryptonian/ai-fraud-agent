# # src/monitoring/phoenix_client.py

# import os
# from typing import Dict, Optional
# from dotenv import load_dotenv
# from src.utils.logger import get_logger

# load_dotenv()
# logger = get_logger(__name__)

# # -------------------------------------------------
# # Config
# # -------------------------------------------------

# PHOENIX_ENABLED = os.getenv("PHOENIX_ENABLED", "false").lower() == "true"
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Lazy singletons
# _evaluator = None


# # -------------------------------------------------
# # Lazy initializer
# # -------------------------------------------------

# def _get_evaluator():
#     global _evaluator

#     if not PHOENIX_ENABLED:
#         return None

#     if _evaluator is not None:
#         return _evaluator

#     try:
#         from phoenix.evals import LLMEvaluator, OpenAIModel
#     except ImportError as e:
#         raise RuntimeError(
#             "PHOENIX_ENABLED=true but phoenix is not installed. "
#             "Install arize-phoenix to enable monitoring."
#         ) from e

#     model = OpenAIModel(
#         model="gpt-4o-mini",
#         api_key=OPENAI_API_KEY,
#     )

#     _evaluator = LLMEvaluator(
#         name="rag_eval",
#         llm=model,
#         prompt_template="RAG_RELEVANCY_PROMPT_TEMPLATE",
#     )

#     logger.info("[phoenix] evaluator initialized")
#     return _evaluator


# # -------------------------------------------------
# # Public API
# # -------------------------------------------------

# def run_phoenix_llm_judge(
#     question: str,
#     answer: str,
#     context: str,
# ) -> Optional[Dict]:
#     """
#     Evaluate RAG model answers using Phoenix evaluator.

#     Returns:
#         Dict with metric scores, or None if Phoenix is disabled.
#     """
#     evaluator = _get_evaluator()

#     if evaluator is None:
#         return None  # no-op in CI / local

#     try:
#         result = evaluator.evaluate(
#             input=question,
#             prediction=answer,
#             reference=context,
#         )

#         metrics = result.to_dict()
#         logger.info(f"[phoenix] eval metrics = {metrics}")
#         return metrics

#     except Exception as e:
#         logger.error(f"[phoenix] evaluation failed: {e}")
#         return {"error": str(e)}


# src/monitoring/phoenix_client.py

import os
from typing import Dict, Optional
from dotenv import load_dotenv

from src.utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()

# -------------------------------------------------
# Config
# -------------------------------------------------

PHOENIX_ENABLED = os.getenv("PHOENIX_ENABLED", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------------------------------
# Lazy Phoenix loader
# -------------------------------------------------

_evaluator = None


def _get_phoenix_evaluator():
    """
    Lazily initialize Phoenix evaluator.
    This function is NEVER called unless PHOENIX_ENABLED=true.
    """
    global _evaluator

    if _evaluator is not None:
        return _evaluator

    if not PHOENIX_ENABLED:
        return None

    if not OPENAI_API_KEY:
        logger.warning("Phoenix enabled but OPENAI_API_KEY is missing")
        return None

    try:
        from phoenix.evals import LLMEvaluator, OpenAIModel
    except ImportError:
        logger.warning("Phoenix enabled but phoenix is not installed")
        return None

    try:
        model = OpenAIModel(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
        )

        _evaluator = LLMEvaluator(
            name="rag_eval",
            llm=model,
            prompt_template="RAG_RELEVANCY_PROMPT_TEMPLATE",
        )

        logger.info("Phoenix evaluator initialized")
        return _evaluator

    except Exception as e:
        logger.error(f"Failed to initialize Phoenix evaluator: {e}")
        return None


# -------------------------------------------------
# Public API
# -------------------------------------------------

def run_phoenix_llm_judge(
    question: str,
    answer: str,
    context: str,
) -> Optional[Dict]:
    """
    Safe Phoenix evaluation wrapper.

    - Returns None if Phoenix is disabled or unavailable
    - NEVER raises
    - CI-safe
    """

    evaluator = _get_phoenix_evaluator()
    if evaluator is None:
        return None

    try:
        result = evaluator.evaluate(
            input=question,
            prediction=answer,
            reference=context,
        )
        metrics = result.to_dict()
        logger.info(f"[phoenix] metrics={metrics}")
        return metrics

    except Exception as e:
        logger.error(f"[phoenix] evaluation failed: {e}")
        return None
