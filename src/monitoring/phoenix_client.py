"""
PHOENIX LLM EVALUATION CLIENT
----------------------------

This module provides an OPTIONAL LLM-based evaluation layer using
Arize Phoenix. It is designed to be:

- Fully optional (feature-flagged)
- Lazy-loaded (no import-time dependency)
- CI-safe (never raises, never blocks execution)
- Non-critical-path (used only for scoring / monitoring)

Typical use:
- Evaluate RAG answer relevance / groundedness
- Sample-based or low-confidence auditing
- Offline quality monitoring (not user-facing)
"""

import os
from typing import Dict, Optional

from dotenv import load_dotenv
from src.utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

PHOENIX_ENABLED = os.getenv("PHOENIX_ENABLED", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =============================================================================
# LAZY SINGLETON (CRITICAL FOR CI)
# =============================================================================

_phoenix_evaluator = None


def _get_phoenix_evaluator():
    """
    Lazily initialize Phoenix evaluator.

    Guarantees:
    - Never called unless PHOENIX_ENABLED=true
    - Never raises
    - Safe in CI / local environments
    """
    global _phoenix_evaluator

    if _phoenix_evaluator is not None:
        return _phoenix_evaluator

    if not PHOENIX_ENABLED:
        return None

    if not OPENAI_API_KEY:
        logger.warning("[phoenix] Enabled but OPENAI_API_KEY is missing")
        return None

    try:
        from phoenix.evals import LLMEvaluator, OpenAIModel
    except ImportError:
        logger.warning("[phoenix] Enabled but phoenix is not installed")
        return None

    try:
        model = OpenAIModel(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
        )

        _phoenix_evaluator = LLMEvaluator(
            name="rag_eval",
            llm=model,
            prompt_template="RAG_RELEVANCY_PROMPT_TEMPLATE",
        )

        logger.info("[phoenix] Evaluator initialized")
        return _phoenix_evaluator

    except Exception as e:
        logger.error(f"[phoenix] Failed to initialize evaluator: {e}")
        return None

# =============================================================================
# PUBLIC API
# =============================================================================

def run_phoenix_llm_judge(
    question: str,
    answer: str,
    context: str,
) -> Optional[Dict]:
    """
    Evaluate a RAG answer using Phoenix (if enabled).

    Behavior:
    - Returns None if Phoenix is disabled or unavailable
    - Never raises
    - Safe for production and CI pipelines

    Returns:
        Dict of evaluation metrics, or None
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
        logger.error(f"[phoenix] Evaluation failed: {e}")
        return None
