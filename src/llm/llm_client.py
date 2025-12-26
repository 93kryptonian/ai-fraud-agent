# src/llm/llm_client.py
"""
Enterprise LLM Client Wrapper.

Design goals:
- CI-safe (no external calls at import time)
- Lazy client initialization
- Deterministic defaults (temperature=0)
- Cost tracking & soft budget enforcement
- Optional schema validation
- Retry with backoff
"""

import os
import re
import time
from typing import Optional, Type, Any, Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# =============================================================================
# ENV CONFIGURATION (SAFE AT IMPORT)
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o-mini")

MAX_RETRIES = 4
MAX_TOKENS = 2048

MAX_COST_USD = float(os.getenv("MAX_COST_USD", "0.10"))

# Session-scoped soft budget (intentionally global)
SESSION_COST_USD = 0.0

# =============================================================================
# LAZY OPENAI CLIENT
# =============================================================================

_openai_client = None


def get_openai_client():
    """
    Lazily instantiate OpenAI client.

    Important:
    - No API key validation at import time
    - Raises ONLY when actually invoked
    - Safe for CI / unit tests
    """
    global _openai_client

    if _openai_client is not None:
        return _openai_client

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    from openai import OpenAI  # local import → CI safe
    _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

# =============================================================================
# COST ESTIMATION
# =============================================================================

# USD per 1K tokens (approx, intentionally conservative)
PRICES_PER_1K: Dict[str, float] = {
    "gpt-4o": 0.0025,
    "gpt-4o-mini": 0.00015,
}


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """
    Estimate request cost in USD.
    Returns 0.0 if model pricing is unknown.
    """
    price = PRICES_PER_1K.get(model)
    if not price:
        return 0.0

    return round((prompt_tokens + completion_tokens) / 1000 * price, 6)

# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    """
    Thin, production-safe wrapper around OpenAI Chat Completions.

    Guarantees:
    - No external calls at import time
    - Deterministic defaults
    - Graceful degradation on failure
    """

    def __init__(self):
        self.default_model = DEFAULT_MODEL
        self.fallback_model = FALLBACK_MODEL

    # ------------------------------------------------------------------
    # MESSAGE BUILDERS
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: Optional[str],
        messages: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        """
        Normalize input into OpenAI-compatible message format.
        """
        if messages:
            return messages

        result: List[Dict[str, str]] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        result.append({"role": "user", "content": prompt})
        return result

    # ------------------------------------------------------------------
    # RESPONSE PARSING
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """
        Extract JSON from fenced or inline LLM output.
        """
        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1)

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return text[first:last + 1]

        return text

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        response_schema: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a single LLM request.

        Behavior:
        - Retries with backoff
        - Falls back to cheaper model after budget threshold
        - Optionally validates structured responses
        """
        global SESSION_COST_USD

        selected_model = model or self.default_model
        extra_params = extra_params or {}

        for attempt in range(1, MAX_RETRIES + 1):

            # Soft budget guard → downgrade model
            if SESSION_COST_USD >= MAX_COST_USD:
                selected_model = self.fallback_model

            try:
                client = get_openai_client()

                msgs = self._build_messages(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    messages=messages,
                )

                response = client.chat.completions.create(
                    model=selected_model,
                    messages=msgs,
                    max_tokens=MAX_TOKENS,
                    temperature=temperature,
                    **extra_params,
                )

                text = (response.choices[0].message.content or "").strip()

                # -------------------------------
                # Cost tracking
                # -------------------------------
                usage = getattr(response, "usage", None)
                if usage:
                    cost = estimate_cost(
                        selected_model,
                        usage.prompt_tokens or 0,
                        usage.completion_tokens or 0,
                    )
                    SESSION_COST_USD += cost

                # -------------------------------
                # Optional schema validation
                # -------------------------------
                if response_schema:
                    try:
                        json_str = self._extract_json_block(text)
                        return response_schema.model_validate_json(json_str)
                    except ValidationError:
                        logger.warning(
                            "[llm] Schema validation failed — returning raw text"
                        )

                return text

            except Exception as e:
                logger.warning(
                    f"[llm] Error attempt={attempt}/{MAX_RETRIES}: {e}"
                )
                time.sleep(1.2 * attempt)

        return "LLM failed after retries."

# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

llm = LLMClient()
