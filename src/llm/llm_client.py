# src/llm/llm_client.py

import os
import time
import re
from typing import Optional, Type, Any, Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from src.utils.logger import get_logger
from src.llm.response_schema import RAGResponse  # type hints only

load_dotenv()
logger = get_logger(__name__)

# ======================================================
# ENV SETTINGS (SAFE TO READ AT IMPORT)
# ======================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o-mini")

MAX_RETRIES = 4
MAX_TOKENS = 2048

MAX_COST_USD = float(os.getenv("MAX_COST_USD", "0.10"))
SESSION_COST = 0.0

# ======================================================
# LAZY OPENAI CLIENT (CRITICAL FIX)
# ======================================================
_openai_client = None


def get_openai_client():
    """
    Lazily create OpenAI client.
    - SAFE at import time
    - Raises ONLY when actually used
    """
    global _openai_client

    if _openai_client is not None:
        return _openai_client

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    from openai import OpenAI  # local import = CI safe
    _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ======================================================
# COST ESTIMATION
# ======================================================
PRICES: Dict[str, float] = {
    "gpt-4o": 0.0025,
    "gpt-4o-mini": 0.00015,
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model not in PRICES:
        return 0.0
    return round((prompt_tokens + completion_tokens) / 1000 * PRICES[model], 6)


# ======================================================
# LLM CLIENT
# ======================================================
class LLMClient:
    """
    CI-safe LLM wrapper.
    No external calls or secrets at import time.
    """

    def __init__(self):
        self.default_model = DEFAULT_MODEL
        self.fallback_model = FALLBACK_MODEL

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        if messages:
            return messages

        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    @staticmethod
    def _extract_json_block(text: str) -> str:
        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1)

        first, last = text.find("{"), text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return text[first:last + 1]

        return text

    # -----------------------------
    # Public API
    # -----------------------------
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

        selected_model = model or self.default_model
        global SESSION_COST

        extra_params = extra_params or {}

        for attempt in range(MAX_RETRIES):

            if SESSION_COST >= MAX_COST_USD:
                selected_model = self.fallback_model

            try:
                client = get_openai_client()  # ← ONLY HERE

                msgs = self._build_messages(prompt, system_prompt, messages)

                resp = client.chat.completions.create(
                    model=selected_model,
                    messages=msgs,
                    max_tokens=MAX_TOKENS,
                    temperature=temperature,
                    **extra_params,
                )

                text = (resp.choices[0].message.content or "").strip()

                usage = getattr(resp, "usage", None)
                if usage:
                    cost = estimate_cost(
                        selected_model,
                        usage.prompt_tokens or 0,
                        usage.completion_tokens or 0,
                    )
                    SESSION_COST += cost

                if response_schema:
                    try:
                        json_str = self._extract_json_block(text)
                        return response_schema.model_validate_json(json_str)
                    except ValidationError:
                        logger.warning("Schema validation failed, returning raw text")

                return text

            except Exception as e:
                logger.warning(f"LLM error attempt={attempt + 1}: {e}")
                time.sleep(1.2 * (attempt + 1))

        return "LLM failed after retries."


# ======================================================
# SINGLETON
# ======================================================
llm = LLMClient()
