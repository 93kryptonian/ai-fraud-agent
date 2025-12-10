import os
import time
import re

from typing import Optional, Type, Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.utils.logger import get_logger
from src.llm.response_schema import RAGResponse  # noqa: F401  # used in type hints elsewhere

load_dotenv()
logger = get_logger(__name__)

# ======================================================
# ENV SETTINGS
# ======================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o-mini")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "ollama/llama3")  # optional

MAX_RETRIES = 4
MAX_TOKENS = 2048

# Soft budget limit
MAX_COST_USD = float(os.getenv("MAX_COST_USD", "0.10"))  # 10 cents/session

# Internal token accounting
SESSION_COST = 0.0

# ======================================================
# OpenAI CLIENT
# ======================================================
client = OpenAI(api_key=OPENAI_API_KEY)


# ======================================================
# Cost Estimation
# ======================================================
PRICES: Dict[str, float] = {
    "gpt-4o": 0.0025,       # approx per 1K tokens
    "gpt-4o-mini": 0.00015,
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Rough cost estimation to prevent runaway usage."""
    if model not in PRICES:
        return 0.0
    total = (prompt_tokens + completion_tokens) / 1000 * PRICES[model]
    return round(total, 6)


# ======================================================
# LOCAL FALLBACK (OPTIONAL)
# ======================================================
def local_llm_fallback(prompt: str) -> str:
    logger.warning("Falling back to local model...")
    try:
        import subprocess

        proc = subprocess.run(
            ["ollama", "run", LOCAL_MODEL, prompt],
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception as e:
        logger.error(f"Local model failed: {e}")
        return "Model unavailable."


# ======================================================
# LLM RUNNER
# ======================================================
class LLMClient:
    """
    Thin wrapper around OpenAI chat completion with:
      - soft cost guardrail
      - retry & model downgrade
      - optional system prompt / multi-message support
      - tolerant Pydantic schema parsing for RAG JSON responses
    """

    def __init__(self):
        self.default_model = DEFAULT_MODEL
        self.fallback_model = FALLBACK_MODEL
        self.client = client

    # -----------------------------
    # Internal helpers
    # -----------------------------
    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build OpenAI chat messages.
        - If `messages` is provided, use it directly (caller fully controls).
        - Else, compose [system?, user] from system_prompt + prompt.
        """
        if messages:
            return messages

        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """
        Try to extract a JSON block from an LLM response.

        Handles:
          - ```json ... ``` fenced blocks
          - ``` ... ``` generic fenced blocks
          - bare JSON starting at first '{' and ending at last '}'

        Falls back to the original text if nothing reasonable is found.
        """

        # ```json ... ```
        fenced_json = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced_json:
            return fenced_json.group(1).strip()

        # ``` ... ```
        fenced = re.search(r"```(?:[a-zA-Z]*)\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()

        # First {...} span in the text
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = text[first:last + 1].strip()
            return candidate

        return text.strip()

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
        """
        Main entrypoint.

        Args:
            prompt: legacy single user prompt (ignored if `messages` is provided).
            model: override model name.
            temperature: sampling temperature.
            response_schema: optional Pydantic model to validate a JSON response.
            system_prompt: optional system message for better RAG control.
            messages: optional full chat history; if provided, overrides prompt/system_prompt.
            extra_params: optional dict forwarded to OpenAI API (e.g. top_p, frequency_penalty).

        Returns:
            - If `response_schema` is provided and validation succeeds: an instance of that schema.
            - Otherwise: raw text content from the model.
        """

        selected_model = model or self.default_model
        global SESSION_COST

        # small preview to help debug prompt explosions
        try:
            preview_src = prompt if messages is None else str(messages)
            logger.debug(f"[LLM] prompt_preview={preview_src[:400]}...")
        except Exception:
            pass

        extra_params = extra_params or {}

        for attempt in range(MAX_RETRIES):

            # Guardrail: soft cost limit
            if SESSION_COST >= MAX_COST_USD:
                if selected_model != self.fallback_model:
                    logger.warning(
                        f"Cost limit reached (SESSION_COST={SESSION_COST:.4f} >= {MAX_COST_USD:.4f}) "
                        f"— switching to fallback model: {self.fallback_model}"
                    )
                selected_model = self.fallback_model

            try:
                msgs = self._build_messages(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    messages=messages,
                )

                logger.info(
                    f"LLM call: model={selected_model}, "
                    f"temperature={temperature}, attempt={attempt + 1}"
                )

                resp = self.client.chat.completions.create(
                    model=selected_model,
                    messages=msgs,
                    max_tokens=MAX_TOKENS,
                    temperature=temperature,
                    **extra_params,
                )

                text = (resp.choices[0].message.content or "").strip()

                # Token & cost tracking
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    p = usage.prompt_tokens or 0
                    c = usage.completion_tokens or 0
                    cost = estimate_cost(selected_model, p, c)
                    SESSION_COST += cost

                    logger.info(
                        f"Tokens prompt={p}, completion={c}, "
                        f"cost={cost}, total_cost={SESSION_COST}"
                    )
                else:
                    logger.info("No usage information returned by model.")

                # If schema required — validate JSON with robust extraction
                if response_schema:
                    try:
                        json_str = self._extract_json_block(text)
                        parsed = response_schema.model_validate_json(json_str)
                        return parsed
                    except ValidationError as ve:
                        logger.error(
                            "Schema validation failed for model=%s attempt=%s\n"
                            "Raw text:\n%s\nError: %s",
                            selected_model,
                            attempt + 1,
                            text,
                            ve,
                        )
                        # fall-through: return raw text below

                return text

            except Exception as e:
                logger.warning(f"LLM error (model={selected_model}, attempt={attempt + 1}): {e}")

                # After 2 failures, downgrade model if not already
                if attempt == 1 and selected_model != self.fallback_model:
                    logger.warning("Downgrading model after repeated failures.")
                    selected_model = self.fallback_model

                # Final fallback: Local
                if attempt == MAX_RETRIES - 1:
                    return local_llm_fallback(prompt)

                time.sleep(1.2 * (attempt + 1))

        return "Model error."


# Singleton
llm = LLMClient()
