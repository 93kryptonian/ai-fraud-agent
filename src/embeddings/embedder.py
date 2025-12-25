# import os
# import time
# from typing import List, Union
# from dotenv import load_dotenv
# from pathlib import Path
# from openai import OpenAI

# # ------------------------------------------------------------
# # LOAD .env RELIABLY
# # ------------------------------------------------------------
# env_path = Path(__file__).resolve().parents[2] / ".env"
# load_dotenv(env_path, override=True)

# # ------------------------------------------------------------
# # CONFIG
# # ------------------------------------------------------------
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# EMBED_MODEL = "text-embedding-3-small"  
# MAX_RETRIES = 5
# MAX_CHARS = 7000

# # client = get_openai_client()
# _openai_client = None

# def get_openai_client():
#     global _openai_client
#     if _openai_client is None:
#         if not OPENAI_API_KEY:
#             raise RuntimeError("OPENAI_API_KEY not set")
#         _openai_client = OpenAI(api_key=OPENAI_API_KEY)
#     return _openai_client

# # ------------------------------------------------------------
# # CLIENT
# # ------------------------------------------------------------
# # openai_client = OpenAI(api_key=OPENAI_API_KEY)
# openai_client = get_openai_client()



# # ------------------------------------------------------------
# # CLEAN TEXT
# # ------------------------------------------------------------
# def _clean_text(text: str) -> str:
#     if not text:
#         return ""
#     text = text.replace("\n", " ").replace("\t", " ").strip()
#     return text[:MAX_CHARS]


# # ------------------------------------------------------------
# # EMBEDDING MODEL (OpenAI)
# # ------------------------------------------------------------
# class OpenAIEmbeddingModel:

#     def embed_one(self, text: str) -> List[float]:
#         text = _clean_text(text)
#         for attempt in range(MAX_RETRIES):
#             try:
#                 resp = openai_client.embeddings.create(
#                     model=EMBED_MODEL,
#                     input=text
#                 )
#                 return resp.data[0].embedding
#             except Exception:
#                 if attempt == MAX_RETRIES - 1:
#                     raise
#                 time.sleep(1.2 * (attempt + 1))

#     def embed_batch(self, texts: List[str]) -> List[List[float]]:
#         cleaned = [_clean_text(t) for t in texts]
#         for attempt in range(MAX_RETRIES):
#             try:
#                 resp = openai_client.embeddings.create(
#                     model=EMBED_MODEL,
#                     input=cleaned
#                 )
#                 return [d.embedding for d in resp.data]
#             except Exception:
#                 if attempt == MAX_RETRIES - 1:
#                     raise
#                 time.sleep(1.2 * (attempt + 1))

#     def embed(self, data: Union[str, List[str]]):
#         if isinstance(data, str):
#             return self.embed_one(data)
#         return self.embed_batch(data)


# # ------------------------------------------------------------
# # EXPORTS
# # ------------------------------------------------------------
# embedding_model = OpenAIEmbeddingModel()
# chat_model = openai_client


# src/embeddings/embedder.py

import os
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ======================================================
# CONFIG
# ======================================================
EMBEDDINGS_ENABLED = os.getenv("EMBEDDINGS_ENABLED", "true").lower() == "true"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ======================================================
# LAZY CLIENTS (CRITICAL)
# ======================================================
_openai_client = None


def _get_openai_client():
    """
    Lazily create OpenAI client.
    Safe at import time, validated at runtime.
    """
    global _openai_client

    if _openai_client is not None:
        return _openai_client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    from openai import OpenAI  # local import = CI safe
    _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# ======================================================
# PUBLIC API
# ======================================================
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.

    CI-safe behavior:
    - If EMBEDDINGS_ENABLED=false → returns empty vectors
    - No OpenAI usage at import
    """

    if not EMBEDDINGS_ENABLED:
        logger.warning("[embedder] Embeddings disabled — returning empty vectors")
        return [[0.0] * 384 for _ in texts]  # stable dummy vectors

    if not texts:
        return []

    client = _get_openai_client()

    try:
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        embeddings = [item.embedding for item in resp.data]

        logger.info(
            f"[embedder] Generated {len(embeddings)} embeddings"
        )
        return embeddings

    except Exception as e:
        logger.error(f"[embedder] Embedding failed: {e}")
        return [[0.0] * 384 for _ in texts]


# ======================================================
# SINGLE-TEXT HELPER (OPTIONAL)
# ======================================================
def embed_text(text: str) -> Optional[List[float]]:
    if not text:
        return None

    vectors = embed_texts([text])
    return vectors[0] if vectors else None

# ======================================================
# BACKWARD COMPATIBILITY (CRITICAL)
# ======================================================

# class _LazyEmbeddingModel:
#     """
#     Backward-compatible proxy so existing code that imports
#     `embedding_model` does not break.

#     Embeddings are generated lazily and CI-safe.
#     """

#     def embed_documents(self, texts):
#         return embed_texts(texts)

#     def embed_query(self, text):
#         vec = embed_text(text)
#         return vec or []

class _LazyEmbeddingModel:
    """
    Backward-compatible proxy for legacy embedding_model usage.
    CI-safe and lazy.
    """

    def embed_documents(self, texts):
        return embed_texts(texts)

    def embed_query(self, text):
        vec = embed_text(text)
        return vec or []

    def embed_one(self, text):
        """
        Legacy method expected by ranking.py
        """
        vec = embed_text(text)
        return vec or []


# Legacy symbol expected by scoring / ranking
embedding_model = _LazyEmbeddingModel()
