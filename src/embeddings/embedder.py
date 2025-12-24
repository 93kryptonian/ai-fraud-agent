import os
import time
from typing import List, Union
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

# ------------------------------------------------------------
# LOAD .env RELIABLY
# ------------------------------------------------------------
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path, override=True)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"  
MAX_RETRIES = 5
MAX_CHARS = 7000

# client = get_openai_client()
_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

# ------------------------------------------------------------
# CLIENT
# ------------------------------------------------------------
# openai_client = OpenAI(api_key=OPENAI_API_KEY)
openai_client = get_openai_client()



# ------------------------------------------------------------
# CLEAN TEXT
# ------------------------------------------------------------
def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\t", " ").strip()
    return text[:MAX_CHARS]


# ------------------------------------------------------------
# EMBEDDING MODEL (OpenAI)
# ------------------------------------------------------------
class OpenAIEmbeddingModel:

    def embed_one(self, text: str) -> List[float]:
        text = _clean_text(text)
        for attempt in range(MAX_RETRIES):
            try:
                resp = openai_client.embeddings.create(
                    model=EMBED_MODEL,
                    input=text
                )
                return resp.data[0].embedding
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(1.2 * (attempt + 1))

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        cleaned = [_clean_text(t) for t in texts]
        for attempt in range(MAX_RETRIES):
            try:
                resp = openai_client.embeddings.create(
                    model=EMBED_MODEL,
                    input=cleaned
                )
                return [d.embedding for d in resp.data]
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(1.2 * (attempt + 1))

    def embed(self, data: Union[str, List[str]]):
        if isinstance(data, str):
            return self.embed_one(data)
        return self.embed_batch(data)


# ------------------------------------------------------------
# EXPORTS
# ------------------------------------------------------------
embedding_model = OpenAIEmbeddingModel()
chat_model = openai_client
