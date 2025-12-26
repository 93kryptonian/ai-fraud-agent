"""
PAGE-BASED PDF INGESTION PIPELINE
----------------------------------

This pipeline ingests PDFs at the **page level** (1 page = 1 record).

Why page-based ingestion:
- Avoids chunk-boundary ambiguity
- Ideal for legal, regulatory, and financial PDFs
- Simple, predictable retrieval semantics
- Fast embedding (hundreds of pages, not thousands of chunks)
- Fully compatible with Supabase vector search (top-k + filters)

Usage:
    python -m src.ingest.ingest_docs
"""

import os
import hashlib
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

from src.db.supabase_client import DB
from src.embeddings.embedder import embedding_model
from src.utils.logger import get_logger

# =============================================================================
# INITIALIZATION
# =============================================================================

load_dotenv()
logger = get_logger(__name__)

DATA_DIR = "data/raw"
EMB_BATCH_SIZE = 32

# =============================================================================
# UTILITIES
# =============================================================================

def sha256_text(text: str) -> str:
    """
    Deterministic content hash for deduplication and traceability.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_uuid() -> str:
    """
    Generate UUID using system utility (portable & collision-safe).
    """
    return os.popen("uuidgen").read().strip()


def safe_insert(table: str, rows: List[Dict]):
    """
    Insert rows into Supabase with safety logging.
    """
    if not rows:
        return

    try:
        return DB.insert(table, rows)
    except Exception:
        logger.exception(
            f"[ingest] DB.insert failed | table={table} | rows={len(rows)}"
        )
        raise

# =============================================================================
# PAGE-LEVEL PDF EXTRACTION
# =============================================================================

def load_pdf_pages(path: str) -> List[Dict]:
    """
    Load a PDF and extract raw pages.

    Each page becomes ONE ingestion unit.
    """
    loader = PyPDFLoader(path)
    pages = loader.load()

    results: List[Dict] = []

    for idx, page in enumerate(pages):
        text = (page.page_content or "").strip()
        if not text:
            continue

        results.append({
            "page": idx + 1,   # 1-indexed page number
            "content": text,
            "hash": sha256_text(text),
        })

    logger.info(
        f"[ingest] Extracted {len(results)} pages from {os.path.basename(path)}"
    )
    return results

# =============================================================================
# PROCESS A SINGLE PDF
# =============================================================================

def process_pdf(pdf_path: str):
    """
    Ingest a single PDF:
    1) Extract pages
    2) Insert page records
    3) Generate and store embeddings
    """
    filename = Path(pdf_path).stem
    logger.info(f"[ingest] Processing PDF: {filename}")

    pages = load_pdf_pages(pdf_path)
    total_pages = len(pages)

    # -------------------------------------------------------------
    # 1) INSERT DOCUMENT RECORDS (page-level)
    # -------------------------------------------------------------
    document_rows = []

    for p in pages:
        document_rows.append({
            "id": generate_uuid(),
            "source_name": filename,
            "page": p["page"],
            "content": p["content"],
            "hash": p["hash"],
            "created_at": datetime.utcnow().isoformat(),
        })

    safe_insert("documents", document_rows)

    # Map content hash → document_id (stable & deterministic)
    hash_to_doc_id = {
        row["hash"]: row["id"] for row in document_rows
    }

    # -------------------------------------------------------------
    # 2) EMBEDDING GENERATION (BATCHED)
    # -------------------------------------------------------------
    logger.info(
        f"[ingest] Embedding {total_pages} pages for {filename}"
    )

    embedding_rows: List[Dict] = []
    batch_texts: List[str] = []
    batch_hashes: List[str] = []

    for p in pages:
        batch_texts.append(p["content"])
        batch_hashes.append(p["hash"])

        if len(batch_texts) == EMB_BATCH_SIZE:
            vectors = embedding_model.embed(batch_texts)

            for h, v in zip(batch_hashes, vectors):
                embedding_rows.append({
                    "id": generate_uuid(),
                    "document_id": hash_to_doc_id[h],
                    "embedding": v,
                    "created_at": datetime.utcnow().isoformat(),
                })

            batch_texts.clear()
            batch_hashes.clear()

    # Final partial batch
    if batch_texts:
        vectors = embedding_model.embed(batch_texts)
        for h, v in zip(batch_hashes, vectors):
            embedding_rows.append({
                "id": generate_uuid(),
                "document_id": hash_to_doc_id[h],
                "embedding": v,
                "created_at": datetime.utcnow().isoformat(),
            })

    safe_insert("document_embeddings", embedding_rows)

    logger.info(
        f"[ingest] Completed {filename} | pages={total_pages}"
    )

# =============================================================================
# RUN PIPELINE
# =============================================================================

def main():
    pdf_paths = [str(p) for p in Path(DATA_DIR).glob("*.pdf")]

    if not pdf_paths:
        logger.error("[ingest] No PDFs found to ingest")
        return

    logger.info(f"[ingest] Found {len(pdf_paths)} PDFs")

    # Controlled parallelism (I/O bound, API bound)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(process_pdf, pdf)
            for pdf in pdf_paths
        ]
        for f in futures:
            f.result()

    logger.info("[ingest] All PDFs ingested successfully")

if __name__ == "__main__":
    main()
