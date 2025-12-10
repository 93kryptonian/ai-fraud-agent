


"""
PAGE-BASED PDF INGESTION PIPELINE
----------------------------------

This ingestion pipeline extracts **full pages**, not chunks.
This avoids complex RPC signatures and works perfectly with
SupabaseVectorStore + similarity_search(filter/k).

Benefits:
- No chunk boundary issues
- Each page = 1 record, perfect for analytics/legal PDFs
- Very fast (only ~200 total embeddings)
- Drop-in compatible with your current retriever

Usage:
    python -m src.ingest.ingest_docs
"""

import os
import hashlib
import concurrent.futures
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

from src.db.supabase_client import DB
from src.embeddings.embedder import embedding_model
from src.utils.logger import get_logger

# ======================================================================
# INIT
# ======================================================================

load_dotenv()
logger = get_logger(__name__)

DATA_DIR = "data/raw"
EMB_BATCH_SIZE = 32   


# ======================================================================
# UTILITIES
# ======================================================================

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_insert(table: str, rows: list):
    """Insert rows safely into Supabase (batched by the DB wrapper)."""
    if not rows:
        return

    try:
        return DB.insert(table, rows)
    except Exception:
        logger.exception(f"[ingest] DB.insert failed (table={table}, rows={len(rows)})")
        raise


# ======================================================================
# PAGE-LEVEL PDF EXTRACTION
# ======================================================================

def load_pdf_pages(path: str):
    """
    Load PDF and extract raw pages.
    Each page becomes ONE RAG unit (recommended for analytics/financial PDFs).
    """
    loader = PyPDFLoader(path)
    pages = loader.load()

    results = []
    for idx, page in enumerate(pages):
        text = page.page_content.strip()
        if not text:
            continue

        results.append({
            "page": idx + 1,  # real page number (1-indexed)
            "content": text,
            "hash": sha256_text(text),
        })

    logger.info(f"[ingest] Extracted {len(results)} pages from {os.path.basename(path)}")
    return results


# ======================================================================
# PROCESS ONE PDF
# ======================================================================

def process_pdf(pdf_path: str):
    filename = Path(pdf_path).stem
    logger.info(f"[ingest] Processing PDF: {filename}")

    pages = load_pdf_pages(pdf_path)
    total_pages = len(pages)

    # -------------------------------------------------------------
    # 1) INSERT DOCUMENTS (each page = 1 row)
    # -------------------------------------------------------------
    doc_rows = []
    for p in pages:
        doc_rows.append({
            "id": os.popen("uuidgen").read().strip(),
            "source_name": filename,
            "page": p["page"],
            "content": p["content"],
            "hash": p["hash"],
            "created_at": datetime.utcnow().isoformat(),
        })

    safe_insert("documents", doc_rows)

    # Map hash → id for embeddings
    hash_to_doc_id = {r["hash"]: r["id"] for r in doc_rows}

    # -------------------------------------------------------------
    # 2) EMBEDDINGS
    # -------------------------------------------------------------
    logger.info(f"[ingest] Embedding {total_pages} pages for {filename}...")

    emb_rows = []
    batch_texts = []
    batch_hashes = []

    for p in pages:
        batch_texts.append(p["content"])
        batch_hashes.append(p["hash"])

        # batch embed
        if len(batch_texts) == EMB_BATCH_SIZE:
            vecs = embedding_model.embed(batch_texts)
            for h, v in zip(batch_hashes, vecs):
                emb_rows.append({
                    "id": os.popen("uuidgen").read().strip(),
                    "document_id": hash_to_doc_id[h],
                    "embedding": v,
                    "created_at": datetime.utcnow().isoformat(),
                })
            batch_texts.clear()
            batch_hashes.clear()

    # final batch
    if batch_texts:
        vecs = embedding_model.embed(batch_texts)
        for h, v in zip(batch_hashes, vecs):
            emb_rows.append({
                "id": os.popen("uuidgen").read().strip(),
                "document_id": hash_to_doc_id[h],
                "embedding": v,
                "created_at": datetime.utcnow().isoformat(),
            })

    safe_insert("document_embeddings", emb_rows)

    logger.info(f"[ingest] Finished {filename} (pages={total_pages})")


# ======================================================================
# RUN ALL PDFs
# ======================================================================

def main():
    pdfs = [str(p) for p in Path(DATA_DIR).glob("*.pdf")]

    if not pdfs:
        logger.error("[ingest] No PDFs found!")
        return

    logger.info(f"[ingest] Found {len(pdfs)} PDFs to process")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as exe:
        futs = [exe.submit(process_pdf, pdf) for pdf in pdfs]
        for fut in futs:
            fut.result()

    logger.info("[ingest] All PDFs processed successfully.")


if __name__ == "__main__":
    main()
