"""
TABULAR INGESTION PIPELINE (FRAUD TRANSACTIONS)
-----------------------------------------------

This script ingests the fraud transactions CSV into Supabase Postgres
using chunked inserts for scalability and safety.

Design choices:
- Chunked ingestion (memory-safe for large CSVs)
- Explicit schema creation (idempotent)
- Server-side batch insert (execute_values)
- Minimal transformation (preserve raw data fidelity)

Usage:
    python -m src.ingest.ingest_tabular
"""

import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

from src.utils.logger import get_logger

# =============================================================================
# INITIALIZATION
# =============================================================================

load_dotenv()
logger = get_logger(__name__)

CSV_PATH = os.getenv("FRAUD_CSV_PATH", "data/raw/fraudTrain.csv")
DB_URL = os.getenv("SUPABASE_DB_URL")

TABLE_NAME = "fraud_transactions"
CHUNK_SIZE = 50_000

# =============================================================================
# TABLE DEFINITION (IDEMPOTENT)
# =============================================================================

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    trans_date_trans_time TIMESTAMPTZ,
    cc_num BIGINT,
    merchant TEXT,
    category TEXT,
    amt DOUBLE PRECISION,
    gender TEXT,
    city TEXT,
    state TEXT,
    zip INTEGER,
    lat DOUBLE PRECISION,
    long DOUBLE PRECISION,
    city_pop INTEGER,
    dob TIMESTAMPTZ,
    trans_num TEXT,
    unix_time BIGINT,
    merch_lat DOUBLE PRECISION,
    merch_long DOUBLE PRECISION,
    isFraud BOOLEAN
);
"""

# Ordered column list (authoritative)
COLUMNS: List[str] = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "gender",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "isFraud",
]

# =============================================================================
# BATCH INSERT
# =============================================================================

def batch_insert(conn, df: pd.DataFrame):
    """
    Insert a dataframe chunk into Postgres using execute_values.
    """
    # Normalize NaN → NULL
    df = df.where(pd.notnull(df), None)

    rows = [tuple(row) for row in df[COLUMNS].to_numpy()]

    insert_sql = f"""
        INSERT INTO {TABLE_NAME} ({",".join(COLUMNS)})
        VALUES %s;
    """

    with conn.cursor() as cur:
        execute_values(cur, insert_sql, rows)

    conn.commit()

# =============================================================================
# INGESTION LOGIC
# =============================================================================

def run_ingest():
    """
    Main ingestion routine.
    """
    if not Path(CSV_PATH).exists():
        logger.error(f"[ingest] CSV not found: {CSV_PATH}")
        sys.exit(1)

    if not DB_URL:
        logger.error("[ingest] SUPABASE_DB_URL is not set")
        sys.exit(1)

    logger.info("[ingest] Connecting to Supabase Postgres")
    conn = psycopg2.connect(DB_URL)

    try:
        # ---------------------------------------------------------
        # Ensure table exists
        # ---------------------------------------------------------
        logger.info("[ingest] Ensuring target table exists")
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()

        # ---------------------------------------------------------
        # Chunked CSV ingestion
        # ---------------------------------------------------------
        logger.info("[ingest] Starting CSV ingestion")

        for idx, chunk in enumerate(
            pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE)
        ):
            logger.info(
                f"[ingest] Loaded chunk {idx + 1} | rows={len(chunk)}"
            )

            # Drop irrelevant fields (safe if missing)
            chunk = chunk.drop(
                columns=["first", "last", "street", "job", "Unnamed: 0"],
                errors="ignore",
            )

            # Timestamp normalization
            chunk["trans_date_trans_time"] = pd.to_datetime(
                chunk["trans_date_trans_time"], errors="coerce"
            )
            chunk["dob"] = pd.to_datetime(
                chunk["dob"], errors="coerce"
            )

            # Normalize fraud label
            chunk = chunk.rename(columns={"is_fraud": "isFraud"})
            chunk["isFraud"] = chunk["isFraud"].astype(bool)

            logger.info(
                f"[ingest] Inserting chunk {idx + 1}"
            )
            batch_insert(conn, chunk)

        logger.info("[ingest] Ingestion completed successfully")

    finally:
        conn.close()
        logger.info("[ingest] Database connection closed")

# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    logger.info("=== Fraud transactions ingestion started ===")
    run_ingest()
    logger.info("=== Fraud transactions ingestion finished ===")
