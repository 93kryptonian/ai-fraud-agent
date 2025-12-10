import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

CSV_PATH = os.getenv("FRAUD_CSV_PATH", "data/raw/fraudTrain.csv")
DB_URL = os.getenv("SUPABASE_DB_URL")

TABLE_NAME = "fraud_transactions"
CHUNK = 50000


# ---------------------------------------------
# Ensure table exists
# ---------------------------------------------
CREATE_SQL = f"""
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


# ---------------------------------------------
# Batch insertion
# ---------------------------------------------
def batch_insert(conn, df):
    df = df.where(pd.notnull(df), None)  # NaN → None

    cols = [
        "trans_date_trans_time","cc_num","merchant","category","amt",
        "gender","city","state","zip","lat","long","city_pop","dob",
        "trans_num","unix_time","merch_lat","merch_long","isFraud"
    ]

    rows = [tuple(row) for row in df[cols].to_numpy()]

    sql = f"""
        INSERT INTO {TABLE_NAME} ({",".join(cols)})
        VALUES %s;
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()


# ---------------------------------------------
# Main ingestion
# ---------------------------------------------
def run_ingest():
    if not Path(CSV_PATH).exists():
        logger.error(f"CSV not found: {CSV_PATH}")
        sys.exit(1)

    logger.info("Connecting to Supabase Postgres (pooler)...")
    conn = psycopg2.connect(DB_URL)

    logger.info("Ensuring table exists...")
    with conn.cursor() as cur:
        cur.execute(CREATE_SQL)
    conn.commit()

    logger.info("Starting ingestion...")
    for i, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNK)):
        logger.info(f"Chunk {i+1} loaded")

        # Drop useless fields
        chunk = chunk.drop(columns=["first", "last", "street", "job", "Unnamed: 0"], errors="ignore")

        # Convert timestamps
        chunk["trans_date_trans_time"] = pd.to_datetime(chunk["trans_date_trans_time"], errors="coerce")
        chunk["dob"] = pd.to_datetime(chunk["dob"], errors="coerce")

        # Normalize boolean
        chunk = chunk.rename(columns={"is_fraud": "isFraud"})
        chunk["isFraud"] = chunk["isFraud"].astype(bool)

        logger.info(f"Inserting chunk {i+1} ({len(chunk)} rows)...")
        batch_insert(conn, chunk)

    conn.close()
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    logger.info("=== Ingestion start ===")
    run_ingest()
    logger.info("=== Ingestion done ===")
