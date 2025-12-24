import os
from typing import Any, Dict, List, Optional

from supabase import create_client, Client
from dotenv import load_dotenv

# import psycopg2
# import psycopg2.extras
try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None


# =====================================================
# Load environment variables
# =====================================================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("Missing Supabase credentials in .env")


# =====================================================
# Initialize Supabase REST client
# =====================================================
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


# =====================================================
# PostgreSQL direct connection 
# =====================================================
def get_pg_conn():
    """
    Create a PostgreSQL connection to Supabase Cloud.
    """
    if not SUPABASE_DB_URL:
        raise RuntimeError("Missing SUPABASE_DB_URL in .env")
    return psycopg2.connect(SUPABASE_DB_URL)


# =====================================================
# Raw SQL execution helper
# =====================================================
def run_sql(query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
    """
    Execute raw SQL on Supabase Cloud via psycopg2.

    Returns:
        List of dict rows (RealDictCursor)
    """
    conn = get_pg_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)

            # SELECT queries return rows
            if cur.description:
                return cur.fetchall()

            # Non-select queries commit
            conn.commit()
            return []
    finally:
        conn.close()


# =====================================================
# Batch insert helper (for ingestion)
# =====================================================
def batch_insert(table: str, rows: List[Dict[str, Any]], batch_size: int = 500):
    """
    Inserts rows into Supabase using batches.

    Args:
        table (str): table name
        rows (list): rows as dict
        batch_size (int): batch upload size

    Returns:
        dict: status message
    """
    total = len(rows)
    for i in range(0, total, batch_size):
        chunk = rows[i : i + batch_size]

        resp = supabase.table(table).insert(chunk).execute()

        if hasattr(resp, "error") and resp.error:
            raise RuntimeError(f"Supabase insert error: {resp.error}")

    return {"status": "success", "rows": total}


# =====================================================
# Health Check
# =====================================================
def health() -> Dict[str, Any]:
    """
    Check Supabase REST availability and DB connectivity.
    """
    try:
        supabase.table("documents").select("id").limit(1).execute()
        return {"supabase": "ok"}
    except Exception as e:
        return {"supabase": "error", "details": str(e)}


# =====================================================
# High-Level DB Wrapper (singleton)
# =====================================================
class SupabaseDB:
    """
    High-level wrapper used across the whole project:

        from src.db.supabase_client import DB

        DB.sql(...)
        DB.insert(...)
        DB.vector(...)
        DB.ping()
    """

    @staticmethod
    def sql(query: str, params: Optional[tuple] = None):
        return run_sql(query, params)

    @staticmethod
    def insert(table: str, rows: List[Dict[str, Any]]):
        return batch_insert(table, rows)

    # @staticmethod
    # def vector(embedding: List[float], top_k: int = 5, source: Optional[str] = None):
    #     return vector_search(embedding, top_k, source)

    @staticmethod
    def ping():
        return health()


# Singleton instance
DB = SupabaseDB()