# src/db/supabase_client.py
"""
Supabase client utilities for an Enterprise Fraud Intelligence System.

Responsibilities:
- Lazy initialization of Supabase client
- CI-safe import behavior
- Centralized database access abstraction

Design principles:
- No network calls at import time
- Fail fast only when explicitly used
- Single shared client instance
"""

import os
import threading
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from supabase import create_client, Client

from src.utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()

# =============================================================================
# ENV CONFIGURATION (SAFE TO READ AT IMPORT)
# =============================================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
SQL_TIMEOUT_MS = int(os.getenv("SQL_TIMEOUT_MS", "15000"))
# HTTP deadline for PostgREST calls (the vector-search RPC). The client default
# is 120 s. The direct Postgres path is already bounded (connect_timeout=10,
# statement_timeout=SQL_TIMEOUT_MS).
SUPABASE_TIMEOUT_SECONDS = float(os.getenv("SUPABASE_TIMEOUT_SECONDS", "15"))

# =============================================================================
# LAZY SUPABASE CLIENT (SINGLETON)
# =============================================================================

_supabase_client: Optional[Client] = None
_CLIENT_LOCK = threading.Lock()


def _client_options():
    """Client options carrying the HTTP deadline (the class name differs by version)."""
    try:
        from supabase.lib.client_options import SyncClientOptions as Options  # newer releases
    except ImportError:
        from supabase.lib.client_options import ClientOptions as Options  # pinned release
    return Options(postgrest_client_timeout=SUPABASE_TIMEOUT_SECONDS)


def get_supabase() -> Client:
    """
    Lazily create and return a Supabase client.

    Guarantees:
    - No Supabase initialization during import
    - Raises configuration errors only when accessed
    - Safe for CI, tests, and local development
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError(
            "Supabase is not configured. "
            "Please set SUPABASE_URL and SUPABASE_ANON_KEY."
        )

    with _CLIENT_LOCK:
        if _supabase_client is None:  # another thread may have created it first
            logger.info("[db] Initializing Supabase client")
            _supabase_client = create_client(
                SUPABASE_URL,
                SUPABASE_ANON_KEY,
                options=_client_options(),
            )
        return _supabase_client

# =============================================================================
# OPTIONAL DB WRAPPER
# =============================================================================

class DB:
    """
    Thin database access wrapper.

    This exists to:
    - decouple business logic from Supabase client details
    - simplify mocking in tests
    - provide a clear extension point for future DB logic
    """

    def __init__(self):
        self.client = get_supabase()

    def table(self, name: str):
        """
        Access a Supabase table.
        """
        return self.client.table(name)

    @staticmethod
    def sql(query: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a read-only SQL query via direct Postgres connection.

        Safety:
        - Transaction is READ ONLY → DB rejects any DDL/DML, even if the
          SELECT-only string check upstream is bypassed
        - statement_timeout bounds runaway queries
        """
        if not SUPABASE_DB_URL:
            raise RuntimeError("SUPABASE_DB_URL is not set")

        import psycopg2  # local import → CI safe
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(SUPABASE_DB_URL, connect_timeout=10)
        try:
            conn.set_session(readonly=True)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"SET LOCAL statement_timeout = {SQL_TIMEOUT_MS}")
                cur.execute(query, params)
                rows = cur.fetchall() if cur.description else []
            conn.rollback()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    @staticmethod
    def insert(table: str, rows: List[Dict[str, Any]]):
        """
        Insert rows through Supabase (service role preferred, bypasses RLS).
        """
        if SUPABASE_SERVICE_ROLE_KEY and SUPABASE_URL:
            client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        else:
            client = get_supabase()
        return client.table(table).insert(rows).execute()
