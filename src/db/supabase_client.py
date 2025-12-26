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
from typing import Optional

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

# =============================================================================
# LAZY SUPABASE CLIENT (SINGLETON)
# =============================================================================

_supabase_client: Optional[Client] = None


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

    logger.info("[db] Initializing Supabase client")
    _supabase_client = create_client(
        SUPABASE_URL,
        SUPABASE_ANON_KEY,
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
