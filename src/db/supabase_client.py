# src/db/supabase_client.py

import os
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client

from src.utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()

# =====================================================
# ENV (SAFE TO READ AT IMPORT)
# =====================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# =====================================================
# LAZY CLIENT (CRITICAL FIX)
# =====================================================
_supabase_client: Optional[Client] = None


def get_supabase() -> Client:
    """
    Lazily create Supabase client.

    - SAFE during import
    - Raises ONLY when actually used
    - CI-friendly
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError(
            "Supabase is not configured. "
            "Set SUPABASE_URL and SUPABASE_ANON_KEY."
        )

    logger.info("Initializing Supabase client")
    _supabase_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _supabase_client


# =====================================================
# OPTIONAL DB WRAPPER (IF YOU USE IT)
# =====================================================
class DB:
    """
    Thin wrapper for DB access.
    Instantiated only when explicitly used.
    """

    def __init__(self):
        self.client = get_supabase()

    def table(self, name: str):
        return self.client.table(name)
