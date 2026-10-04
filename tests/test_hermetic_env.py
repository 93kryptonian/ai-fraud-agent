"""
The default test run must never hold real credentials (see tests/conftest.py).
Opt in to live services with RUN_LIVE_TESTS=1.
"""

import os

import pytest

from tests.conftest import CREDENTIAL_VARS

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") == "1", reason="live tests were requested explicitly"
)


@pytest.mark.parametrize("var", CREDENTIAL_VARS)
def test_credentials_are_blank_in_the_default_run(var):
    assert os.environ.get(var) == ""


def test_importing_the_app_does_not_pick_up_a_dotenv_file():
    """load_dotenv() must not override the blanks, even when a .env exists."""
    import src.db.supabase_client as db  # noqa: F401 (import triggers load_dotenv)
    import src.llm.llm_client as llm

    assert not db.SUPABASE_URL and not db.SUPABASE_DB_URL and not db.SUPABASE_ANON_KEY
    assert not llm.OPENAI_API_KEY
