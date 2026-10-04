import os

# The app's per-IP rate limiter is per-process and shared by every
# TestClient request in the run; keep it out of the way of the suite.
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "100000")

# Hermetic by default. Several modules call load_dotenv() at import, so a
# developer's .env would otherwise hand real credentials to the suite and let
# ordinary tests reach real services (one test ran real SQL against the
# production database this way; CI has no .env, so it never showed up there).
# Blank values are not overridden by load_dotenv(), so setting them here, before
# any test module imports the app, keeps every test offline. Opt in to live
# services explicitly:  RUN_LIVE_TESTS=1 pytest -m llm
CREDENTIAL_VARS = (
    "OPENAI_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_DB_URL",
)

if os.environ.get("RUN_LIVE_TESTS") != "1":
    for _var in CREDENTIAL_VARS:
        os.environ[_var] = ""
