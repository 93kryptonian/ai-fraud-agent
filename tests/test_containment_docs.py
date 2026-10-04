"""The failure-containment contract's knobs must match the code."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

KNOBS = {
    "LLM_TIMEOUT_SECONDS": ("src/llm/llm_client.py", "30"),
    "EMBEDDING_TIMEOUT_SECONDS": ("src/embeddings/embedder.py", "10"),
    "SUPABASE_TIMEOUT_SECONDS": ("src/db/supabase_client.py", "15"),
    "SQL_TIMEOUT_MS": ("src/db/supabase_client.py", "15000"),
}


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_documented_defaults_match_the_code():
    for name, (module, documented) in KNOBS.items():
        m = re.search(rf'os\.getenv\("{name}",\s*"([\d.]+)"\)', _read(module))
        assert m, f"{name} not read from the environment in {module}"
        assert m.group(1) == documented, f"{name}: code default {m.group(1)} vs documented {documented}"
        for doc in ("docs/failure_modes.md", "docs/operations.md"):
            assert re.search(rf"`{name}`\s*\|\s*`{documented}`", _read(doc)), f"{doc} must document {name}={documented}"


def test_documented_worst_case_matches_the_computed_bound():
    import src.llm.llm_client as llm

    assert llm.MAX_RETRIES == 4 and llm.RETRY_BACKOFF_SECONDS == 1.2
    # defaults: 4 x 30 + 1.2 x (1 + 2 + 3)
    assert 4 * 30 + 1.2 * (1 + 2 + 3) == 127.2
    assert "**127.2 s**" in _read("docs/failure_modes.md")
