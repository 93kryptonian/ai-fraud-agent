"""
Explicit deadlines on external calls (Part 5).

API requests now run in worker threads, so a hung upstream no longer freezes
the event loop, but it would still pin a thread until the library default
(600 s for the OpenAI SDK, 120 s for PostgREST). These tests keep every external
call bounded, with ONE retry policy so the worst case is knowable.
"""

import json
import logging
import socket
import threading
import time

import pytest


@pytest.fixture(autouse=True)
def _isolate():
    from src.observability import context as context_module
    from src.observability import cost as cost_module

    t1 = context_module._request_id.set(None)
    t2 = cost_module._request_cost.set(cost_module.RequestCost())
    try:
        yield
    finally:
        cost_module._request_cost.reset(t2)
        context_module._request_id.reset(t1)


# =============================================================================
# Configuration reaches the clients
# =============================================================================

def test_default_deadlines_are_far_below_the_library_defaults():
    import src.db.supabase_client as db
    import src.embeddings.embedder as embedder
    import src.llm.llm_client as llm

    assert 0 < llm.LLM_TIMEOUT_SECONDS < 600          # OpenAI SDK default read timeout
    assert 0 < embedder.EMBEDDING_TIMEOUT_SECONDS < 600
    assert 0 < db.SUPABASE_TIMEOUT_SECONDS < 120      # PostgREST client default


def test_llm_client_gets_a_deadline_and_no_hidden_sdk_retries(monkeypatch):
    import openai

    import src.llm.llm_client as llm

    seen = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "x")
    monkeypatch.setattr(llm, "_openai_client", None)
    monkeypatch.setattr(llm, "LLM_TIMEOUT_SECONDS", 12.5)

    llm.get_openai_client()

    assert seen["timeout"] == 12.5
    assert seen["max_retries"] == 0      # LLMClient.run() is the only retry policy


def test_embedding_client_gets_a_deadline(monkeypatch):
    import openai

    import src.embeddings.embedder as embedder

    seen = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(embedder, "_openai_client", None)
    monkeypatch.setattr(embedder, "EMBEDDING_TIMEOUT_SECONDS", 4.0)

    embedder._get_openai_client()

    assert seen["timeout"] == 4.0


def test_supabase_client_gets_an_http_deadline(monkeypatch):
    import src.db.supabase_client as db

    captured = {}

    def fake_create_client(url, key, options=None):
        captured["options"] = options
        return object()

    monkeypatch.setattr(db, "create_client", fake_create_client)
    monkeypatch.setattr(db, "SUPABASE_URL", "https://example.invalid")
    monkeypatch.setattr(db, "SUPABASE_ANON_KEY", "anon")
    monkeypatch.setattr(db, "SUPABASE_TIMEOUT_SECONDS", 3.0)
    monkeypatch.setattr(db, "_supabase_client", None)

    db.get_supabase()

    assert captured["options"].postgrest_client_timeout == 3.0


# =============================================================================
# One retry policy, and a knowable worst case
# =============================================================================

def test_worst_case_call_duration_is_the_documented_bound(monkeypatch):
    import src.llm.llm_client as llm

    monkeypatch.setattr(llm, "LLM_TIMEOUT_SECONDS", 30.0)
    monkeypatch.setattr(llm, "MAX_RETRIES", 4)
    monkeypatch.setattr(llm, "RETRY_BACKOFF_SECONDS", 1.2)

    # 4 attempts x 30 s, plus backoff only BETWEEN attempts: 1.2 * (1 + 2 + 3)
    assert llm.llm_call_worst_case_seconds() == pytest.approx(4 * 30 + 7.2)


def test_there_is_no_backoff_sleep_after_the_final_attempt(monkeypatch):
    import src.llm.llm_client as llm

    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(llm, "MAX_RETRIES", 4)

    class AlwaysFails:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise ConnectionError("down")

    monkeypatch.setattr(llm, "get_openai_client", lambda: AlwaysFails)

    with pytest.raises(llm.LLMExhaustedRetriesError):
        llm.LLMClient().run("hi", purpose="rag_answer")

    assert sleeps == pytest.approx([1.2, 2.4, 3.6])   # 3 sleeps for 4 attempts


# =============================================================================
# The real SDK against a hung upstream (a local tarpit)
# =============================================================================

class _Tarpit:
    """Accepts connections and never replies: a hung upstream."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.accepted = 0
        self._held = []
        self._stop = False
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        self.sock.settimeout(0.1)
        while not self._stop:
            try:
                conn, _ = self.sock.accept()
            except (socket.timeout, OSError):
                continue
            self.accepted += 1
            self._held.append(conn)

    def close(self):
        self._stop = True
        for c in self._held:
            c.close()
        self.sock.close()


def test_a_hung_upstream_fails_at_the_deadline_with_no_hidden_retries(monkeypatch):
    import src.llm.llm_client as llm

    tarpit = _Tarpit()
    records = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(json.loads(record.getMessage()))

    handler = _H()
    events_logger = logging.getLogger("observability.events")
    events_logger.addHandler(handler)
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{tarpit.port}/v1")
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(llm, "_openai_client", None)
    monkeypatch.setattr(llm, "LLM_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(llm, "MAX_RETRIES", 2)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)

    started = time.perf_counter()
    try:
        with pytest.raises(llm.LLMExhaustedRetriesError) as info:
            llm.LLMClient().run("hello", purpose="rag_answer")
        elapsed = time.perf_counter() - started
    finally:
        events_logger.removeHandler(handler)
        tarpit.close()

    assert info.value.last_error_type == "APITimeoutError"
    assert elapsed < 3.0, f"a hung upstream held the call for {elapsed:.1f}s (deadline is 0.3s x 2 attempts)"
    assert elapsed >= 0.5                      # it really waited for the deadline, not an instant failure
    assert tarpit.accepted <= 2, "hidden SDK retries opened extra connections"

    (failed,) = [e for e in records if e["event"] == "llm.failed"]
    assert failed["metadata"]["error_type"] == "APITimeoutError"
    assert failed["metadata"]["retry_count"] == 1      # MAX_RETRIES - 1: what telemetry reports is what happened
