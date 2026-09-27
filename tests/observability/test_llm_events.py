# tests/observability/test_llm_events.py
"""
Tests for M6.3: LLM instrumentation (src/llm/llm_client.py).

Covers the explicit hard invariant: exactly one terminal llm event per
LLMClient.run() call, with retry_count reflecting the whole retry loop
(attempts beyond the first — same convention on both the success and
failure paths), never one event per attempt. Plus purpose tagging,
token/cost metadata, and the fallback event on budget threshold.
"""

import json
import logging
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _reset_request_id_context():
    from src.observability import context as context_module

    token = context_module._request_id.set(None)
    try:
        yield
    finally:
        context_module._request_id.reset(token)


@pytest.fixture(autouse=True)
def _reset_request_cost():
    from src.observability import cost as cost_module

    token = cost_module._request_cost_usd.set(0.0)
    try:
        yield
    finally:
        cost_module._request_cost_usd.reset(token)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    import src.llm.llm_client as llm_client_module

    monkeypatch.setattr(llm_client_module.time, "sleep", lambda *_: None)


def _capture_event_records():
    records = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _ListHandler()
    logger = logging.getLogger("observability.events")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


@pytest.fixture
def event_records():
    yield from _capture_event_records()


def _parsed(records):
    return [json.loads(r.getMessage()) for r in records]


def _names(records):
    return [e["event"] for e in _parsed(records)]


def _fake_openai_succeeds(content="an answer", prompt_tokens=10, completion_tokens=5):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))


def _fake_openai_fails_then_succeeds(fail_times, **succeed_kwargs):
    calls = {"n": 0}
    response = _fake_openai_succeeds(**succeed_kwargs)

    def create(**kw):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise ConnectionError("transient failure")
        return response.chat.completions.create(**kw)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_llm_completed_carries_purpose_tokens_cost_and_zero_retries(event_records, monkeypatch):
    from src.llm.llm_client import LLMClient

    monkeypatch.setattr("src.llm.llm_client.get_openai_client", lambda: _fake_openai_succeeds())

    LLMClient().run("hello", purpose="rag_answer")

    events = _parsed(event_records)
    assert len(events) == 1
    evt = events[0]
    assert evt["event"] == "llm.completed"
    assert evt["metadata"]["purpose"] == "rag_answer"
    assert evt["metadata"]["prompt_tokens"] == 10
    assert evt["metadata"]["completion_tokens"] == 5
    assert evt["metadata"]["total_tokens"] == 15
    assert evt["metadata"]["estimated_cost_usd"] is not None
    assert evt["metadata"]["retry_count"] == 0
    assert isinstance(evt["duration_ms"], int)


def test_llm_completed_retry_count_reflects_prior_failed_attempts(event_records, monkeypatch):
    from src.llm.llm_client import LLMClient

    # Built once, outside the lambda: get_openai_client() is called fresh
    # on every retry attempt, so a lambda that constructs a new fake here
    # would reset its internal call counter every attempt too, never
    # actually accumulating failures across the loop.
    fake_client = _fake_openai_fails_then_succeeds(fail_times=2)
    monkeypatch.setattr("src.llm.llm_client.get_openai_client", lambda: fake_client)

    LLMClient().run("hello", purpose="rag_answer")

    events = _parsed(event_records)
    # Exactly ONE terminal event, not one per attempt — the hard invariant.
    assert len(events) == 1
    assert events[0]["event"] == "llm.completed"
    assert events[0]["metadata"]["retry_count"] == 2


def test_llm_failed_is_exactly_one_event_not_one_per_attempt(event_records, monkeypatch):
    from src.llm.llm_client import LLMClient, LLMExhaustedRetriesError

    def _always_fails():
        def create(**kw):
            raise ConnectionError("simulated network failure")
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr("src.llm.llm_client.get_openai_client", _always_fails)

    with pytest.raises(LLMExhaustedRetriesError):
        LLMClient().run("hello", purpose="rag_answer")

    events = _parsed(event_records)
    assert len(events) == 1  # not 4 — one terminal event for the whole retry loop
    evt = events[0]
    assert evt["event"] == "llm.failed"
    # 4 total attempts (MAX_RETRIES) means 3 retries beyond the first —
    # same "attempts beyond the first" convention as the success path's
    # retry_count == attempt - 1, checked above.
    assert evt["metadata"]["retry_count"] == 3
    assert evt["metadata"]["error_type"] == "ConnectionError"
    assert evt["metadata"]["purpose"] == "rag_answer"


def test_llm_fallback_event_on_budget_threshold(event_records, monkeypatch):
    import src.llm.llm_client as llm_client_module

    monkeypatch.setattr(llm_client_module, "SESSION_COST_USD", llm_client_module.MAX_COST_USD)
    monkeypatch.setattr(llm_client_module, "FALLBACK_MODEL", "a-different-fallback-model")
    client = llm_client_module.LLMClient()
    client.fallback_model = "a-different-fallback-model"
    monkeypatch.setattr("src.llm.llm_client.get_openai_client", lambda: _fake_openai_succeeds())

    client.run("hello", purpose="rag_answer")

    names = _names(event_records)
    assert "llm.fallback" in names
    fallback_evt = next(e for e in _parsed(event_records) if e["event"] == "llm.fallback")
    assert fallback_evt["metadata"]["to_model"] == "a-different-fallback-model"
    assert fallback_evt["metadata"]["reason"] == "budget_threshold"
