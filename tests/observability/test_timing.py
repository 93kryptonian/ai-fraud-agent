# tests/observability/test_timing.py
"""
Tests for src/observability/timing.py and the M5 wiring into
api/routers.py and src/orchestrator.py.

Covers the locked M5 contract:
1. elapsed_timer() computes deterministic durations off a mocked clock,
   never a weak `> 0` check.
2. observe_step() emits "{step}.completed" on success, "{step}.failed" on
   exception — and always re-raises, never swallows.
3. duration_ms is real (int, computed) on every terminal event this
   milestone times, and stays None on request.started (a marker, not a
   span) and on anything not yet instrumented.
4. Parent (request) duration contains child (guardrails) duration for the
   current sequential pipeline — a construction fact, not a universal
   invariant (see timing.py docstring; concurrency would break equality
   assumptions, not this >= check).
5. Exactly one request.completed per request, including on an unhandled
   exception — status="error", duration present, exception still
   propagates (observability never swallows it).
6. language_detection/intent only ever appear on the /query route (the
   only one that goes through the orchestrator), never on /rag or
   /analytics.
7. business_outcome is explicitly NOT introduced in M5 (deferred to M6,
   per the design review) — locking that decision with a test so a future
   accidental addition shows up as a diff here, not a silent scope creep.

(A former item 8 here locked "retrieval/ranking/LLM/analytics internals
remain uninstrumented" as an M5 scope boundary. M6 was built explicitly to
add that instrumentation — see tests/observability/test_m6_pipeline.py —
so that test was removed rather than left permanently failing; the M6
tests lock the next boundary instead.)
"""

import json
import logging
import time

import pytest

from src.observability.events import emit_event
from src.observability.timing import elapsed_timer, observe_step


@pytest.fixture(autouse=True)
def _reset_request_id_context():
    from src.observability import context as context_module

    token = context_module._request_id.set(None)
    try:
        yield
    finally:
        context_module._request_id.reset(token)


@pytest.fixture(autouse=True)
def _fast_llm(monkeypatch):
    monkeypatch.setattr(
        "src.llm.llm_client.LLMClient.run",
        lambda self, *a, **kw: "LLM failed after retries.",
    )
    monkeypatch.setenv("RETRIEVER_ENABLED", "false")
    monkeypatch.setenv("EMBEDDINGS_ENABLED", "false")


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


# =============================================================================
# 1. Deterministic clock, not `> 0`
# =============================================================================

def test_elapsed_timer_is_deterministic_against_a_mocked_clock(monkeypatch):
    ticks = iter([100.000, 100.042])  # start, then read
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

    with elapsed_timer() as elapsed:
        pass

    assert elapsed() == 42


# =============================================================================
# 2. observe_step: success / failure, always re-raises
# =============================================================================

def test_observe_step_emits_completed_on_success(event_records, monkeypatch):
    ticks = iter([100.000, 100.010])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

    with observe_step("intent"):
        pass

    events = _parsed(event_records)
    assert len(events) == 1
    assert events[0]["event"] == "intent.completed"
    assert events[0]["status"] == "success"
    assert events[0]["duration_ms"] == 10


def test_observe_step_emits_failed_and_reraises_on_exception(event_records, monkeypatch):
    ticks = iter([100.000, 100.005])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))

    class _Boom(ValueError):
        pass

    with pytest.raises(_Boom):
        with observe_step("intent"):
            raise _Boom("simulated failure")

    events = _parsed(event_records)
    assert len(events) == 1
    assert events[0]["event"] == "intent.failed"
    assert events[0]["status"] == "failure"
    assert events[0]["duration_ms"] == 5


# =============================================================================
# 3 & 4. Real duration on terminal events, None on markers, containment
# =============================================================================

def test_request_started_has_no_duration_but_completed_does(event_records):
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    client.post("/query", json={"query": "what is the fraud rate trend"})

    events = _parsed(event_records)
    started = next(e for e in events if e["event"] == "request.started")
    completed = next(e for e in events if e["event"] == "request.completed")
    guardrails = next(e for e in events if e["event"] == "guardrails.completed")

    assert started["duration_ms"] is None
    assert isinstance(completed["duration_ms"], int)
    assert isinstance(guardrails["duration_ms"], int)

    # Containment for this (sequential) pipeline: the request span contains
    # the guardrails span. Not a universal invariant under concurrency —
    # see timing.py.
    assert completed["duration_ms"] >= guardrails["duration_ms"]


# =============================================================================
# 5. Exactly one request.completed, including on an unhandled exception
# =============================================================================

def test_unhandled_exception_still_emits_request_completed_and_reraises(event_records, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated orchestrator crash")

    monkeypatch.setattr("api.routers.run_query", _boom)

    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    with pytest.raises(RuntimeError, match="simulated orchestrator crash"):
        client.post("/query", json={"query": "what is the fraud rate trend"})

    events = _parsed(event_records)
    completed_events = [e for e in events if e["event"] == "request.completed"]
    assert len(completed_events) == 1  # exactly one, not zero, not two

    evt = completed_events[0]
    assert evt["status"] == "error"
    assert isinstance(evt["duration_ms"], int)
    assert evt["metadata"]["error_type"] == "RuntimeError"
    # Privacy/minimalism: no raw exception message, just the type name.
    assert "simulated orchestrator crash" not in json.dumps(evt)


# =============================================================================
# 6. language_detection/intent only appear on /query
# =============================================================================

def test_orchestrator_stages_only_appear_on_query_route(event_records):
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    client.post("/rag", json={"query": "what is the fraud rate trend"})
    client.post("/analytics", json={"query": "what is the fraud rate trend"})

    names = {e["event"] for e in _parsed(event_records)}
    assert "language_detection.completed" not in names
    assert "language_detection.failed" not in names
    assert "intent.completed" not in names
    assert "intent.failed" not in names


# =============================================================================
# 7. business_outcome explicitly deferred to M6 — lock the decision
# =============================================================================

def test_business_outcome_field_not_introduced_in_m5(event_records):
    """
    Analytics fails internally (no SUPABASE_DB_URL in tests) while the
    endpoint still completes normally — the exact case M4 discovered.
    M5 deliberately does NOT add a business_outcome field to distinguish
    it; that's M6's job, once there's an actual business-outcome contract
    rather than an error-envelope guess. This test locks that decision so
    an accidental addition shows up as a diff here.
    """
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    client.post("/query", json={"query": "what is the fraud rate trend"})

    events = _parsed(event_records)
    completed = next(e for e in events if e["event"] == "request.completed")
    assert completed["status"] == "success"
    assert "business_outcome" not in completed["metadata"]


# =============================================================================
# 8. Still uninstrumented: retrieval, ranking, LLM, analytics internals
# =============================================================================

# The M5 version of this test asserted retrieval/ranking/llm/analytics
# events never appeared. M6 (tests/observability/test_m6_pipeline.py) was
# built specifically to add them, so that assertion is gone rather than
# left permanently failing — see the module docstring above.
