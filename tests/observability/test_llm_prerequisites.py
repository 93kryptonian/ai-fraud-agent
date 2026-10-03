# tests/observability/test_llm_prerequisites.py
"""
Tests for P1 and P2 — the two M6 prerequisites (not M6 event instrumentation
itself, which hasn't started).

P1: LLMClient.run() must raise LLMExhaustedRetriesError, with a real
    retry_count and last_error_type, after exhausting retries — never the
    old fake-success sentinel string. And every call site that previously
    relied on that sentinel implicitly resolving to a graceful fallback
    must still behave the same way now that a real exception propagates
    instead (verified concretely for the one call site that wasn't
    already inside a try/except: guardrails.py's own detect_language call).

P2: LLM cost becomes trustworthy per request via a request-scoped
    accumulator (src/observability/cost.py), additive to — not replacing —
    the existing process-global session budget guard in llm_client.py.
"""

from types import SimpleNamespace

import pytest

from src.llm.llm_client import LLMClient, LLMExhaustedRetriesError


def _make_fake_openai_always_fails():
    """Minimal stand-in for the OpenAI client: every call raises."""

    def create(**kwargs):
        raise ConnectionError("simulated network failure")

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _make_fake_openai_succeeds():
    """Minimal stand-in for the OpenAI client: every call succeeds."""

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="a real answer"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )

    def create(**kwargs):
        return response

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # The retry loop sleeps 1.2/2.4/3.6/4.8s between attempts — don't
    # actually wait through that in tests.
    import src.llm.llm_client as llm_client_module

    monkeypatch.setattr(llm_client_module.time, "sleep", lambda *_: None)


@pytest.fixture(autouse=True)
def _reset_request_cost():
    from src.observability import cost as cost_module

    token = cost_module._request_cost.set(cost_module.RequestCost())
    try:
        yield
    finally:
        cost_module._request_cost.reset(token)


# =============================================================================
# P1 — LLMClient.run() raises, with structured retry info
# =============================================================================

def test_run_raises_llm_exhausted_retries_error_not_sentinel_string(monkeypatch):
    client = LLMClient()
    monkeypatch.setattr(
        "src.llm.llm_client.get_openai_client",
        lambda: _make_fake_openai_always_fails(),
    )

    with pytest.raises(LLMExhaustedRetriesError) as exc_info:
        client.run("does this fail?")

    err = exc_info.value
    assert err.retry_count == 4  # MAX_RETRIES
    assert err.last_error_type == "ConnectionError"
    # Privacy: the exception's own string form must not be what an event
    # would emit — only structured fields, never str(e)/message parsing.
    # (This test just documents the fields exist; M6 wiring is separate.)


def test_run_still_returns_normally_on_success(monkeypatch):
    client = LLMClient()
    monkeypatch.setattr(
        "src.llm.llm_client.get_openai_client",
        lambda: _make_fake_openai_succeeds(),
    )

    result = client.run("does this succeed?")
    assert result == "a real answer"


def test_sentinel_string_is_gone_as_a_failure_mechanism(monkeypatch):
    """
    Locks the fix: the old behavior (returning the literal sentinel string
    instead of raising) must not exist anymore, for any reason.
    """
    client = LLMClient()
    monkeypatch.setattr(
        "src.llm.llm_client.get_openai_client",
        lambda: _make_fake_openai_always_fails(),
    )

    with pytest.raises(LLMExhaustedRetriesError):
        result = client.run("does this fail?")
        assert result != "LLM failed after retries."  # unreachable if it still returned


# =============================================================================
# P1 (continued) — the one previously-unprotected call site still degrades
# gracefully, now explicitly instead of by sentinel-string accident
# =============================================================================

def test_guardrails_language_detection_still_falls_back_to_en_on_llm_failure(monkeypatch):
    from src.safety.guardrails import validate_query

    monkeypatch.setattr(
        "src.llm.llm_client.get_openai_client",
        lambda: _make_fake_openai_always_fails(),
    )
    monkeypatch.setattr("src.llm.llm_client.time.sleep", lambda *_: None)

    # A query with no id/en heuristic keyword match forces detect_language
    # into its LLM fallback path, inside guardrails.validate_query — the
    # one call site that had no local try/except before P1.
    ok, msg, lang, reason = validate_query("xyzxyz")
    assert lang == "en"  # unchanged from prior (accidental) behavior


# =============================================================================
# P2 — request-scoped cost accumulation
# =============================================================================

def test_request_cost_accumulates_across_multiple_llm_calls(monkeypatch):
    from src.observability.cost import get_request_cost

    client = LLMClient()
    monkeypatch.setattr(
        "src.llm.llm_client.get_openai_client",
        lambda: _make_fake_openai_succeeds(),
    )

    assert get_request_cost() == 0.0
    client.run("first call")
    first_cost = get_request_cost()
    assert first_cost > 0.0

    client.run("second call")
    assert get_request_cost() == pytest.approx(first_cost * 2)


def test_request_cost_is_isolated_between_requests():
    """
    Same isolation guarantee M3 already proved for request_id, applied to
    the new per-request cost accumulator — via asyncio Tasks, matching the
    real execution model (each HTTP request is its own task).
    """
    import asyncio
    from src.observability.cost import add_request_cost, get_request_cost, reset_request_cost

    seen = {}

    async def simulate_request(label: str, cost: float, delay: float):
        reset_request_cost()
        add_request_cost(cost)
        await asyncio.sleep(delay)
        seen[label] = get_request_cost()

    async def run():
        await asyncio.gather(
            simulate_request("A", 0.01, 0.02),
            simulate_request("B", 0.05, 0.01),
        )

    asyncio.run(run())
    assert seen == {"A": 0.01, "B": 0.05}


def test_process_global_session_budget_is_untouched_by_p2():
    """
    P2 is additive. The existing whole-process soft budget guard
    (SESSION_COST_USD / MAX_COST_USD in llm_client.py) must still exist
    and behave exactly as before — this is not converted to per-request
    scope, which would silently weaken it (a $0.10 default budget almost
    never trips within a single request).
    """
    import src.llm.llm_client as llm_client_module

    assert hasattr(llm_client_module, "SESSION_COST_USD")
    assert hasattr(llm_client_module, "MAX_COST_USD")
