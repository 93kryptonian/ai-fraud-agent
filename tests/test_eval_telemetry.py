"""eval/telemetry.py: the live report reads real telemetry events and never invents a number."""

import pytest

from eval.telemetry import EventCollector, summarize
from src.observability.events import emit_event


def test_collector_reads_the_events_the_system_really_emits():
    with EventCollector() as c:
        emit_event("intent.completed", step="intent", status="success", duration_ms=3,
                   metadata={"intent": "analytics", "confidence": 0.95, "method": "heuristic", "route": "analytics"})
        emit_event("llm.completed", step="llm", status="success", duration_ms=800,
                   metadata={"purpose": "rag_answer", "model": "m", "estimated_cost_usd": 0.0004, "retry_count": 0})
        emit_event("request.completed", step="request", status="success", duration_ms=1234,
                   metadata={"route": "/query", "cost_status": "complete", "cost_usd_total": 0.0004})
    s = summarize(c.events)
    assert s["duration_ms"] == 1234 and s["cost_usd"] == 0.0004 and s["cost_status"] == "complete"
    assert (s["intent_method"], s["intent_confidence"]) == ("heuristic", 0.95)
    assert s["llm_calls"] == 1 and s["llm_purposes"] == ["rag_answer"] and s["llm_failures"] == 0


def test_collector_detaches_when_done():
    import logging

    logger = logging.getLogger("observability.events")
    before = list(logger.handlers)
    with EventCollector():
        pass
    assert logger.handlers == before


def test_a_request_with_no_priced_calls_reports_zero_calls_not_a_made_up_cost():
    s = summarize([{"event": "request.completed", "status": "blocked", "duration_ms": 5,
                    "metadata": {"cost_status": "not_applicable", "cost_usd_total": 0.0}},
                   {"event": "guardrails.blocked", "metadata": {}}])
    assert s["llm_calls"] == 0 and s["cost_status"] == "not_applicable" and s["guardrail_blocked"] is True


def test_unknown_cost_stays_unknown_and_missing_events_stay_none():
    s = summarize([{"event": "request.completed", "status": "success", "duration_ms": 9,
                    "metadata": {"cost_status": "unknown", "cost_usd_total": None}}])
    assert s["cost_usd"] is None and s["cost_status"] == "unknown"
    empty = summarize([])
    assert empty["duration_ms"] is None and empty["cost_usd"] is None and empty["intent_method"] is None


def test_llm_failures_are_counted():
    s = summarize([{"event": "llm.failed", "metadata": {"purpose": "intent_classification"}},
                   {"event": "llm.completed", "metadata": {"purpose": "intent_classification"}}])
    assert s["llm_calls"] == 2 and s["llm_failures"] == 1


def test_embedding_calls_tokens_and_cost_are_reported_and_unknown_stays_none():
    def done(tokens, cost):
        return {"event": "embedding.completed", "metadata": {"model": "m", "total_tokens": tokens, "estimated_cost_usd": cost}}

    s = summarize([done(10, 2e-7), done(20, 4e-7), {"event": "embedding.failed", "metadata": {"model": "m"}}])
    assert (s["embedding_calls"], s["embedding_failures"], s["embedding_tokens"]) == (3, 1, 30)
    assert s["embedding_cost_usd"] == pytest.approx(6e-7)
    unknown = summarize([done(10, 2e-7), done(None, None)])
    assert unknown["embedding_cost_usd"] is None and unknown["embedding_tokens"] is None
    none = summarize([])
    assert (none["embedding_calls"], none["embedding_cost_usd"]) == (0, None)
