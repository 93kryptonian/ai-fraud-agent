# tests/observability/test_analytics_events.py
"""
Tests for M6.4-M6.5: analytics instrumentation (src/analytics/fraud_analytics.py).

Covers the explicit hard invariant: exactly one terminal analytics.sql
event even when a primary attempt fails and a fallback succeeds (never
.failed followed by .completed), and the request-lifecycle-vs-business-
outcome distinction extended to analytics.completed — a legitimate
"insufficient data" answer is status=success, only a genuine internal
exception (caught by the outer handler) is status=failure.
"""

import json
import logging

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
def _analytics_env(monkeypatch):
    monkeypatch.setenv("ANALYTICS_USE_LLM_SQL", "0")  # generic intent unreachable in these tests anyway


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


def test_analytics_sql_one_event_when_primary_succeeds(event_records, monkeypatch):
    import src.analytics.fraud_analytics as fa

    monkeypatch.setattr(fa, "execute_sql", lambda sql: __import__("pandas").DataFrame({"date": [1, 2], "fraud_rate": [0.1, 0.2]}))

    fa.run_analytics("what is the fraud rate trend?", lang="en")

    events = _parsed(event_records)
    sql_events = [e for e in events if e["step"] == "analytics.sql"]
    assert len(sql_events) == 1
    assert sql_events[0]["event"] == "analytics.sql.completed"
    assert sql_events[0]["metadata"]["used_fallback_sql"] is False


def test_analytics_sql_one_event_when_fallback_succeeds_after_primary_fails(event_records, monkeypatch):
    import src.analytics.fraud_analytics as fa

    calls = {"n": 0}

    def _execute_sql(sql):
        import pandas as pd
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("primary attempt failed")
        return pd.DataFrame({"date": [1, 2], "fraud_rate": [0.1, 0.2]})

    monkeypatch.setattr(fa, "execute_sql", _execute_sql)

    fa.run_analytics("what is the fraud rate trend?", lang="en")

    events = _parsed(event_records)
    sql_events = [e for e in events if e["step"] == "analytics.sql"]
    # The hard invariant: ONE terminal event, not .failed followed by .completed.
    assert len(sql_events) == 1
    assert sql_events[0]["event"] == "analytics.sql.completed"
    assert sql_events[0]["metadata"]["used_fallback_sql"] is True
    assert sql_events[0]["metadata"]["primary_error_type"] == "RuntimeError"


def test_analytics_sql_failed_once_when_both_primary_and_fallback_fail(event_records, monkeypatch):
    import src.analytics.fraud_analytics as fa

    def _always_fails(sql):
        raise RuntimeError("DB unavailable")

    monkeypatch.setattr(fa, "execute_sql", _always_fails)

    result = fa.run_analytics("what is the fraud rate trend?", lang="en")

    events = _parsed(event_records)
    sql_events = [e for e in events if e["step"] == "analytics.sql"]
    assert len(sql_events) == 1
    assert sql_events[0]["event"] == "analytics.sql.failed"

    # And the outer analytics.completed correctly reports failure — this
    # is a genuine internal exception, not a business "insufficient data"
    # outcome, so it must NOT be status=success.
    completed_evt = next(e for e in events if e["event"] == "analytics.completed")
    assert completed_evt["status"] == "failure"
    assert "error" in result


def test_analytics_completed_success_for_insufficient_data(event_records, monkeypatch):
    """
    Locks the request-lifecycle-vs-business-outcome distinction at the
    analytics stage too: an honest "insufficient data" business answer is
    NOT a stage failure.
    """
    import src.analytics.fraud_analytics as fa
    import pandas as pd

    monkeypatch.setattr(fa, "execute_sql", lambda sql: pd.DataFrame())

    result = fa.run_analytics("what is the fraud rate trend?", lang="en")

    completed_evt = next(e for e in _parsed(event_records) if e["event"] == "analytics.completed")
    assert completed_evt["status"] == "success"
    assert result["confidence"] == 0.0  # legitimately "we don't know", not a crash
