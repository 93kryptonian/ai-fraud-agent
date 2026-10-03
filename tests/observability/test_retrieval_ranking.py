# tests/observability/test_retrieval_ranking.py
"""
Tests for M6.1-M6.2: retrieval and ranking instrumentation
(src/rag/retriever_direct.py).

Covers retrieval's status semantics — success / failure / skipped are
genuinely different conditions, previously all collapsed into a silent
empty list — and the separated retrieval/ranking exception boundaries
(a ranking bug is never misattributed as a retrieval failure), plus
privacy (no raw document/query content in any event).
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


# =============================================================================
# Retrieval — status semantics (success / failure / skipped are distinct)
# =============================================================================

def test_retrieval_skipped_when_disabled_by_flag(event_records, monkeypatch):
    import src.rag.retriever_direct as retriever_module

    monkeypatch.setattr(retriever_module, "RETRIEVER_ENABLED", False)

    rows = retriever_module.retrieve_top_k("card not present fraud")

    assert rows == []
    events = _parsed(event_records)
    assert len(events) == 1
    assert events[0]["event"] == "retrieval.skipped"
    assert events[0]["metadata"]["reason"] == "retriever_disabled"
    assert "ranking" not in _names(event_records)[0]  # no ranking event at all


def test_retrieval_skipped_when_embedding_unavailable(event_records, monkeypatch):
    import src.rag.retriever_direct as retriever_module

    monkeypatch.setattr(retriever_module, "RETRIEVER_ENABLED", True)
    monkeypatch.setattr(retriever_module, "get_supabase", lambda: object())
    monkeypatch.setattr(retriever_module, "embed_text", lambda q: [0.0, 0.0, 0.0])

    rows = retriever_module.retrieve_top_k("card not present fraud")

    assert rows == []
    names = _names(event_records)
    assert names == ["retrieval.skipped"]
    assert _parsed(event_records)[0]["metadata"]["reason"] == "no_embedding"


def test_retrieval_failed_on_supabase_init_error(event_records, monkeypatch):
    import src.rag.retriever_direct as retriever_module

    def _raise():
        raise RuntimeError("Supabase is not configured")

    monkeypatch.setattr(retriever_module, "RETRIEVER_ENABLED", True)
    monkeypatch.setattr(retriever_module, "get_supabase", _raise)

    rows = retriever_module.retrieve_top_k("card not present fraud")

    assert rows == []
    events = _parsed(event_records)
    assert events[0]["event"] == "retrieval.failed"
    assert events[0]["status"] == "failure"
    assert events[0]["metadata"]["error_type"] == "RuntimeError"
    assert isinstance(events[0]["duration_ms"], int)


def test_retrieval_completed_then_ranking_skipped_on_zero_candidates(event_records, monkeypatch):
    import src.rag.retriever_direct as retriever_module

    fake_supabase = SimpleNamespace(
        rpc=lambda name, params: SimpleNamespace(execute=lambda: SimpleNamespace(data=[]))
    )
    monkeypatch.setattr(retriever_module, "RETRIEVER_ENABLED", True)
    monkeypatch.setattr(retriever_module, "get_supabase", lambda: fake_supabase)
    monkeypatch.setattr(retriever_module, "embed_text", lambda q: [0.1, 0.2])

    rows = retriever_module.retrieve_top_k("card not present fraud")

    assert rows == []
    assert _names(event_records) == ["retrieval.completed", "ranking.skipped"]
    events = _parsed(event_records)
    assert events[0]["metadata"]["candidate_count"] == 0
    assert events[1]["status"] == "skipped"


def test_retrieval_and_ranking_completed_with_real_candidates(event_records, monkeypatch):
    import src.rag.retriever_direct as retriever_module

    candidates = [
        {"content": "card-not-present fraud is high risk", "source_name": "Bhatla", "page": 1, "similarity": 0.9},
        {"content": "unrelated content about the weather", "source_name": "Bhatla", "page": 2, "similarity": 0.1},
    ]
    fake_supabase = SimpleNamespace(
        rpc=lambda name, params: SimpleNamespace(execute=lambda: SimpleNamespace(data=candidates))
    )
    monkeypatch.setattr(retriever_module, "RETRIEVER_ENABLED", True)
    monkeypatch.setattr(retriever_module, "get_supabase", lambda: fake_supabase)
    monkeypatch.setattr(retriever_module, "embed_text", lambda q: [0.1, 0.2])

    rows = retriever_module.retrieve_top_k("card not present fraud", top_k=2)

    assert len(rows) == 2
    assert _names(event_records) == ["retrieval.completed", "ranking.completed"]
    events = _parsed(event_records)
    assert events[0]["metadata"]["candidate_count"] == 2
    assert events[1]["metadata"]["candidate_count"] == 2
    assert events[1]["metadata"]["selected_count"] == 2
    assert events[1]["metadata"]["reranker"] == "hybrid"


def test_ranking_failed_fails_closed(event_records, monkeypatch):
    import src.rag.retriever_direct as retriever_module

    candidates = [{"content": "x", "source_name": "Bhatla", "page": 1, "similarity": 0.5}]
    fake_supabase = SimpleNamespace(
        rpc=lambda name, params: SimpleNamespace(execute=lambda: SimpleNamespace(data=candidates))
    )
    monkeypatch.setattr(retriever_module, "RETRIEVER_ENABLED", True)
    monkeypatch.setattr(retriever_module, "get_supabase", lambda: fake_supabase)
    monkeypatch.setattr(retriever_module, "embed_text", lambda q: [0.1, 0.2])

    def _raise(*a, **kw):
        raise ValueError("simulated ranking bug")

    monkeypatch.setattr(retriever_module, "rerank_chunks", _raise)

    rows = retriever_module.retrieve_top_k("card not present fraud")

    assert rows == []  # fail closed, doesn't crash the caller
    names = _names(event_records)
    assert names == ["retrieval.completed", "ranking.failed"]
    events = _parsed(event_records)
    assert events[1]["metadata"]["error_type"] == "ValueError"


def test_no_raw_content_in_retrieval_ranking_events(event_records, monkeypatch):
    import src.rag.retriever_direct as retriever_module

    secret = "super-secret-query-about-jane.doe@example.com"
    candidates = [{"content": "this document mentions " + secret, "source_name": "Bhatla", "page": 1, "similarity": 0.9}]
    fake_supabase = SimpleNamespace(
        rpc=lambda name, params: SimpleNamespace(execute=lambda: SimpleNamespace(data=candidates))
    )
    monkeypatch.setattr(retriever_module, "RETRIEVER_ENABLED", True)
    monkeypatch.setattr(retriever_module, "get_supabase", lambda: fake_supabase)
    monkeypatch.setattr(retriever_module, "embed_text", lambda q: [0.1, 0.2])

    retriever_module.retrieve_top_k(secret)

    raw_lines = [r.getMessage() for r in event_records]
    for line in raw_lines:
        assert secret not in line
        assert "jane.doe@example.com" not in line
