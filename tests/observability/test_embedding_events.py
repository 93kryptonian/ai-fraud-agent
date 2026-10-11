"""Embedding calls are metered like LLM calls: priced from usage, unknown is never $0, telemetry never changes the result."""

from types import SimpleNamespace

import pytest

import src.embeddings.embedder as emb
from src.observability import cost as cost_module
from src.observability.cost import cost_metadata


@pytest.fixture(autouse=True)
def _fresh_cost():
    token = cost_module._request_cost.set(cost_module.RequestCost())
    try:
        yield
    finally:
        cost_module._request_cost.reset(token)


@pytest.fixture
def events(monkeypatch):
    captured = []
    monkeypatch.setattr(emb, "emit_event", lambda name, **kw: captured.append((name, kw)))
    monkeypatch.setattr(emb, "EMBEDDINGS_ENABLED", True)
    return captured


def _client(create):
    return SimpleNamespace(embeddings=SimpleNamespace(create=create))


def _response(tokens=1000, usage=True):
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2])],
        usage=SimpleNamespace(total_tokens=tokens, prompt_tokens=tokens) if usage else None,
    )


def _use(monkeypatch, create):
    monkeypatch.setattr(emb, "_get_openai_client", lambda: _client(create))


def test_price_is_per_million_input_tokens():
    assert emb.estimate_embedding_cost("text-embedding-3-small", 1_000_000) == pytest.approx(0.02)
    assert emb.estimate_embedding_cost("text-embedding-3-small", 1000) == pytest.approx(0.00002)


def test_a_single_short_query_is_not_rounded_down_to_zero():
    # ~15 tokens is ~$0.0000003; at 6 decimals this would read as $0.00
    cost = emb.estimate_embedding_cost("text-embedding-3-small", 15)
    assert cost == pytest.approx(3e-7) and cost > 0


def test_unpriced_model_or_missing_usage_is_unknown_never_zero():
    assert emb.estimate_embedding_cost("some-other-model", 1000) is None
    assert emb.estimate_embedding_cost("text-embedding-3-small", None) is None


def test_a_priced_embedding_emits_tokens_and_cost_and_counts_toward_the_request(monkeypatch, events):
    _use(monkeypatch, lambda **kw: _response(tokens=2000))
    vectors = emb.embed_texts(["which merchants have the highest fraud?"])
    assert vectors == [[0.1, 0.2]]

    ((name, kw),) = events
    assert name == "embedding.completed"
    assert kw["metadata"] == {"model": "text-embedding-3-small", "total_tokens": 2000, "estimated_cost_usd": 0.00004}
    assert cost_metadata() == {"cost_status": "complete", "cost_usd_total": 0.00004}


def test_no_text_reaches_the_event(monkeypatch, events):
    _use(monkeypatch, lambda **kw: _response())
    emb.embed_texts(["a very specific secret question"])
    assert "secret" not in repr(events)


def test_missing_usage_makes_cost_unknown_not_zero(monkeypatch, events):
    _use(monkeypatch, lambda **kw: _response(usage=False))
    emb.embed_texts(["q"])
    assert events[0][1]["metadata"]["estimated_cost_usd"] is None
    assert cost_metadata() == {"cost_status": "unknown", "cost_usd_total": None}


def test_an_unpriced_embedding_model_is_partial_next_to_a_priced_call(monkeypatch, events):
    _use(monkeypatch, lambda **kw: _response(tokens=1000))
    emb.embed_texts(["q"])                                   # priced
    monkeypatch.setattr(emb, "EMBEDDING_MODEL", "some-other-model")
    emb.embed_texts(["q"])                                   # unpriced
    assert cost_metadata()["cost_status"] == "partial"


def test_a_failure_emits_embedding_failed_and_still_degrades_to_a_dummy_vector(monkeypatch, events):
    def boom(**kw):
        raise ConnectionError("down")

    _use(monkeypatch, boom)
    vectors = emb.embed_texts(["q"])
    assert vectors == [[0.0] * emb.DUMMY_VECTOR_DIM]
    ((name, kw),) = events
    assert name == "embedding.failed"
    assert kw["metadata"] == {"model": "text-embedding-3-small", "error_type": "ConnectionError"}
    assert cost_metadata()["cost_status"] == "not_applicable"      # no response, no cost, no unknown


def test_a_telemetry_failure_never_turns_a_good_embedding_into_a_dummy_vector(monkeypatch):
    monkeypatch.setattr(emb, "EMBEDDINGS_ENABLED", True)
    _use(monkeypatch, lambda **kw: _response())

    def broken(*a, **k):
        raise RuntimeError("telemetry down")

    monkeypatch.setattr(emb, "emit_event", broken)
    assert emb.embed_texts(["q"]) == [[0.1, 0.2]]


def test_disabled_embeddings_make_no_call_and_emit_nothing(monkeypatch, events):
    monkeypatch.setattr(emb, "EMBEDDINGS_ENABLED", False)
    emb.embed_texts(["q"])
    assert events == [] and cost_metadata()["cost_status"] == "not_applicable"


# ---- the aggregator turns embedding events into signals -------------------

def test_embedding_events_aggregate_into_calls_tokens_cost_and_failures():
    from src.observability import signals as sig

    def ev(name, status, metadata):
        return {"schema_version": 1, "timestamp": "t", "request_id": "r1", "event": name,
                "step": "embedding", "status": status, "duration_ms": 5, "metadata": metadata}

    m = "text-embedding-3-small"
    snap = sig.replay([
        ev("embedding.completed", "success", {"model": m, "total_tokens": 1000, "estimated_cost_usd": 0.00002}),
        ev("embedding.completed", "success", {"model": m, "total_tokens": 3000, "estimated_cost_usd": 0.00006}),
        ev("embedding.completed", "success", {"model": m, "total_tokens": None, "estimated_cost_usd": None}),
        ev("embedding.failed", "failure", {"model": m, "error_type": "ConnectionError"}),
    ], strict=True)

    def series(metric):
        return {tuple(sorted(s["labels"].items())): s["value"] for s in snap["metrics"].get(metric, [])}

    assert series("embedding_calls_total") == {
        (("model", m), ("status", "success")): 3, (("model", m), ("status", "failure")): 1}
    assert series("embedding_failures_total") == {(("error_type", "ConnectionError"),): 1}
    tokens = series("embedding_tokens_total")[(("model", m),)]
    assert (tokens["sum"], tokens["count"], tokens["null_count"]) == (4000, 2, 1)   # the missing-usage call is null, not 0
    cost = series("embedding_cost_usd_total")[(("model", m),)]
    assert cost["sum"] == pytest.approx(0.00008) and cost["null_count"] == 1
    assert series("embedding_cost_unknown_total") == {(("model", m),): 1}
