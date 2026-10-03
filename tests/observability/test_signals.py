# tests/observability/test_signals.py
"""
Tests for M7.3: operational signals (src/observability/signals.py).

Pure replay: event dicts -> Aggregator -> snapshot. Synthetic streams cover
the acceptance scenarios (A healthy RAG, B LLM degradation, C retrieval
failure, D repeated language detection); the invariants (label traceability,
absence-is-not-zero, bounded per-request state) have their own tests; and a
runtime-captured stream proves the aggregator handles real emitted events
with nothing unclassified.
"""

import io
import json
import logging
import math
import pathlib

import pytest

from src.observability import dimensions as dim
from src.observability import signals as sig


# =============================================================================
# helpers
# =============================================================================

def ev(name, status="success", metadata=None, duration_ms=None, rid="r1"):
    return {
        "schema_version": 1, "timestamp": "t", "request_id": rid,
        "event": name, "step": name.split(".")[0], "status": status,
        "duration_ms": duration_ms, "metadata": metadata or {},
    }


def series(snapshot, metric):
    return {tuple(sorted(s["labels"].items())): s["value"] for s in snapshot["metrics"].get(metric, [])}


def val(snapshot, metric, **labels):
    return series(snapshot, metric)[tuple(sorted(labels.items()))]


def request_done(rid="r1", route="/query", status="success", cost="complete", total=0.001, ms=40):
    return ev("request.completed", status, {"route": route, "cost_status": cost, "cost_usd_total": total},
              duration_ms=ms, rid=rid)


def llm_ok(purpose="rag_answer", model="gpt-4o-mini", retries=0, cost=0.0001, rid="r1", ms=300):
    return ev("llm.completed", "success", {
        "purpose": purpose, "model": model, "prompt_tokens": 100, "completion_tokens": 50,
        "total_tokens": 150, "estimated_cost_usd": cost, "retry_count": retries,
    }, duration_ms=ms, rid=rid)


def llm_fail(purpose="rag_answer", model="gpt-4o-mini", rid="r1"):
    return ev("llm.failed", "failure", {
        "purpose": purpose, "model": model, "retry_count": 3, "error_type": "ConnectionError",
    }, duration_ms=5000, rid=rid)


# =============================================================================
# Scenario A — healthy RAG
# =============================================================================

def healthy_rag(rid="r1"):
    return [
        ev("request.started", metadata={"route": "/query"}, rid=rid),
        ev("guardrails.completed", metadata={"blocked": False, "reason": None, "query_length": 20,
                                             "query_hash": "abc"}, duration_ms=1, rid=rid),
        ev("language_detection.completed", duration_ms=2, rid=rid),
        ev("intent.completed", metadata={"intent": "rag", "confidence": 0.9, "method": "heuristic",
                                         "route": "rag"}, duration_ms=1, rid=rid),
        ev("retrieval.completed", metadata={"retrieval_method": "vector_rpc", "candidate_count": 5,
                                            "source_filter": None}, duration_ms=30, rid=rid),
        ev("ranking.completed", metadata={"candidate_count": 5, "selected_count": 3,
                                          "reranker": "hybrid"}, duration_ms=4, rid=rid),
        llm_ok("rag_answer", rid=rid),
        llm_ok("rag_insight", rid=rid),
        request_done(rid=rid, ms=420),
    ]


def test_scenario_a_healthy_rag():
    snap = sig.replay(healthy_rag())

    assert val(snap, "requests_total", route="/query", status="success") == 1
    assert val(snap, "retrieval_total", status="success", retrieval_method="vector_rpc") == 1
    assert val(snap, "ranking_total", status="success", reranker="hybrid") == 1
    assert val(snap, "llm_calls_total", purpose="rag_answer", model="gpt-4o-mini", status="success") == 1
    assert val(snap, "llm_cost_usd_total", purpose="rag_answer", model="gpt-4o-mini")["sum"] == pytest.approx(0.0001)
    assert val(snap, "request_cost_usd_total", route="/query", cost_status="complete")["sum"] == pytest.approx(0.001)
    assert "llm_fallbacks_total" not in snap["metrics"]          # absent, not zero
    assert snap["rates"]["request_error_rate"] == 0.0
    assert snap["rates"]["llm_failure_rate"] == 0.0
    assert snap["rates"]["llm_retry_rate"] == 0.0
    assert snap["rates"]["retrieval_empty_rate"] == 0.0
    assert snap["metrics"]["signals_unclassified_total"][0]["value"] == 0


# =============================================================================
# Scenario B — LLM degradation: retry and fallback are distinct signals
# =============================================================================

def test_scenario_b_retry_and_fallback_are_distinct():
    events = [
        llm_ok("rag_answer", retries=2),                                   # retried, then succeeded
        ev("llm.fallback", metadata={"purpose": "rag_answer", "from_model": "gpt-4o",
                                     "to_model": "gpt-4o-mini", "reason": "budget_threshold",
                                     "cumulative_session_cost_usd": 0.11}),
        llm_ok("rag_answer", model="gpt-4o-mini"),                          # downgraded call succeeded
        request_done(),
    ]
    snap = sig.replay(events)

    assert val(snap, "llm_retried_calls_total", purpose="rag_answer") == 1
    assert val(snap, "llm_retries_total", purpose="rag_answer")["sum"] == 2
    assert val(snap, "llm_fallbacks_total", purpose="rag_answer", reason="budget_threshold") == 1
    # the fallback is its own terminal event, not an LLM call outcome
    assert sum(series(snap, "llm_calls_total").values()) == 2
    assert val(snap, "requests_total", route="/query", status="success") == 1
    assert snap["rates"]["llm_retry_rate"] == 0.5
    assert snap["rates"]["llm_fallback_rate"] == 0.5


def test_llm_failure_is_counted_with_error_type():
    snap = sig.replay([llm_fail(), llm_ok()])
    assert val(snap, "llm_calls_total", purpose="rag_answer", model="gpt-4o-mini", status="failure") == 1
    assert val(snap, "llm_failures_total", purpose="rag_answer", error_type="ConnectionError") == 1
    assert snap["rates"]["llm_failure_rate"] == 0.5


# =============================================================================
# Scenario C — retrieval failure vs empty vs skipped
# =============================================================================

def test_scenario_c_failure_empty_and_skip_are_distinct():
    events = [
        ev("retrieval.failed", "failure", {"retrieval_method": "vector_rpc", "error_type": "RuntimeError",
                                           "source_filter": None}),
        ev("retrieval.completed", metadata={"retrieval_method": "vector_rpc", "candidate_count": 0,
                                            "source_filter": None}),
        ev("retrieval.skipped", "skipped", {"retrieval_method": "vector_rpc", "reason": "no_embedding",
                                            "candidate_count": 0, "selected_count": 0,
                                            "source_filter": None}),
        ev("ranking.skipped", "skipped", {"candidate_count": 0, "selected_count": 0}),
    ]
    snap = sig.replay(events)

    assert val(snap, "retrieval_failures_total", error_type="RuntimeError") == 1
    assert val(snap, "retrieval_empty_total", retrieval_method="vector_rpc") == 1
    assert val(snap, "retrieval_total", status="skipped", retrieval_method="vector_rpc",
               reason="no_embedding") == 1
    assert snap["rates"]["retrieval_failure_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert snap["rates"]["retrieval_skip_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert snap["rates"]["retrieval_empty_rate"] == 1.0  # 1 empty of 1 successful retrieval


# =============================================================================
# Scenario D — the repeated language-detection finding
# =============================================================================

def test_scenario_d_three_language_detection_calls_in_one_request():
    events = [
        llm_ok("language_detection"), llm_ok("language_detection"), llm_ok("language_detection"),
        llm_ok("rag_answer"),
        ev("language_detection.completed", duration_ms=2),
        request_done(),
    ]
    snap = sig.replay(events)

    lang = val(snap, "llm_calls_per_request", purpose="language_detection")
    assert (lang["count"], lang["sum"]) == (1, 3.0)   # one request made 3 calls
    answer = val(snap, "llm_calls_per_request", purpose="rag_answer")
    assert (answer["count"], answer["sum"]) == (1, 1.0)
    # the stage event happened once while the LLM calls happened three times
    assert val(snap, "language_detection_total", status="success") == 1
    assert val(snap, "llm_calls_total", purpose="language_detection",
               model="gpt-4o-mini", status="success") == 3


# =============================================================================
# Per-request state semantics
# =============================================================================

def test_per_request_state_flushes_on_completion_and_is_removed():
    agg = sig.Aggregator()
    agg.observe(llm_ok("rag_answer"))
    assert agg.snapshot()["meta"]["in_flight_requests"] == 1
    assert "llm_calls_per_request" not in agg.snapshot()["metrics"]  # not flushed yet

    agg.observe(request_done())
    snap = agg.snapshot()
    assert snap["meta"]["in_flight_requests"] == 0
    assert val(snap, "llm_calls_per_request", purpose="rag_answer")["count"] == 1


def test_requests_are_grouped_by_request_id_and_never_labelled_by_it():
    snap = sig.replay([
        llm_ok("language_detection", rid="a"), llm_ok("language_detection", rid="a"),
        llm_ok("language_detection", rid="b"),
        request_done(rid="a"), request_done(rid="b"),
    ])
    h = val(snap, "llm_calls_per_request", purpose="language_detection")
    assert (h["count"], h["sum"]) == (2, 3.0)
    assert "request_id" not in json.dumps(snap["metrics"])


def test_dash_request_id_does_not_participate():
    agg = sig.Aggregator()
    agg.observe(llm_ok(rid="-"))
    assert agg.snapshot()["meta"]["in_flight_requests"] == 0
    assert val(agg.snapshot(), "llm_calls_total", purpose="rag_answer",
               model="gpt-4o-mini", status="success") == 1   # still counted as a call


def test_in_flight_state_is_capped_and_eviction_is_observable():
    agg = sig.Aggregator(max_in_flight=2)
    for rid in ("a", "b", "c"):
        agg.observe(llm_ok(rid=rid))
    snap = agg.snapshot()
    assert snap["meta"]["in_flight_requests"] == 2
    assert snap["metrics"]["signals_dropped_requests_total"][0]["value"] == 1


# =============================================================================
# Histograms
# =============================================================================

def test_durations_above_10s_land_in_the_inf_bucket_not_dropped():
    snap = sig.replay([request_done(ms=12000), request_done(ms=3)])
    h = val(snap, "request_duration_ms", route="/query", status="success")
    assert h["count"] == 2
    assert h["buckets"]["+Inf"] == 1 and h["buckets"]["5"] == 1
    assert h["p99"] == "+Inf"


def test_percentiles_are_bucket_upper_bounds():
    events = [request_done(ms=ms) for ms in (3, 4, 6, 8, 90)]
    h = val(sig.replay(events), "request_duration_ms", route="/query", status="success")
    assert h["p50"] == 10      # 3rd of 5 observations falls in the <=10 bucket
    assert h["p95"] == 100
    assert h["sum"] == 111


def test_unmeasured_duration_is_skipped_not_zero():
    snap = sig.replay([request_done(ms=None)])
    assert "request_duration_ms" not in snap["metrics"]
    assert val(snap, "requests_total", route="/query", status="success") == 1


def test_histogram_percentile_edge_cases():
    h = sig._Histogram(sig.DURATION_BUCKETS_MS)
    assert h.percentile(0.5) is None
    h.observe(10_001)
    assert math.isinf(h.percentile(0.5))


# =============================================================================
# Absence is not zero; optional dimensions are omitted
# =============================================================================

def test_ratios_are_none_when_denominator_is_zero():
    snap = sig.replay([])
    assert all(v is None for v in snap["rates"].values())
    assert snap["metrics"]["signals_unclassified_total"][0]["value"] == 0  # meta counters are known


def test_absent_optional_dimension_is_omitted_not_synthesized():
    snap = sig.replay([
        ev("retrieval.completed", metadata={"retrieval_method": "vector_rpc", "candidate_count": 2,
                                            "source_filter": None}),
    ])
    (labels,) = series(snap, "retrieval_total")
    assert dict(labels) == {"status": "success", "retrieval_method": "vector_rpc"}   # no reason="none"


def test_present_null_value_is_a_real_emitted_value():
    snap = sig.replay([ev("guardrails.completed", metadata={"blocked": False, "reason": None,
                                                           "query_length": 3, "query_hash": "x"})])
    assert val(snap, "guardrails_total", status="success", reason="none") == 1


def test_null_cost_is_counted_unknown_and_excluded_from_sums():
    snap = sig.replay([
        llm_ok(cost=None),
        request_done(cost="unknown", total=None),
    ])
    assert val(snap, "llm_cost_unknown_total", purpose="rag_answer", model="gpt-4o-mini") == 1
    cost = val(snap, "llm_cost_usd_total", purpose="rag_answer", model="gpt-4o-mini")
    assert (cost["sum"], cost["count"], cost["null_count"]) == (0, 0, 1)
    req = val(snap, "request_cost_usd_total", route="/query", cost_status="unknown")
    assert req["null_count"] == 1 and req["sum"] == 0


def test_out_of_domain_label_values_become_other():
    snap = sig.replay([llm_ok(purpose="brand_new_purpose", model="mystery-model")])
    assert val(snap, "llm_calls_total", purpose="other", model="other", status="success") == 1


# =============================================================================
# Label traceability
# =============================================================================

def test_specs_validate():
    sig.validate_specs()


def test_every_metric_label_is_a_dimension_of_its_source_events():
    for spec in sig.METRICS:
        for label in spec.labels:
            assert label not in ("request_id", "query_hash", "outcome", "bucket", "operation")
            carried = [e for e in spec.events if (e, label) in dim.POLICY or label == "status"]
            assert carried, f"{spec.name}.{label} not carried by any source event"
            for e in carried:
                assert dim.classify(e, label).kind == dim.DIMENSION


def test_a_non_dimension_label_is_rejected(monkeypatch):
    bad = sig.MetricSpec("bad_total", sig.COUNTER, ("llm.completed",), ("prompt_tokens",))
    monkeypatch.setattr(sig, "METRICS", sig.METRICS + (bad,))
    with pytest.raises(ValueError):
        sig.validate_specs()


# =============================================================================
# Unclassified input
# =============================================================================

@pytest.mark.parametrize("bad", [
    ev("brand_new.completed"),
    ev("llm.completed", metadata={"purpose": "rag_answer", "model": "gpt-4o-mini", "surprise": 1}),
    ev("llm.completed", metadata={"purpose": "rag_answer", "prompt": "secret text"}),   # forbidden key
    {"not": "an event"},
    "garbage",
    None,
])
def test_unclassified_input_is_counted_and_skipped(bad):
    agg = sig.Aggregator()
    agg.observe(bad)
    agg.observe(llm_ok())
    snap = agg.snapshot()
    assert snap["metrics"]["signals_unclassified_total"][0]["value"] == 1
    assert sum(series(snap, "llm_calls_total").values()) == 1    # the good event still counted


def test_strict_mode_raises_on_unclassified():
    with pytest.raises(KeyError):
        sig.Aggregator(strict=True).observe(ev("brand_new.completed"))
    with pytest.raises(ValueError):
        sig.Aggregator(strict=True).observe(ev("llm.completed", metadata={"prompt": "x"}))


# =============================================================================
# Determinism, scope
# =============================================================================

def test_replay_is_deterministic_and_json_serializable():
    events = healthy_rag("a") + healthy_rag("b")
    first, second = sig.replay(events), sig.replay(events)
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_aggregator_is_not_wired_to_emission():
    import ast

    root = pathlib.Path(__file__).resolve().parents[2] / "src"
    assert "signals" not in (root / "observability" / "events.py").read_text(encoding="utf-8")

    tree = ast.parse((root / "observability" / "signals.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
            imported |= {f"{node.module}.{a.name}" for a in node.names}
    # no emission path, no logging handlers, no vendor SDKs
    assert "src.observability.events" not in imported
    assert "logging" not in imported
    assert not {m for m in imported if m and m.split(".")[0] in ("prometheus_client", "opentelemetry")}


# =============================================================================
# CLI (replay/debug only)
# =============================================================================

def test_cli_replays_a_jsonl_file(tmp_path):
    f = tmp_path / "events.jsonl"
    lines = [json.dumps(e) for e in healthy_rag()]
    lines.insert(1, "[2026-10-03 21:00:00] INFO - a human log line, not an event")
    lines.insert(2, "")
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out, err = io.StringIO(), io.StringIO()
    assert sig.main([str(f)], stdout=out, stderr=err) == 0

    snap = json.loads(out.getvalue())
    assert val(snap, "requests_total", route="/query", status="success") == 1
    assert "skipped 1 non-event" in err.getvalue()


def test_cli_reads_stdin_by_default():
    out = io.StringIO()
    sig.main([], stdin=io.StringIO(json.dumps(request_done()) + "\n"), stdout=out, stderr=io.StringIO())
    assert val(json.loads(out.getvalue()), "requests_total", route="/query", status="success") == 1


# =============================================================================
# Runtime-captured events (real emission, every path) replay cleanly
# =============================================================================

@pytest.fixture
def runtime_events(monkeypatch):
    from src.observability import context as context_module
    from src.observability import cost as cost_module
    import src.llm.llm_client as llm_client_module
    from tests.observability import test_dimensions as td

    t1 = context_module._request_id.set(None)
    t2 = cost_module._request_cost.set(cost_module.RequestCost())
    monkeypatch.setattr(llm_client_module.time, "sleep", lambda *_: None)
    monkeypatch.setattr(llm_client_module, "SESSION_COST_USD", 0.0)

    records = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _H()
    lg = logging.getLogger("observability.events")
    lg.addHandler(handler)
    try:
        for driver in (td._drive_router, td._drive_orchestrator, td._drive_retrieval,
                       td._drive_llm, td._drive_analytics):
            driver(monkeypatch)
        yield [json.loads(r.getMessage()) for r in records]
    finally:
        lg.removeHandler(handler)
        cost_module._request_cost.reset(t2)
        context_module._request_id.reset(t1)


def test_runtime_captured_events_replay_with_nothing_unclassified(runtime_events):
    snap = sig.replay(runtime_events, strict=True)   # strict: any unclassified input would raise

    assert snap["metrics"]["signals_unclassified_total"][0]["value"] == 0
    assert snap["meta"]["events_observed"] == len(runtime_events)
    # 3 router requests + 1 allowed by the rate-limit driver (its 2 rejected
    # requests are NOT request.completed; they are rate_limited_total)
    assert sum(series(snap, "requests_total").values()) == 4
    assert sum(series(snap, "rate_limited_total").values()) == 2
    assert val(snap, "rate_limited_total", route="/query") == 1
    assert val(snap, "rate_limited_total", route="other") == 1
    assert val(snap, "requests_total", route="/query", status="blocked") == 1
    # the router driver's failing handler, plus the rate-limit driver's allowed request
    assert val(snap, "requests_total", route="/query", status="error") == 2
    assert val(snap, "analytics_sql_total", status="failure", intent="timeseries",
               used_fallback_sql="true") == 1
    assert val(snap, "llm_fallbacks_total", purpose="rag_answer", reason="budget_threshold") == 1
    json.dumps(snap)   # serializable
