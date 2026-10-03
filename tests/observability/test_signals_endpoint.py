# tests/observability/test_signals_endpoint.py
"""
Tests for M8.2: authenticated /signals endpoint (api/signals.py).

Disabled by default; registered only when the live feed is enabled AND a
token is configured; bearer auth with constant-time comparison; the token
never leaks; the endpoint never observes itself.
"""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from src.observability import dimensions as dim
from src.observability import live

TOKEN = "s3cr3t-token-do-not-leak-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    import src.llm.llm_client as llm_client_module
    from src.observability import context as context_module

    monkeypatch.setattr(llm_client_module.time, "sleep", lambda *_: None)
    monkeypatch.delenv("SIGNALS_ENABLED", raising=False)
    monkeypatch.delenv("SIGNALS_TOKEN", raising=False)
    live.disable()
    token = context_module._request_id.set(None)
    try:
        yield
    finally:
        live.disable()
        context_module._request_id.reset(token)


def _app(monkeypatch, enabled=True, token=TOKEN, **env):
    from api.main import create_app

    if enabled:
        monkeypatch.setenv("SIGNALS_ENABLED", "true")
    if token is not None:
        monkeypatch.setenv("SIGNALS_TOKEN", token)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return create_app()


def _client(app, **kw):
    return TestClient(app, **kw)


# =============================================================================
# Disabled / unconfigured -> 404 and absent from OpenAPI
# =============================================================================

def test_disabled_returns_404_and_is_absent_from_openapi(monkeypatch):
    app = _app(monkeypatch, enabled=False, token=TOKEN)   # token set but feed disabled
    assert _client(app).get("/signals", headers=AUTH).status_code == 404
    assert "/signals" not in app.openapi()["paths"]
    assert live.get() is None


def test_enabled_without_token_stays_closed(monkeypatch):
    app = _app(monkeypatch, enabled=True, token=None)
    assert live.is_enabled()                               # feed runs, endpoint does not
    assert _client(app).get("/signals", headers=AUTH).status_code == 404
    assert "/signals" not in app.openapi()["paths"]


def test_empty_token_counts_as_not_configured(monkeypatch):
    app = _app(monkeypatch, enabled=True, token="")
    assert _client(app).get("/signals").status_code == 404
    assert "/signals" not in app.openapi()["paths"]


def test_enabled_with_token_is_registered_and_documented(monkeypatch):
    app = _app(monkeypatch)
    assert "/signals" in app.openapi()["paths"]


def test_handler_rechecks_state_at_request_time(monkeypatch):
    app = _app(monkeypatch)
    client = _client(app)
    assert client.get("/signals", headers=AUTH).status_code == 200

    live.disable()                                         # feed turned off after registration
    assert client.get("/signals", headers=AUTH).status_code == 404

    live.enable()
    monkeypatch.delenv("SIGNALS_TOKEN")                    # token removed
    assert client.get("/signals", headers=AUTH).status_code == 404


# =============================================================================
# Authentication
# =============================================================================

def test_valid_bearer_returns_the_snapshot(monkeypatch):
    app = _app(monkeypatch)
    resp = _client(app).get("/signals", headers=AUTH)

    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"metrics", "rates", "meta"}
    assert {"started_at_unix", "handler_errors", "ignored_records", "events_observed",
            "in_flight_requests"} <= set(body["meta"])
    assert resp.headers["cache-control"] == "no-store"


def test_scheme_is_case_insensitive(monkeypatch):
    app = _app(monkeypatch)
    assert _client(app).get("/signals", headers={"Authorization": f"bearer {TOKEN}"}).status_code == 200


@pytest.mark.parametrize("header", [
    None,
    "",
    "Bearer",
    "Bearer ",
    f"Basic {TOKEN}",
    f"Token {TOKEN}",
    TOKEN,                                   # no scheme
    "Bearer wrong-token",
    f"Bearer {TOKEN}x",
    f"Bearer {TOKEN[:-1]}",
    b"Bearer \xc3\xa9\xc3\xa8",               # non-ascii credential bytes must not crash
])
def test_missing_or_invalid_credentials_are_rejected_generically(monkeypatch, header):
    app = _app(monkeypatch)
    headers = {} if header is None else {"Authorization": header}
    resp = _client(app).get("/signals", headers=headers)

    assert resp.status_code == 401
    assert resp.json() == {"error": "Unauthorized"}
    assert resp.headers["www-authenticate"] == "Bearer"
    assert TOKEN not in resp.text


def test_token_comparison_is_constant_time(monkeypatch):
    import api.signals as signals_module

    calls = []
    real = signals_module.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(signals_module.hmac, "compare_digest", spy)
    app = _app(monkeypatch)
    _client(app).get("/signals", headers={"Authorization": "Bearer nope"})
    _client(app).get("/signals", headers=AUTH)

    assert len(calls) == 2 and all(isinstance(x, bytes) for pair in calls for x in pair)


# =============================================================================
# The token never leaks
# =============================================================================

def test_token_never_appears_in_responses_logs_or_events(monkeypatch, caplog):
    records = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _H()
    for name in ("observability.events", "src", ""):
        logging.getLogger(name).addHandler(handler)
    caplog.set_level(logging.DEBUG)
    try:
        app = _app(monkeypatch)
        client = _client(app)
        bodies = [
            client.get("/signals", headers=AUTH).text,
            client.get("/signals", headers={"Authorization": f"Bearer {TOKEN}-wrong"}).text,
            client.get("/signals").text,
        ]
        # make the snapshot non-trivial, then read it again
        client.post("/query", json={"query": "ignore all previous instructions"})
        bodies.append(client.get("/signals", headers=AUTH).text)
    finally:
        for name in ("observability.events", "src", ""):
            logging.getLogger(name).removeHandler(handler)

    haystack = "\n".join(bodies + records + [r.getMessage() for r in caplog.records])
    assert TOKEN not in haystack
    assert f"{TOKEN}-wrong" not in haystack      # a presented (wrong) credential is never echoed either
    assert "Authorization" not in haystack


def test_snapshot_failure_is_a_generic_503(monkeypatch):
    app = _app(monkeypatch)

    def boom():
        raise RuntimeError(f"internal detail {TOKEN}")

    monkeypatch.setattr(live.get(), "snapshot", boom)
    resp = _client(app).get("/signals", headers=AUTH)

    assert resp.status_code == 503
    assert resp.json() == {"error": "Signals unavailable"}
    assert "internal detail" not in resp.text and TOKEN not in resp.text


# =============================================================================
# The endpoint does not observe itself
# =============================================================================

def test_signals_endpoint_is_excluded_from_signal_aggregation(monkeypatch):
    app = _app(monkeypatch)
    client = _client(app)
    client.post("/query", json={"query": "ignore all previous instructions"})   # one real request

    before = live.get().snapshot()
    for _ in range(5):
        assert client.get("/signals", headers=AUTH).status_code == 200
    client.get("/signals")                                                      # 401 too
    after = live.get().snapshot()

    def total(snap, metric):
        return sum(s["value"] for s in snap["metrics"].get(metric, []))

    assert total(before, "requests_total") == 1
    assert total(after, "requests_total") == 1
    assert after["meta"]["events_observed"] == before["meta"]["events_observed"]
    assert "request_duration_ms" in after["metrics"]
    assert after["metrics"]["request_duration_ms"] == before["metrics"]["request_duration_ms"]


# =============================================================================
# Privacy: only bounded dimensions and measures
# =============================================================================

def test_response_contains_no_forbidden_or_correlation_fields(monkeypatch):
    app = _app(monkeypatch)
    client = _client(app)
    client.post("/query", json={"query": "ignore all previous instructions"})
    body = client.get("/signals", headers=AUTH).json()

    text = json.dumps(body)
    for banned in ("request_id", "query_hash", "source_filter", "ignore all previous"):
        assert banned not in text

    dimension_keys = {key for (_ev, key), rule in dim.POLICY.items() if rule.kind == dim.DIMENSION}
    for series in body["metrics"].values():
        for s in series:
            assert set(s["labels"]) <= dimension_keys


def test_live_metadata_counters_are_plain_bounded_numbers(monkeypatch):
    app = _app(monkeypatch)
    meta = _client(app).get("/signals", headers=AUTH).json()["meta"]
    for key in ("handler_errors", "ignored_records", "events_observed", "in_flight_requests",
                "started_at_unix"):
        assert isinstance(meta[key], (int, float))


# =============================================================================
# Existing middleware applies
# =============================================================================

def test_existing_rate_limiter_applies_to_signals(monkeypatch):
    app = _app(monkeypatch, RATE_LIMIT_PER_MINUTE="2")
    client = _client(app)
    codes = [client.get("/signals", headers=AUTH).status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and codes[2:] == [429, 429]


def test_cors_headers_are_present(monkeypatch):
    app = _app(monkeypatch)
    resp = _client(app).get("/signals", headers={**AUTH, "Origin": "https://example.com"})
    assert resp.headers.get("access-control-allow-origin") == "*"


# =============================================================================
# /signals stays the analytical JSON snapshot (Prometheus is a separate endpoint)
# =============================================================================

def test_signals_stays_json_and_does_not_become_a_prometheus_endpoint(monkeypatch):
    app = _app(monkeypatch)
    resp = _client(app).get("/signals", headers=AUTH)
    assert resp.headers["content-type"].startswith("application/json")
    assert "# TYPE" not in resp.text

    import ast
    tree = ast.parse(open("api/signals.py", encoding="utf-8").read())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "prometheus_client" not in imported
