"""
Live demo UI (live_ui/): a thin client of the API.

The UI must stay a thin client: one setting (the API URL), no provider or
database credentials, no imports from the API's own packages, and model text
never rendered as HTML.
"""

import ast
import json
import re
import sys
from pathlib import Path

import httpx
import pytest

LIVE_UI = Path(__file__).resolve().parent.parent / "live_ui"
sys.path.insert(0, str(LIVE_UI))

import api_client  # noqa: E402


def _ask(handler, **kw):
    return api_client.ask("https://api.example.test/", "a question", transport=httpx.MockTransport(handler), **kw)


def _json(status, body, headers=None):
    return lambda request: httpx.Response(status, json=body, headers=headers)


# ---- Parsing the API's real response shapes ------------------------------

def test_analytics_answer_carries_the_sql_and_rows():
    body = {"query": "q", "intent": "analytics", "error": None, "result": {
        "type": "analytics",
        "analytics": {"answer": "A leads.", "sql": "SELECT 1", "data_points": [{"merchant": "A"}], "confidence": 0.73, "chart_data": []},
    }}
    out = _ask(_json(200, body))
    assert out.kind == api_client.ANALYTICS
    assert (out.answer, out.sql, out.rows, out.confidence) == ("A leads.", "SELECT 1", [{"merchant": "A"}], 0.73)


def test_analytics_without_sql_is_visible_as_having_no_sql():
    body = {"error": None, "result": {"type": "analytics", "analytics": {"answer": "x", "sql": None, "data_points": []}}}
    out = _ask(_json(200, body))
    assert out.kind == api_client.ANALYTICS and out.sql is None


def test_rag_answer_carries_citations_and_flags_degraded_retrieval():
    body = {"error": None, "result": {
        "type": "rag", "answer": "Text.", "retrieval_status": "ok",
        "citations": [{"source": "Bhatla", "page": 5, "preview": "p"}], "score": {"final_score": 0.5},
    }}
    out = _ask(_json(200, body))
    assert out.kind == api_client.RAG and out.citations[0]["source"] == "Bhatla" and out.message == ""
    body["result"]["retrieval_status"] = "unavailable"
    assert _ask(_json(200, body)).message  # the page tells the reader


def test_guardrail_rejection_is_not_an_error():
    body = {"error": None, "intent": "reject", "result": {"type": "reject", "message": "Potential prompt-injection attempt detected."}}
    out = _ask(_json(200, body))
    assert out.kind == api_client.REJECTED and "prompt-injection" in out.message


# ---- Failure modes: every one is classified, none raises -----------------

def test_rate_limit_reads_retry_after_header_then_body():
    out = _ask(_json(429, {"error": "Rate limit exceeded.", "retry_after_seconds": 12}, {"Retry-After": "7"}))
    assert (out.kind, out.retry_after_seconds) == (api_client.RATE_LIMITED, 7)
    out = _ask(_json(429, {"retry_after_seconds": 12}))
    assert out.retry_after_seconds == 12


def test_a_429_without_the_limiters_fingerprint_is_the_platform_not_a_rate_limit():
    # the API's own limiter always sends Retry-After and a JSON body; anything else came from in front of it
    for response in (
        lambda r: httpx.Response(429, text="<html>Too Many Requests</html>"),
        lambda r: httpx.Response(429, json={"error": "slow down"}),
        lambda r: httpx.Response(429, text=""),
    ):
        out = _ask(response)
        assert out.kind == api_client.UNAVAILABLE and out.retry_after_seconds is None


def test_a_proxy_that_cannot_reach_the_api_is_unavailable_but_a_500_is_an_api_error():
    for status in (502, 503, 504):
        assert _ask(_json(status, {"detail": "x"})).kind == api_client.UNAVAILABLE
    assert _ask(_json(500, {"detail": "x"})).kind == api_client.API_ERROR
    assert _ask(_json(501, {"detail": "x"})).kind == api_client.API_ERROR


def test_the_real_limiter_response_is_still_a_rate_limit():
    out = _ask(_json(429, {"error": "Rate limit exceeded. Please slow down.", "retry_after_seconds": 48}, {"Retry-After": "48"}))
    assert (out.kind, out.retry_after_seconds) == (api_client.RATE_LIMITED, 48)


def test_server_error_and_bad_request_and_garbage_are_classified():
    assert _ask(_json(500, {"detail": "boom"})).kind == api_client.API_ERROR
    assert _ask(_json(400, {"error": "no"})).kind == api_client.REJECTED
    assert _ask(lambda r: httpx.Response(200, text="<html>")).kind == api_client.UNEXPECTED
    assert _ask(_json(200, ["not", "a", "dict"])).kind == api_client.UNEXPECTED
    assert _ask(_json(200, {"result": {"type": "something-new"}})).kind == api_client.UNEXPECTED
    assert _ask(_json(200, {"error": "x", "result": None})).kind == api_client.API_ERROR


def test_non_2xx_responses_are_logged_without_the_question(caplog):
    with caplog.at_level("WARNING", logger="api_client"):
        _ask(lambda r: httpx.Response(429, text="<html>Too Many Requests</html>", headers={"server": "edge"}))
    line = " ".join(r.getMessage() for r in caplog.records)
    assert "status=429" in line and "Too Many Requests" in line and "edge" in line
    assert "a question" not in line


def test_timeout_and_unreachable_are_classified():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("down", request=request)

    assert _ask(timeout).kind == api_client.TIMEOUT
    assert _ask(refused).kind == api_client.UNREACHABLE


def test_it_posts_the_question_to_query_without_credentials():
    seen = {}

    def handler(request):
        seen["url"], seen["body"], seen["headers"] = str(request.url), json.loads(request.content), dict(request.headers)
        return httpx.Response(200, json={"result": {"type": "reject", "message": "m"}})

    _ask(handler)
    assert seen["url"] == "https://api.example.test/query"
    assert seen["body"] == {"query": "a question"}
    assert "authorization" not in seen["headers"]


# ---- It stays a thin client ----------------------------------------------

def _imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


@pytest.mark.parametrize("name", ["app.py", "api_client.py"])
def test_live_ui_imports_nothing_from_the_api_packages(name):
    banned = {"src", "api", "supabase", "openai", "psycopg2"}
    found = {m.split(".")[0] for m in _imports(LIVE_UI / name)}
    assert not (found & banned), f"{name} imports {found & banned}"


def test_live_ui_reads_no_secrets():
    text = "".join((LIVE_UI / n).read_text() for n in ("app.py", "api_client.py"))
    for secret in ("OPENAI_API_KEY", "SUPABASE", "SIGNALS_TOKEN", "DATABASE_URL"):
        assert secret not in text
    assert set(re.findall(r'environ(?:\.get)?\(\s*"(\w+)"', text)) == {"API_URL"}


def test_live_ui_requirements_exclude_provider_libraries():
    reqs = (LIVE_UI / "requirements.txt").read_text().lower()
    for lib in ("openai", "supabase", "psycopg2", "fastapi"):
        assert lib not in reqs


def test_model_text_is_never_rendered_as_html():
    """unsafe_allow_html is allowed only for static strings and the answer wrapper div, never a model string."""
    tree = ast.parse((LIVE_UI / "app.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and any(k.arg == "unsafe_allow_html" for k in node.keywords):
            arg = node.args[0]
            if isinstance(arg, ast.JoinedStr):  # f-string: only the answer wrapper and the numeric note may interpolate
                src = ast.unparse(arg)
                assert "outcome.answer" not in src.replace("_plain(outcome.answer)", ""), src
                assert "citation" not in src and "sql" not in src.replace("outcome.rows", ""), src


# ---- The page runs and renders each state --------------------------------

def _run_page(monkeypatch, outcome):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(api_client, "ask", lambda *a, **k: outcome)
    at = AppTest.from_file(str(LIVE_UI / "app.py"), default_timeout=20)
    at.run()
    assert not at.exception
    at.text_area[0].set_value("which merchants?").run()
    at.button[-1].click().run()  # the Ask submit button
    assert not at.exception
    return at


def test_page_renders_the_sql_for_an_analytics_answer(monkeypatch):
    out = api_client.Outcome(api_client.ANALYTICS, 1.2, answer="A leads with $5.", sql="SELECT 1", rows=[{"m": "A"}])
    at = _run_page(monkeypatch, out)
    assert any(c.value == "SELECT 1" for c in at.code)
    assert any("A leads with \\$5." in m.value for m in at.markdown)


def test_page_tells_an_unavailable_api_apart_from_a_rate_limit(monkeypatch):
    out = api_client.Outcome(api_client.UNAVAILABLE, 0.2, message="m")
    shown = " ".join(w.value for w in _run_page(monkeypatch, out).warning)
    assert "not available right now" in shown and "waking up or restarting" in shown
    assert "limiting requests" not in shown


@pytest.mark.parametrize("kind,expected", [
    (api_client.RATE_LIMITED, "Try again in 9 seconds"),
    (api_client.TIMEOUT, "waking up"),
    (api_client.UNREACHABLE, "Could not reach the API"),
    (api_client.API_ERROR, "failed while answering"),
])
def test_page_explains_each_failure(monkeypatch, kind, expected):
    out = api_client.Outcome(kind, 0.1, message="m", retry_after_seconds=9)
    at = _run_page(monkeypatch, out)
    shown = " ".join([e.value for e in at.error] + [w.value for w in at.warning])
    assert expected in shown


# ---- the hybrid response ----------------------------------------------------

HYBRID_BODY = {
    "query": "q", "intent": "hybrid", "error": None,
    "result": {
        "type": "hybrid",
        "answer": "fraud_A has the highest fraud incidence.\n\nWhat the documents say: card-not-present exposure.",
        "analytics": {"answer": "x", "sql": "SELECT 1", "data_points": [{"m": "A"}], "confidence": 0.7},
        "rag": {"answer": "y", "citations": [{"source": "Bhatla", "page": 5, "preview": "p"}], "retrieval_status": "ok"},
        "parts": {"analytics": "ok", "rag": "ok"},
    },
}


def test_a_hybrid_answer_carries_both_kinds_of_evidence_and_its_parts():
    out = _ask(_json(200, HYBRID_BODY))
    assert out.kind == api_client.HYBRID
    assert out.sql == "SELECT 1" and out.rows == [{"m": "A"}] and out.citations[0]["source"] == "Bhatla"
    assert out.parts == {"analytics": "ok", "rag": "ok"} and "What the documents say" in out.answer


def test_a_hybrid_answer_with_one_half_missing_still_parses():
    body = json.loads(json.dumps(HYBRID_BODY))
    body["result"]["rag"] = None
    body["result"]["parts"]["rag"] = "failed"
    out = _ask(_json(200, body))
    assert out.kind == api_client.HYBRID and out.citations == [] and out.parts["rag"] == "failed"


def test_page_shows_the_sql_and_the_passages_for_a_hybrid_answer(monkeypatch):
    out = api_client.Outcome(api_client.HYBRID, 4.0, answer="Numbers.\n\nWhat the documents say: why.", sql="SELECT 1",
                             rows=[{"m": "A"}], citations=[{"source": "Bhatla", "page": 5, "preview": "p"}],
                             parts={"analytics": "ok", "rag": "ok"})
    at = _run_page(monkeypatch, out)
    assert any(c.value == "SELECT 1" for c in at.code)
    assert any("Bhatla, page 5" in m.value for m in at.markdown)
    assert not at.warning


def test_page_says_when_the_document_half_could_not_explain(monkeypatch):
    out = api_client.Outcome(api_client.HYBRID, 4.0, answer="Numbers.", sql="SELECT 1", rows=[{"m": "A"}],
                             parts={"analytics": "ok", "rag": "degraded"})
    at = _run_page(monkeypatch, out)
    assert any("did not have enough to explain" in w.value for w in at.warning)
    assert any(c.value == "SELECT 1" for c in at.code)          # the numbers are still shown with their SQL
