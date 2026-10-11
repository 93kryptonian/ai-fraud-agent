"""
The hybrid route: a question that needs numbers AND an explanation runs both
pipelines (analytics with its SQL, then RAG with its citations) instead of
answering only the first half.

Found by the evaluation set: the compound question used to get one path.
"""

import pytest

import src.orchestrator as orch
from src.orchestrator import detect_intent_heuristic

COMPOUND = "Which merchant categories have the highest fraud rates, and what might explain the pattern?"


# ---- the keyword rules -----------------------------------------------------

@pytest.mark.parametrize("query", [
    COMPOUND,
    "Why did the monthly fraud rate increase over time?",
    "Which merchants have the highest fraud and why?",
    "Merchant mana yang memiliki fraud tertinggi dan jelaskan penyebabnya?",
])
def test_numbers_plus_an_explanation_route_to_hybrid(query):
    assert detect_intent_heuristic(query) == ("hybrid", 0.90)


@pytest.mark.parametrize("query,expected", [
    ("Which merchants have the highest fraud incidence?", "analytics"),
    ("What is the monthly fraud rate?", "analytics"),
    ("How does the monthly fraud rate change over time?", "analytics"),
    ("Explain how cross-border card fraud differs from domestic card fraud.", "rag"),
    ("What are the main fraud typologies described in the documents?", "rag"),
    # a document question stays out of the transaction data even with a trend word and "why"
    ("Explain why cross-border fraud increased according to the report.", "analytics"),
])
def test_questions_that_need_only_one_job_are_unchanged(query, expected):
    assert detect_intent_heuristic(query)[0] == expected


def test_hybrid_is_decided_by_keywords_with_enough_confidence_to_skip_the_llm():
    intent, confidence = detect_intent_heuristic(COMPOUND)
    assert intent == "hybrid" and confidence >= 0.80


def test_the_llm_classifier_cannot_return_hybrid(monkeypatch):
    from src.llm import llm_client

    monkeypatch.setattr(llm_client.llm, "run", lambda prompt, **kw: '{"intent": "hybrid", "language": "en"}')
    with pytest.raises(orch.IntentClassificationError):
        orch.detect_intent_llm("anything")


# ---- running it ------------------------------------------------------------

ANALYTICS = {"answer": "fraud_A has the highest fraud incidence with 49 cases.",
             "data_points": [{"merchant": "fraud_A"}], "chart_data": [], "sql": "SELECT 1", "confidence": 0.7}
RAG = {"answer": "Merchant category risk is linked to card-not-present exposure.",
       "chunks": [{"id": "c1", "content": "long text"}], "context_text": "ctx",
       "citations": [{"source": "Bhatla", "page": 5, "preview": "p"}], "retrieval_status": "ok"}


@pytest.fixture
def stubs(monkeypatch):
    def install(analytics=None, rag=None, score=0.5):
        def run_analytics(q, lang):
            if isinstance(analytics, Exception):
                raise analytics
            return analytics if analytics is not None else dict(ANALYTICS)

        def run_rag(query_en, user_lang):
            if isinstance(rag, Exception):
                raise rag
            return dict(rag if rag is not None else RAG)

        monkeypatch.setattr(orch, "run_analytics", run_analytics)
        monkeypatch.setattr(orch, "run_rag", run_rag)
        monkeypatch.setattr(orch, "process_query", lambda q: (q, {"final_query_en": q, "lang": "en"}))
        monkeypatch.setattr(orch, "score_answer", lambda **kw: {"final_score": score})
        monkeypatch.setattr(orch, "generate_insight", lambda *a, **k: "an insight")
        monkeypatch.setattr(orch, "translate_en_to_id", lambda t: "[id] " + t)

    return install


def test_the_compound_question_runs_both_pipelines_and_returns_both_kinds_of_evidence(stubs):
    stubs()
    out = orch.run_query(COMPOUND)
    result = out["result"]
    assert (out["intent"], out["error"], result["type"]) == ("hybrid", None, "hybrid")
    assert result["parts"] == {"analytics": "ok", "rag": "ok"}
    assert ANALYTICS["answer"] in result["answer"] and "What the documents say:" in result["answer"]
    assert RAG["answer"] in result["answer"]
    assert result["analytics"]["sql"] == "SELECT 1"                          # the numbers stay grounded in SQL
    assert result["rag"]["citations"][0]["source"] == "Bhatla"               # the explanation stays grounded in citations
    assert "chunks" not in result["rag"] and "context_text" not in result["rag"]   # raw passages are not dumped into the response


def test_if_the_data_half_fails_the_answer_says_so_and_keeps_the_explanation(stubs):
    stubs(analytics=ConnectionError("db down"))
    out = orch.run_query(COMPOUND)
    result = out["result"]
    assert out["error"] is None and result["parts"] == {"analytics": "failed", "rag": "ok"}
    assert result["analytics"] is None and "could not be analyzed" in result["answer"]
    assert RAG["answer"] in result["answer"]


def test_if_the_document_half_fails_the_answer_says_so_and_keeps_the_numbers(stubs):
    stubs(rag=RuntimeError("rag down"))
    result = orch.run_query(COMPOUND)["result"]
    assert result["parts"] == {"analytics": "ok", "rag": "failed"}
    assert result["rag"] is None and "not available" in result["answer"]
    assert ANALYTICS["answer"] in result["answer"]


def test_if_both_halves_fail_the_request_fails_visibly(stubs):
    stubs(analytics=ConnectionError("db down"), rag=RuntimeError("rag down"))
    out = orch.run_query(COMPOUND)
    assert (out["intent"], out["result"], out["error"]) == ("hybrid", None, "Hybrid pipeline failed.")


def test_weak_document_evidence_is_reported_as_degraded_not_ok(stubs):
    stubs(score=0.05)
    assert orch.run_query(COMPOUND)["result"]["parts"]["rag"] == "degraded"
    stubs(rag={**RAG, "retrieval_status": "unavailable"}, score=0.9)
    assert orch.run_query(COMPOUND)["result"]["parts"]["rag"] == "degraded"


def test_an_analytics_result_with_no_rows_is_no_data_not_ok(stubs):
    stubs(analytics={**ANALYTICS, "data_points": None})
    assert orch.run_query(COMPOUND)["result"]["parts"]["analytics"] == "no_data"


def test_the_sections_are_worded_in_indonesian_for_an_indonesian_question(stubs):
    stubs(analytics=ConnectionError("down"))
    text = "Merchant mana yang memiliki fraud tertinggi dan jelaskan penyebabnya?"
    answer = orch.run_query(text)["result"]["answer"]
    assert "Data transaksi tidak dapat dianalisis" in answer and "Menurut dokumen:" in answer


def test_the_intent_event_reports_hybrid_decided_by_the_keyword_rules(stubs, monkeypatch):
    stubs()
    events = []
    monkeypatch.setattr(orch, "emit_event", lambda name, **kw: events.append((name, kw)))
    orch.run_query(COMPOUND)
    (meta,) = [kw["metadata"] for name, kw in events if name == "intent.completed"]
    assert (meta["intent"], meta["route"], meta["method"]) == ("hybrid", "hybrid", "heuristic")


def test_single_path_questions_still_run_one_pipeline(stubs, monkeypatch):
    stubs()
    called = []
    monkeypatch.setattr(orch, "_run_rag_branch", lambda q, lang: called.append("rag") or {"intent": "rag"})
    orch.run_query("Which merchants have the highest fraud incidence?")
    assert called == []                                                      # analytics only: RAG was not run


# ---- what the document half is asked --------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("Which merchants have the highest fraud incidence, and why does card-not-present fraud occur?",
     "why does card-not-present fraud occur?"),
    (COMPOUND, "what might explain the pattern?"),
    ("Which merchants have the highest fraud; explain how card-not-present fraud works?",
     "explain how card-not-present fraud works?"),
])
def test_the_document_half_is_asked_the_explanation_clause_when_it_can_stand_alone(query, expected):
    assert orch.explanation_clause(query) == expected


@pytest.mark.parametrize("query", [
    "Merchant mana yang memiliki fraud tertinggi dan jelaskan penyebabnya?",   # clause too short to stand alone
    "Which merchants have the highest fraud and why?",
    "Which merchants have the highest fraud incidence?",                        # no explanation word at all
])
def test_otherwise_the_document_half_is_asked_the_whole_question(query):
    assert orch.explanation_clause(query) == query


def test_the_rag_half_really_receives_the_clause_and_the_analytics_half_the_whole_question(stubs, monkeypatch):
    stubs()
    asked = {}
    real_rag, real_analytics = orch.run_rag, orch.run_analytics
    monkeypatch.setattr(orch, "run_rag", lambda query_en, user_lang: asked.setdefault("rag", query_en) and real_rag(query_en, user_lang))
    monkeypatch.setattr(orch, "run_analytics", lambda q, lang: asked.setdefault("analytics", q) and real_analytics(q, lang))
    text = "Which merchants have the highest fraud incidence, and why does card-not-present fraud occur?"
    orch.run_query(text)
    assert asked["analytics"] == text
    assert asked["rag"] == "why does card-not-present fraud occur?"


def test_an_indonesian_refusal_from_the_document_half_is_degraded_not_ok(stubs):
    # the model refuses in the user's language; the scorer must treat that as no evidence (not 0.415)
    stubs(rag={**RAG, "answer": "Maaf, dokumen yang tersedia tidak memberikan cukup informasi untuk menjawab pertanyaan Anda."},
          score=0.0)
    assert orch.run_query(COMPOUND)["result"]["parts"]["rag"] == "degraded"
