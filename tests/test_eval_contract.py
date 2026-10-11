"""
The evaluation contract (eval/evaluator.py).

An evaluator that cannot fail is not a gate. These tests feed it the three real
/query response shapes and then deliberately wrong responses: each wrong one
must be reported as a failure, never quietly passed.
"""

import copy

import pytest

from eval.evaluator import ANSWER, DEGRADED, ERROR, RATE_LIMITED, REFUSAL, answer_language, check_case, observe
from eval.run_eval import load_cases

ANALYTICS = {
    "query": "q", "intent": "analytics", "error": None,
    "result": {"type": "analytics", "analytics": {
        "answer": "The fraud rate by month rises over time.", "sql": "SELECT 1",
        "data_points": [{"m": 1}], "chart_data": [], "confidence": 0.73}},
}
RAG_EN = {
    "query": "q", "intent": "rag", "error": None,
    "result": {"type": "rag", "answer": "Card-not-present fraud is the use of card details without the card.",
               "citations": [{"source": "Bhatla", "page": 5, "preview": "p"}],
               "retrieval_status": "ok", "score": {"final_score": 0.5}},
}
RAG_ID = copy.deepcopy(RAG_EN)
RAG_ID["result"]["answer"] = "Card-not-present adalah penipuan yang terjadi dengan menggunakan data kartu tanpa kartu fisik."
REJECT = {"query": "q", "intent": "reject", "error": None,
          "result": {"type": "reject", "message": "Potential prompt-injection attempt detected."}}

CASE_ANALYTICS = {"id": "A", "expected_route": "analytics", "expected_behavior": "answer",
                  "must_include": ["fraud rate", "time"], "language": "en", "expect_sql": True}
CASE_RAG_ID = {"id": "R", "expected_route": "rag", "expected_behavior": "answer",
               "must_include": ["card-not-present"], "language": "id", "expect_citations": True}
CASE_REJECT = {"id": "X", "expected_route": "reject", "expected_behavior": "refusal"}


def run(case, body, status=200):
    return check_case(case, observe(body, status))


# ---- the three real shapes are read correctly ----------------------------

def test_analytics_answer_is_read_from_its_nested_location():
    o = observe(ANALYTICS)
    assert (o.route, o.behavior, o.has_sql) == ("analytics", ANSWER, True)


def test_a_rejection_is_a_refusal_not_unknown():
    assert (observe(REJECT).route, observe(REJECT).behavior) == ("reject", REFUSAL)


def test_rag_answer_carries_citations():
    o = observe(RAG_EN)
    assert (o.route, o.behavior, o.citations) == ("rag", ANSWER, 1)


@pytest.mark.parametrize("case,body", [(CASE_ANALYTICS, ANALYTICS), (CASE_RAG_ID, RAG_ID), (CASE_REJECT, REJECT)])
def test_correct_responses_pass(case, body):
    result = run(case, body)
    assert result.passed, result.failures


def test_language_comes_from_the_answer_text_not_a_field_the_api_never_sends():
    assert "_lang" not in str(RAG_ID)
    assert run(CASE_RAG_ID, RAG_ID).language_ok


# ---- deliberately wrong responses must fail -------------------------------

def _wrong(case, body, **expect):
    result = run(case, body)
    assert not result.passed
    for check, ok in expect.items():
        assert getattr(result, check) is ok, (check, result.failures)
    return result


def test_wrong_route_fails():
    _wrong(CASE_ANALYTICS, RAG_EN, route_ok=False)


def test_answering_when_a_refusal_was_expected_fails():
    _wrong(CASE_REJECT, ANALYTICS, route_ok=False, behavior_ok=False)


def test_refusing_when_an_answer_was_expected_fails():
    _wrong(CASE_ANALYTICS, REJECT, route_ok=False, behavior_ok=False, include_ok=False)


def test_missing_required_fact_fails():
    body = copy.deepcopy(ANALYTICS)
    body["result"]["analytics"]["answer"] = "Totals are listed below."
    _wrong(CASE_ANALYTICS, body, include_ok=False)


def test_english_answer_to_an_indonesian_question_fails():
    _wrong(CASE_RAG_ID, RAG_EN, language_ok=False)


def test_analytics_answer_without_sql_fails():
    body = copy.deepcopy(ANALYTICS)
    body["result"]["analytics"]["sql"] = None
    _wrong(CASE_ANALYTICS, body, grounding_ok=False)


def test_rag_answer_without_citations_fails():
    body = copy.deepcopy(RAG_ID)
    body["result"]["citations"] = []
    _wrong(CASE_RAG_ID, body, grounding_ok=False)


def test_degraded_retrieval_is_not_an_answer():
    body = copy.deepcopy(RAG_ID)
    body["result"]["retrieval_status"] = "unavailable"
    body["result"]["answer"] = "Sorry, the document search is temporarily unavailable."
    assert observe(body).behavior == DEGRADED
    _wrong(CASE_RAG_ID, body, behavior_ok=False)


def test_empty_answers_are_errors():
    for body, path in ((ANALYTICS, ("analytics", "answer")), (RAG_EN, ("answer",))):
        broken = copy.deepcopy(body)
        target = broken["result"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = "  "
        assert observe(broken).behavior == ERROR


@pytest.mark.parametrize("body,status,behavior", [
    ({"error": "Analytics pipeline failed.", "intent": "analytics", "result": None}, 200, ERROR),
    ({"detail": "boom"}, 500, ERROR),
    ({"error": "Rate limit exceeded."}, 429, RATE_LIMITED),
    ("<html>", 200, ERROR),
    (None, 200, ERROR),
    ({"intent": "analytics", "result": {"type": "rag"}}, 200, ERROR),
    ({"intent": "mystery", "result": {"type": "mystery"}}, 200, ERROR),
])
def test_errors_and_malformed_responses_never_pass(body, status, behavior):
    assert observe(body, status).behavior == behavior
    assert not run(CASE_ANALYTICS, body, status).passed


# ---- the language judge ---------------------------------------------------

def test_answer_language():
    assert answer_language("The fraud rate is the share of transactions that are fraudulent.") == "en"
    assert answer_language("Penipuan adalah tindakan yang dilakukan untuk mendapatkan keuntungan dari kartu.") == "id"
    assert answer_language("Fraud.") == "unknown"
    assert answer_language("") == "unknown"


# ---- the case files -------------------------------------------------------

def test_every_case_has_valid_expectations():
    cases = load_cases()
    assert cases
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        assert c["expected_route"] in {"analytics", "rag", "reject"}, c["id"]
        assert c["expected_behavior"] in {"answer", "refusal", "degraded"}, c["id"]
        assert c["language"] in {"en", "id"}, c["id"]
        if c["expected_route"] == "reject":
            assert c["expected_behavior"] == "refusal", c["id"]


# ---- fixes made after the first live run (answers read, see eval report) ---

def test_a_short_correct_indonesian_analytics_answer_is_judged_indonesian():
    # the real answer to A6/A7; the first judge found one Indonesian word in it and said "unknown"
    assert answer_language("fraud_Rau and Sons memiliki insiden fraud tertinggi dengan 49 kasus (fraud rate ~1.97%).") == "id"


def test_a_short_correct_english_analytics_answer_is_judged_english():
    assert answer_language("fraud_Rau and Sons has the highest fraud incidence with 49 cases (fraud rate ~1.97%).") == "en"


def test_the_wider_judge_still_rejects_the_wrong_language():
    id_case = {"id": "A6", "expected_route": "analytics", "expected_behavior": "answer", "language": "id"}
    english = copy.deepcopy(ANALYTICS)
    english["result"]["analytics"]["answer"] = "fraud_Rau and Sons has the highest fraud incidence with 49 cases."
    assert not run(id_case, english).language_ok
    en_case = dict(id_case, language="en")
    indonesian = copy.deepcopy(ANALYTICS)
    indonesian["result"]["analytics"]["answer"] = "fraud_Rau and Sons memiliki insiden fraud tertinggi dengan 49 kasus."
    assert not run(en_case, indonesian).language_ok


def test_a_contentless_aggregate_answer_is_not_treated_as_a_good_answer():
    # the real answer to A5 ("how many fraudulent transactions?"): no number, no mention of fraud
    case = {"id": "A5", "expected_route": "analytics", "expected_behavior": "answer",
            "must_include": ["fraud"], "language": "en", "expect_sql": True}
    body = copy.deepcopy(ANALYTICS)
    body["result"]["analytics"]["answer"] = "Analysis processed 1 rows and 1 columns."
    result = run(case, body)
    assert not result.include_ok and not result.passed


def test_alternatives_accept_a_translated_term_but_still_require_one_of_them():
    case = {"id": "R2", "expected_route": "rag", "expected_behavior": "answer", "language": "id",
            "must_include": [["card-not-present", "tidak hadir"]], "expect_citations": True}
    translated = copy.deepcopy(RAG_ID)
    translated["result"]["answer"] = "Penipuan kartu yang tidak hadir adalah penipuan yang terjadi tanpa kartu fisik dan dengan data kartu."
    assert run(case, translated).passed
    unrelated = copy.deepcopy(RAG_ID)
    unrelated["result"]["answer"] = "Penipuan adalah tindakan yang dilakukan untuk mendapatkan keuntungan dengan cara curang."
    assert not run(case, unrelated).include_ok
