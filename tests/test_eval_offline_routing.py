"""
Offline routing gate: runs the eval cases through the PRODUCTION routing code
with no credentials and no network.

What this proves: the deterministic rules (keyword intent heuristic, input
guardrails) send each labelled case where eval/cases.yaml says, and cases that
need the LLM really do defer to it.

What it does NOT prove: that the real LLM classifies ambiguous questions
correctly, that retrieval finds anything, or that any final answer is right.
That is the live evaluation (python -m eval.run_eval).

It calls src.orchestrator.detect_intent and src.safety.guardrails.validate_query
themselves; there is no second copy of the rules here. Any call that reaches the
LLM is recorded and fails the test, so "offline" is enforced, not assumed.
"""

import json

import pytest

from eval.run_eval import load_cases
from src.llm import llm_client
from src.orchestrator import detect_intent, detect_intent_heuristic
from src.safety.guardrails import validate_query

CASES = load_cases()


def of(decision):
    """Cases for one deciding layer. A case with a recorded known_gap is expected to fail, STRICTLY:
    the day the gap is fixed this test fails until the record is removed, so the record cannot go stale."""
    return [
        pytest.param(
            c, id=c["id"],
            marks=[pytest.mark.xfail(strict=True, reason=c["known_gap"])] if c.get("known_gap") else [],
        )
        for c in CASES if c["decision"] == decision
    ]


@pytest.fixture
def llm_calls(monkeypatch):
    """Record every LLM call; an offline case must make none. (Recording, because
    the code under test swallows exceptions and would hide a raised error.)"""
    calls = []

    def fake_run(prompt, **kwargs):
        calls.append(kwargs.get("purpose"))
        raise ConnectionError("offline routing check: no LLM")

    monkeypatch.setattr(llm_client.llm, "run", fake_run)
    return calls


def llm_says(monkeypatch, label, language="en"):
    calls = []

    def fake_run(prompt, **kwargs):
        calls.append(kwargs.get("purpose"))
        return json.dumps({"intent": label, "language": language})

    monkeypatch.setattr(llm_client.llm, "run", fake_run)
    return calls


# ---- cases the keyword rules decide ---------------------------------------

@pytest.mark.parametrize("case", of("heuristic"))
def test_keyword_rules_route_the_case_without_the_llm(case, llm_calls):
    intent, _lang, confidence, method = detect_intent(case["text"], case["language"])
    assert (intent, method) == (case["expected_route"], "heuristic"), case["text"]
    assert confidence is not None and confidence >= 0.8
    assert llm_calls == []


@pytest.mark.parametrize("case", [c for c in CASES if c["category"] == "compound"], ids=lambda c: c["id"])
def test_compound_questions_take_the_hybrid_route_without_the_llm(case, llm_calls):
    intent, _lang, confidence, method = detect_intent(case["text"], case["language"])
    assert (intent, method) == ("hybrid", "heuristic") and confidence >= 0.8
    assert llm_calls == []


# ---- cases the guardrails reject ------------------------------------------

@pytest.mark.parametrize("case", of("guardrail"))
def test_guardrails_reject_the_case_without_the_llm(case, llm_calls):
    ok, _message, _lang, reason = validate_query(case["text"])
    assert ok is False, f"guardrails let through: {case['text']}"
    assert reason is not None
    assert llm_calls == []


# ---- cases that need the LLM: they must defer, not be guessed -------------

@pytest.mark.parametrize("case", of("llm"))
def test_ambiguous_cases_defer_to_the_llm_classifier(case, monkeypatch):
    calls = llm_says(monkeypatch, case["expected_route"])
    intent, _lang, confidence, method = detect_intent(case["text"], case["language"])
    assert method == "llm" and confidence is None, "a keyword rule decided a case labelled as LLM-only"
    assert calls == ["intent_classification"]
    assert intent == case["expected_route"]   # only shows the LLM's label is used; whether the real LLM agrees is the live run


# ---- the 0.40 fallback: how ambiguity can hide ----------------------------

def test_unrecognized_questions_get_a_weak_rag_guess_not_a_decision():
    intent, confidence = detect_intent_heuristic("Tell me something useful about card fraud.")
    assert (intent, confidence) == ("rag", 0.40)


def test_a_weak_guess_is_handed_to_the_llm_and_the_llm_wins(monkeypatch):
    calls = llm_says(monkeypatch, "analytics")
    intent, _lang, _conf, method = detect_intent("Show me fraud by merchant.", "en")
    assert (intent, method) == ("analytics", "llm") and len(calls) == 1


def test_an_llm_failure_is_reported_as_the_weak_heuristic_guess_not_as_an_llm_decision(llm_calls):
    """A failed classifier used to report method="llm": indistinguishable from a real RAG decision."""
    intent, _lang, confidence, method = detect_intent("Show me fraud by merchant.", "en")
    assert llm_calls == ["intent_classification"]
    assert (intent, method, confidence) == ("rag", "heuristic", 0.40)


@pytest.mark.parametrize("reply", ['{"intent": "hybrid", "language": "en"}', "not json at all", '{"language": "en"}'])
def test_an_unusable_llm_label_is_never_trusted_and_is_reported_as_a_failure(monkeypatch, reply):
    monkeypatch.setattr(llm_client.llm, "run", lambda prompt, **kw: reply)
    intent, _lang, confidence, method = detect_intent("Show me fraud by merchant.", "en")
    assert (intent, method, confidence) == ("rag", "heuristic", 0.40)


def test_an_llm_rejection_is_honored(monkeypatch):
    llm_says(monkeypatch, "reject")
    intent, *_ = detect_intent("What is the best recipe for chocolate cake?", "en")
    assert intent == "reject"
