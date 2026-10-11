"""
The curated evaluation set (eval/cases.yaml): enough cases, in both languages,
across every category, with the compound question labelled as a known limitation.
"""

from collections import Counter

from eval.run_eval import load_cases

CATEGORIES = {"analytics", "rag", "reject", "ambiguous", "compound"}
DECISIONS = {"heuristic", "guardrail", "llm"}


def test_the_set_is_big_enough_and_covers_both_languages():
    cases = load_cases()
    assert len(cases) >= 20
    languages = Counter(c["language"] for c in cases)
    assert languages["en"] >= 8 and languages["id"] >= 4


def test_every_category_is_covered():
    assert {c["category"] for c in load_cases()} == CATEGORIES


def test_every_case_declares_its_category_and_deciding_layer():
    for c in load_cases():
        assert c["category"] in CATEGORIES, c["id"]
        assert c["decision"] in DECISIONS, c["id"]
        assert c["text"].strip(), c["id"]


def test_expected_routes_match_the_category():
    for c in load_cases():
        if c["category"] in {"analytics", "rag", "reject"}:
            assert c["expected_route"] == c["category"], c["id"]


def test_data_answers_must_carry_sql_and_document_answers_citations():
    for c in load_cases():
        if c["expected_route"] == "analytics":
            assert c.get("expect_sql") is True, c["id"]
        if c["expected_route"] == "rag":
            assert c.get("expect_citations") is True, c["id"]


def test_compound_questions_expect_the_hybrid_route_in_both_languages_with_a_supported_and_an_unsupported_case():
    compound = [c for c in load_cases() if c["category"] == "compound"]
    assert len(compound) >= 4
    for c in compound:
        assert c["expected_route"] == "hybrid" and c.get("scored", True) is True, c["id"]
        assert c.get("expect_sql") is True, c["id"]
        if c["expected_behavior"] == "answer":
            assert c.get("expect_citations") is True, c["id"]
    assert {c["language"] for c in compound} == {"en", "id"}
    # the document half is supported in some cases and honestly unsupported in others
    assert {c["expected_behavior"] for c in compound} == {"answer", "degraded"}


def test_only_compound_cases_expect_the_hybrid_route():
    for c in load_cases():
        assert (c["expected_route"] == "hybrid") == (c["category"] == "compound"), c["id"]
    assert {c["expected_route"] for c in load_cases()} <= {"analytics", "rag", "reject", "hybrid"}


def test_any_recorded_known_gap_carries_a_reason():
    # The mechanism stays (a strict expected failure in the offline gate); with no open gap the list is empty.
    for c in load_cases():
        if "known_gap" in c:
            assert len(c["known_gap"]) > 20, c["id"]
