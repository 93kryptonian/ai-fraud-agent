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


def test_the_compound_question_is_a_labelled_known_limitation_not_a_scored_claim():
    compound = [c for c in load_cases() if c["category"] == "compound"]
    assert len(compound) == 1
    (c,) = compound
    assert c["scored"] is False and "single path" in c["note"]
    # nothing else is exempt from scoring
    assert [x["id"] for x in load_cases() if x.get("scored", True) is False] == [c["id"]]


def test_no_hybrid_route_is_expected_anywhere():
    assert {c["expected_route"] for c in load_cases()} <= {"analytics", "rag", "reject"}


def test_known_gaps_are_recorded_with_a_reason():
    gaps = [c for c in load_cases() if c.get("known_gap")]
    assert [c["id"] for c in gaps] == ["X2"]
    assert all(len(c["known_gap"]) > 20 for c in gaps)
