"""docs/evaluation.md must describe the case set that actually exists."""

from collections import Counter
from pathlib import Path

from eval.run_eval import load_cases

DOC = (Path(__file__).resolve().parent.parent / "docs" / "evaluation.md").read_text(encoding="utf-8")


def test_the_documented_counts_match_cases_yaml():
    cases = load_cases()
    scored = sum(1 for c in cases if c.get("scored", True))
    languages = Counter(c["language"] for c in cases)
    scope = "all scored" if scored == len(cases) else f"{scored} scored, {len(cases) - scored} known limitation"
    assert f"<!-- eval-cases: {len(cases)} cases, {scored} scored -->" in DOC
    assert f"**{len(cases)} cases ({scope})**" in DOC
    assert f"{languages['en']} English, {languages['id']} Indonesian" in DOC


def test_the_documented_category_table_matches_cases_yaml():
    for category, n in Counter(c["category"] for c in load_cases()).items():
        assert f"| {category} | {n} |" in DOC, category


def test_the_document_no_longer_claims_what_is_not_true():
    for stale in ("Behavioral regressions block deployment", "~20–50", "Hallucination Rate\n\n**Question:**"):
        assert stale not in DOC.replace("An earlier version of this document said behavioral regressions block deployment.", "")
