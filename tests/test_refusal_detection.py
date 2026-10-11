"""
The RAG refusal must be recognised in both languages.

Found while building the hybrid route: for an Indonesian question the model refuses in
Indonesian ("Maaf, dokumen yang tersedia tidak memberikan cukup informasi..."). The scorer
only knew the English sentence, so that refusal scored 0.415 and was shown as an answer.
"""

import pytest

from src.llm import llm_client
from src.llm.refusal import is_refusal
from src.llm.scoring import heuristic_score
from src.rag.insight_layer import generate_insight

CHUNKS = [{"content": "x" * 6000}]

REFUSALS = [
    "Sorry, the available documents do not provide enough information to answer your question.",
    "Sorry, the documents do not contain enough information.",
    "Maaf, dokumen yang tersedia tidak memberikan cukup informasi untuk menjawab pertanyaan Anda.",
    "Maaf, dokumen yang tersedia tidak memberikan informasi yang cukup untuk menjawab pertanyaan Anda.",
]
ANSWERS = [
    "Card-not-present fraud occurs when card details are used without the physical card.",
    "Penipuan kartu tidak hadir terjadi ketika data kartu digunakan tanpa kartu fisik.",
    "There is enough information in the report to show that fraud rose in 2023.",
    "",
]


@pytest.mark.parametrize("text", REFUSALS)
def test_refusals_are_recognised_in_both_languages(text):
    assert is_refusal(text) is True


@pytest.mark.parametrize("text", ANSWERS)
def test_real_answers_are_not_mistaken_for_refusals(text):
    assert is_refusal(text) is False


@pytest.mark.parametrize("text", REFUSALS)
def test_a_refusal_scores_zero_whatever_its_language(text):
    assert heuristic_score("what explains fraud?", text, CHUNKS) == 0.0


def test_an_indonesian_answer_still_scores_above_zero():
    assert heuristic_score("apa itu fraud", "Penipuan kartu terjadi ketika data kartu digunakan tanpa kartu fisik.", CHUNKS) > 0


@pytest.mark.parametrize("text", REFUSALS)
def test_no_insight_is_generated_for_a_refusal(monkeypatch, text):
    calls = []
    monkeypatch.setattr(llm_client.llm, "run", lambda *a, **k: calls.append(1) or "an insight")
    assert generate_insight(text, "some context", user_lang="en") is None
    assert calls == []
