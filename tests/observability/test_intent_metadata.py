# tests/observability/test_intent_metadata.py
"""
Tests for M6.6: intent metadata enrichment (src/orchestrator.py).

Covers detect_intent()'s new (intent, lang, confidence, method) return
signature: confidence is a real heuristic float on the fast path, and
honestly None on the LLM path — never invented, matching the same
"don't fake a number" rule used for duration_ms elsewhere.
"""


def test_intent_completed_has_real_confidence_on_heuristic_path():
    from src.orchestrator import detect_intent

    intent, lang, confidence, method = detect_intent("what is the fraud rate trend", "en")

    assert method == "heuristic"
    assert isinstance(confidence, float)
    assert confidence >= 0.80


def test_intent_completed_has_none_confidence_on_llm_path(monkeypatch):
    from src.orchestrator import detect_intent

    monkeypatch.setattr(
        "src.orchestrator.detect_intent_llm",
        lambda query: ("rag", "en"),
    )

    # A query with no heuristic keyword match at all forces the LLM path.
    intent, lang, confidence, method = detect_intent("zzz qqq", "en")

    assert method == "llm"
    assert confidence is None  # honestly unknown, never invented
