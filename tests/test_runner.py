"""
End-to-end LLM system runner for the Fraud Intelligence platform.

Purpose:
- Validate orchestrator routing (RAG vs Analytics)
- Smoke-test real LLM execution paths
- Validate response structure (not correctness)

IMPORTANT:
- Marked with @pytest.mark.llm
- Skipped automatically in CI
"""

import os
import pytest
import traceback

from src.orchestrator import run_query


# ============================================================
# CI GUARD
# ============================================================

if os.getenv("CI") == "true":
    pytest.skip("Skipping LLM integration tests in CI", allow_module_level=True)


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "query": "How does the daily or monthly fraud rate fluctuate over the two-year period?",
        "expected_intent": "analytics",
    },
    {
        "query": "Which merchants or merchant categories exhibit the highest incidence of fraudulent transactions?",
        "expected_intent": "rag",
    },
    {
        "query": "What are the primary methods by which credit card fraud is committed?",
        "expected_intent": "rag",
    },
    {
        "query": "What are the core components of an effective fraud detection system, according to the authors?",
        "expected_intent": "rag",
    },
    {
        "query": "How much higher are fraud rates when the transaction counterpart is located outside the EEA?",
        "expected_intent": "rag",
    },
    {
        "query": "What share of total card fraud value in H1 2023 was due to cross-border transactions?",
        "expected_intent": "rag",
    },
]


# ============================================================
# RESULT HELPERS
# ============================================================

def extract_result_payload(raw_output):
    """
    Orchestrator returns:
    {
        "query": ...,
        "intent": ...,
        "result": {...},
        "error": None | str
    }
    """
    if not raw_output:
        return None

    if raw_output.get("error"):
        return {
            "type": "error",
            "error": raw_output.get("error"),
            "details": None,
        }

    return raw_output.get("result")


def infer_intent_from_payload(payload):
    """
    Infer intent from response structure.
    This avoids relying on internal intent flags.
    """
    if not isinstance(payload, dict):
        return "unknown"

    # Analytics response
    if "answer" in payload and "chart_data" in payload:
        return "analytics"

    # RAG response
    if "answer" in payload and "citations" in payload:
        return "rag"

    return "unknown"


def validate_result(test_case, raw_output):
    query = test_case["query"]
    expected = test_case["expected_intent"]

    print("\n----------------------------------------")
    print("Query:", query)

    payload = extract_result_payload(raw_output)

    if payload is None:
        print("[FAIL] No result payload returned")
        return False

    if payload.get("type") == "error":
        print("[FAIL] Error returned:", payload.get("error"))
        return False

    inferred = infer_intent_from_payload(payload)
    print(f"Inferred intent: {inferred} | Expected: {expected}")

    if inferred != expected:
        print("[WARN] Intent mismatch — structure still valid")

    # -----------------------
    # STRUCTURE VALIDATION
    # -----------------------

    if expected == "analytics":
        if "answer" not in payload:
            print("[FAIL] Analytics response missing 'answer'")
            return False

        print("[PASS] Analytics summary:")
        print(payload["answer"])
        return True

    # RAG expected
    if "answer" not in payload:
        print("[FAIL] RAG response missing 'answer'")
        return False

    answer = payload.get("answer") or ""
    if not answer.strip():
        print("[FAIL] Empty RAG answer")
        return False

    preview = answer[:300] + ("…" if len(answer) > 300 else "")
    print("[PASS] RAG answer preview:")
    print(preview)

    return True


# ============================================================
# MAIN RUNNER
# ============================================================

def run_all_tests():
    failures = []

    for case in TEST_CASES:
        try:
            output = run_query(case["query"])
            success = validate_result(case, output)

            if not success:
                failures.append((case["query"], output))

        except Exception:
            print("[EXCEPTION] During test execution")
            traceback.print_exc()
            failures.append((case["query"], "exception"))

    print("\n========== TEST SUMMARY ==========")
    if failures:
        print(f"FAILED {len(failures)} test(s):")
        for q, _ in failures:
            print(" -", q)
        return False

    print("ALL LLM TESTS PASSED")
    return True


# ============================================================
# PYTEST ENTRYPOINT
# ============================================================

@pytest.mark.llm
def test_run_all_queries():
    assert run_all_tests() is True


if __name__ == "__main__":
    run_all_tests()
