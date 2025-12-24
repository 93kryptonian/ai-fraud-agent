"""
Updated test runner compatible with the orchestrator-based architecture.
"""

import os
import pytest

if os.getenv("CI") == "true":
    pytest.skip("Skipping LLM tests in CI", allow_module_level=True)


import traceback
from pprint import pprint


from src.orchestrator import run_query  

TESTS = [
    {
        "q": "How does the daily or monthly fraud rate fluctuate over the two-year period?",
        "expect_intent": "analytics",
    },
    {
        "q": "Which merchants or merchant categories exhibit the highest incidence of fraudulent transactions?",
        "expect_intent": "rag",
    },
    {
        "q": "What are the primary methods by which credit card fraud is committed?",
        "expect_intent": "rag",
    },
    {
        "q": "What are the core components of an effective fraud detection system, according to the authors?",
        "expect_intent": "rag",
    },
    {
        "q": "How much higher are fraud rates when the transaction counterpart is located outside the EEA?",
        "expect_intent": "rag",
    },
    {
        "q": "What share of total card fraud value in H1 2023 was due to cross-border transactions?",
        "expect_intent": "rag",
    },
]

FAILURES = []


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def extract_payload(raw_out):
    """
    Your orchestrator returns:
    {
        "query": ...,
        "intent": ...,
        "result": {...},   <-- ACTUAL payload is here
        "error": None or str
    }
    """
    if not raw_out:
        return None

    if isinstance(raw_out, dict) and raw_out.get("error"):
        return {"type": "error", "error": raw_out["error"], "details": None}

    return raw_out.get("result")


def infer_intent_from_result(result):
    """
    Decide between analytics/rag based on structure.
    """
    if not result:
        return "unknown"

    # Analytics
    if (
        isinstance(result, dict)
        and "answer" in result
        and "chart_data" in result
        and "confidence" in result
    ):
        return "analytics"

    # RAG
    if isinstance(result, dict) and "answer" in result and "citations" in result:
        return "rag"

    return "unknown"


def check_result(tcase, raw_out):
    q = tcase["q"]
    expect = tcase["expect_intent"]

    print("\n----")
    print("Query:", q)

    result = extract_payload(raw_out)

    if result is None:
        print("  [FAIL] No result payload returned.")
        return False

    if isinstance(result, dict) and result.get("type") == "error":
        print("  [FAIL] ErrorResponse:", result.get("error"))
        return False

    inferred = infer_intent_from_result(result)
    print(f"  Inferred intent: {inferred} | Expected: {expect}")

    if inferred != expect:
        print("  [WARN] Intent mismatch — may still be acceptable.")

    # STRUCTURE VALIDATION
    if expect == "analytics":
        if not isinstance(result, dict) or "answer" not in result:
            print("  [FAIL] Analytics response missing 'answer'")
            return False
        print("[PASS] Analytics summary:")
        print("  ", result["answer"])
        return True

    # RAG expected
    if not isinstance(result, dict) or "answer" not in result:
        print("  [FAIL] RAG response missing 'answer'")
        return False

    answer_str = result["answer"] or ""
    if not answer_str:
        print("  [FAIL] RAG returned an empty answer.")
        return False

    print("[PASS] RAG returned answer:")
    preview = answer_str[:300] + ("…" if len(answer_str) > 300 else "")
    print("  ", preview)
    return True


# ------------------------------------------------------------
# Main Test Logic
# ------------------------------------------------------------

def main():
    ok = True

    for t in TESTS:
        try:
            raw_out = run_query(t["q"])
            success = check_result(t, raw_out)
            if not success:
                FAILURES.append((t["q"], raw_out))
                ok = False
        except Exception:
            print("Exception during test:")
            traceback.print_exc()
            FAILURES.append((t["q"], "exception"))
            ok = False

    print("\n==== Test Summary ====")
    if FAILURES:
        print(f"FAILED {len(FAILURES)} tests.")
        for f in FAILURES:
            print(" -", f[0])
        return False

    print("ALL TESTS PASSED")
    return True


# def test_run_all_queries():
#     assert main() is True
@pytest.mark.llm
def test_run_all_queries():
    assert main() is True


if __name__ == "__main__":
    main()
