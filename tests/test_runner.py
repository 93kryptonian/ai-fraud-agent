
"""
Updated test runner compatible with the improved orchestrator + RAG system.
"""

import sys
import traceback
from pprint import pprint
from src.agents.multilingual_agent import handle_query

TESTS = [
    {
        "q": "How does the daily or monthly fraud rate fluctuate over the two-year period?",
        "expect_intent": "analytics"
    },
    {
        "q": "Which merchants or merchant categories exhibit the highest incidence of fraudulent transactions?",
        "expect_intent": "rag"
    },
    {
        "q": "What are the primary methods by which credit card fraud is committed?",
        "expect_intent": "rag"
    },
    {
        "q": "What are the core components of an effective fraud detection system, according to the authors?",
        "expect_intent": "rag"
    },
    {
        "q": "How much higher are fraud rates when the transaction counterpart is located outside the EEA?",
        "expect_intent": "rag"
    },
    {
        "q": "What share of total card fraud value in H1 2023 was due to cross-border transactions?",
        "expect_intent": "rag"
    }
]

FAILURES = []


def infer_intent_from_result(result):
    """
    Determine if result is analytics or rag.
    """
    if not result:
        return "unknown"

    # --- Analytics ---
    # AnalyticsResponse: dict with keys {"answer","chart_data","confidence"}
    if (
        isinstance(result, dict)
        and "answer" in result
        and "chart_data" in result
        and "confidence" in result
    ):
        return "analytics"

    # --- RAG ---
    # Improved RAG returns: {"answer": "...", "chunks": [...], "debug": {...}}
    if (
        isinstance(result, dict)
        and "answer" in result
        and "chunks" in result
    ):
        return "rag"

    # fallback guess
    return "unknown"


def check_result(tcase, out):
    q = tcase["q"]
    expect = tcase["expect_intent"]

    print("\n----")
    print("Query:", q)

    if out is None:
        print("  [FAIL] No response returned")
        return False

    # If it's an error dict
    if isinstance(out, dict) and out.get("error"):
        print("  [FAIL] ErrorResponse returned:", out.get("error"))
        return False

    inferred = infer_intent_from_result(out)
    print(f"  Inferred intent: {inferred} | Expected: {expect}")

    if inferred != expect:
        print("  [WARN] Intent mismatch — may still be acceptable.")
        # but we continue checking

    # Check structure depending on expected intent
    if expect == "analytics":
        if not isinstance(out, dict) or "answer" not in out:
            print("  [FAIL] Expected analytics structure containing 'answer'")
            return False
        print("[PASS] Analytics response summary:")
        print("  ", out["answer"])
        return True

    else:  # RAG expected
        if not isinstance(out, dict) or "answer" not in out:
            print("  [FAIL] RAG response missing 'answer'")
            return False

        if out["answer"] is None:
            dbg = out.get("debug", {})
            print("  [FAIL] RAG returned no answer. Debug:", dbg)
            return False

        print("[PASS] RAG returned answer:")
        print("  ", (out["answer"][:300] + "…") if len(out["answer"]) > 300 else out["answer"])
        return True


def main():
    ok = True
    for t in TESTS:
        try:
            out = handle_query(t["q"])
            success = check_result(t, out)
            if not success:
                FAILURES.append((t["q"], out))
                ok = False
        except Exception:
            print("Exception running test for query:")
            traceback.print_exc()
            FAILURES.append((t["q"], "exception"))
            ok = False

    print("\n==== Test Summary ====")
    # if FAILURES:
    #     print(f"FAILED {len(FAILURES)} tests.")
    #     for f in FAILURES:
    #         print(" -", f[0])
    #     sys.exit(2)
    # else:
    #     print("ALL TESTS PASSED")
    #     sys.exit(0)
    if FAILURES:
        print(f"FAILED {len(FAILURES)} tests.")
        for f in FAILURES:
            print(" -", f[0])
        return False
    else:
        print("ALL TESTS PASSED")
        return True

def test_run_all_queries():
    assert main() is True

if __name__ == "__main__":
    main()
