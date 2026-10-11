# eval/run_eval.py
"""
Live evaluation: send every case through the real API app (guardrails, router,
orchestrator) and compare each response with its expectations.

This is an integration check. It calls the real LLM and database when
credentials are present, so run it by hand:

    python -m eval.run_eval
"""

import os
from pathlib import Path

import yaml

# The limiter is per client address and a run is one client: keep it out of the way.
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")

from eval.evaluator import check_case, observe  # noqa: E402

HERE = Path(__file__).parent


def load_cases():
    queries = yaml.safe_load((HERE / "queries.yaml").read_text())
    expectations = yaml.safe_load((HERE / "expectations.yaml").read_text())
    return [{**q, **expectations[q["id"]]} for q in queries]


def run() -> int:
    from fastapi.testclient import TestClient

    from api.main import create_app

    client = TestClient(create_app())
    results = []

    for case in load_cases():
        response = client.post("/query", json={"query": case["text"]})
        try:
            body = response.json()
        except ValueError:
            body = None
        result = check_case(case, observe(body, response.status_code))
        results.append(result)

        mark = "PASS" if result.passed else "FAIL"
        print(f"{mark} {result.id}  route={result.observed.route}  behavior={result.observed.behavior}")
        for failure in result.failures:
            print(f"     - {failure}")

    passed = sum(r.passed for r in results)
    print(f"\nSUMMARY: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(run())
