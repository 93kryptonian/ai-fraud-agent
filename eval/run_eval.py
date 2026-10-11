# eval/run_eval.py
"""
Live evaluation: send every case through the real API app (guardrails, router,
orchestrator) and compare each response with its expectations.

This is an integration check. It calls the real LLM and database when
credentials are present, so run it by hand:

    python -m eval.run_eval
"""

import argparse
import json
import os
import time
from pathlib import Path

import yaml

# The limiter is per client address and a run is one client: keep it out of the way.
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")

from eval.evaluator import check_case, observe  # noqa: E402
from eval.telemetry import EventCollector, summarize  # noqa: E402

HERE = Path(__file__).parent


def load_cases():
    return yaml.safe_load((HERE / "cases.yaml").read_text())


def run(out_path=None) -> int:
    # Credentials come from the git-ignored .env, loaded from an explicit path. Never printed.
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env")

    from fastapi.testclient import TestClient

    from api.main import create_app

    client = TestClient(create_app())
    results = []
    records = []

    for case in load_cases():
        with EventCollector() as collector:
            started = time.perf_counter()
            response = client.post("/query", json={"query": case["text"]})
            wall_ms = round((time.perf_counter() - started) * 1000)
        try:
            body = response.json()
        except ValueError:
            body = None
        result = check_case(case, observe(body, response.status_code))
        records.append({
            "id": case["id"], "category": case["category"], "decision": case["decision"],
            "scored": case.get("scored", True), "passed": result.passed,
            "observed_route": result.observed.route, "observed_behavior": result.observed.behavior,
            "failures": result.failures, "wall_ms": wall_ms, **summarize(collector.events),
        })
        scored = case.get("scored", True)
        if scored:
            results.append(result)

        mark = ("PASS" if result.passed else "FAIL") if scored else "INFO"
        print(f"{mark} {result.id}  route={result.observed.route}  behavior={result.observed.behavior}")
        for failure in result.failures:
            print(f"     - {failure}")

    passed = sum(r.passed for r in results)
    print(f"\nSUMMARY: {passed}/{len(results)} scored cases passed (INFO cases are known limitations, not counted)")
    if out_path:
        # Measurements and verdicts only: no answer text and no credentials.
        Path(out_path).write_text(json.dumps(records, indent=2))
        print(f"wrote {out_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live evaluation of the fraud-intelligence API app.")
    parser.add_argument("--out", help="write per-case measurements as JSON to this path")
    raise SystemExit(run(parser.parse_args().out))
