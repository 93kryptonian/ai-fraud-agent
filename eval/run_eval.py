# eval/run_eval.py

import yaml
from pprint import pprint
from src.orchestrator import run_query


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def check_contains(text: str, keywords):
    text = text.lower()
    return all(k.lower() in text for k in keywords)


def infer_behavior(result):
    if not result:
        return "unknown"

    if result.get("type") == "error":
        return "refusal"

    if "answer" in result and result.get("answer"):
        return "answer"

    return "unknown"


def run():
    queries = load_yaml("eval/queries.yaml")
    expectations = load_yaml("eval/expectations.yaml")

    results = []

    for q in queries:
        qid = q["id"]
        text = q["text"]

        print("\n" + "=" * 60)
        print(f"Query {qid}: {text}")

        out = run_query(text)
        intent = out.get("intent")
        result = out.get("result")

        exp = expectations[qid]

        # ---- checks ----
        route_ok = intent == exp["expected_route"]
        behavior = infer_behavior(result)
        behavior_ok = behavior == exp["expected_behavior"]

        include_ok = True
        if "must_include" in exp and behavior == "answer":
            include_ok = check_contains(
                result.get("answer", ""),
                exp["must_include"],
            )

        lang_ok = True
        if behavior == "answer":
            lang_ok = result.get("_lang") == exp.get("language")

        passed = all([route_ok, behavior_ok, include_ok, lang_ok])

        outcome = {
            "id": qid,
            "route_ok": route_ok,
            "behavior_ok": behavior_ok,
            "include_ok": include_ok,
            "language_ok": lang_ok,
            "passed": passed,
        }

        pprint(outcome)
        results.append(outcome)

    # ---- summary ----
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} passed")

    if passed != total:
        raise SystemExit("Evaluation failed")


if __name__ == "__main__":
    run()
