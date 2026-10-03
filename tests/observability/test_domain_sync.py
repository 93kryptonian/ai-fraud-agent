# tests/observability/test_domain_sync.py
"""
Domain-sync tests (post-M8 reconciliation).

The M7.2 policy (src/observability/dimensions.py) declares closed value
domains for dimensions. Several of those values are also defined where the
telemetry is produced. Nothing used to tie the two together, so a new purpose,
reason or route in code would silently become the label `other`. These tests
read each authoritative definition from the CODE and require it to equal the
policy domain, in both directions.
"""

import ast
import pathlib
import re
import typing

from src.observability import dimensions as dim

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _policy_domain(event, key):
    return set(dim.POLICY[(event, key)].domain)


# =============================================================================
# The three domains duplicated across modules
# =============================================================================

def test_llm_purposes_in_code_equal_the_policy_domain():
    found = set()
    for path in (ROOT / "src").rglob("*.py"):
        if path.name == "dimensions.py":
            continue
        found |= set(re.findall(r'purpose="([a-z_]+)"', path.read_text(encoding="utf-8")))

    assert found == set(dim._PURPOSES), (
        f"purpose literals in code {sorted(found)} != policy {sorted(dim._PURPOSES)}"
    )
    for event in ("llm.completed", "llm.failed", "llm.fallback"):
        assert _policy_domain(event, "purpose") == found


def test_guardrail_reasons_in_code_equal_the_policy_domain():
    from src.safety.guardrails import GuardrailReason

    in_code = set(typing.get_args(GuardrailReason))
    for event in ("guardrails.completed", "guardrails.blocked"):
        assert _policy_domain(event, "reason") == in_code | {dim.NONE_VALUE}  # none = "not blocked"


def test_application_routes_agree_across_router_limiter_and_policy():
    from api.routers import router
    from src.safety.rate_limit import _APPLICATION_ROUTES

    router_paths = {r.path for r in router.routes}
    source = _read("api/routers.py")
    guardrail_routes = set(re.findall(r'_run_guardrails\("(/[a-z]+)"', source))
    completed_routes = set(re.findall(r'"route": "(/[a-z]+)"', source))

    assert router_paths == guardrail_routes == completed_routes == set(_APPLICATION_ROUTES)
    for event in ("request.started", "request.completed"):
        assert _policy_domain(event, "route") == router_paths
    # rate_limit.blocked adds exactly the overflow bucket for unmapped paths
    assert _policy_domain("rate_limit.blocked", "route") == router_paths | {"other"}


# =============================================================================
# The remaining closed domains
# =============================================================================

def _returned_strings(path, function):
    tree = ast.parse(_read(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return {
                n.value.value for n in ast.walk(node)
                if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)
            }
    raise AssertionError(f"{function} not found in {path}")


def test_analytics_intents_in_code_equal_the_policy_domain():
    returned = _returned_strings("src/analytics/fraud_analytics.py", "classify_analytics_intent")
    for event in ("analytics.completed", "analytics.sql.completed", "analytics.sql.failed"):
        # "none" is the emitted value for a non-fraud query that never got an intent
        assert _policy_domain(event, "intent") == returned | {dim.NONE_VALUE}


def test_request_intents_and_intent_methods_in_code_equal_the_policy_domain():
    source = _read("src/orchestrator.py")
    tree = ast.parse(source)
    sets = [
        {e.value for e in n.elts}
        for n in ast.walk(tree)
        if isinstance(n, ast.Set) and n.elts and all(isinstance(e, ast.Constant) for e in n.elts)
    ]
    assert dim._REQUEST_INTENTS in [s for s in sets if all(isinstance(v, str) for v in s)]

    methods = set(re.findall(r'(?:conf|None), "(heuristic|llm)"', source))
    assert methods == _policy_domain("intent.completed", "method")


def test_retrieval_ranking_and_fallback_values_in_code_equal_the_policy_domain():
    retriever = _read("src/rag/retriever_direct.py")
    assert set(re.findall(r'"reason": "([a-z_]+)"', retriever)) == _policy_domain("retrieval.skipped", "reason")
    methods = set(re.findall(r'"retrieval_method": "([a-z_]+)"', retriever))
    for event in ("retrieval.completed", "retrieval.failed", "retrieval.skipped"):
        assert methods == _policy_domain(event, "retrieval_method")
    assert set(re.findall(r'"reranker": "([a-z_]+)"', retriever)) == _policy_domain("ranking.completed", "reranker")

    llm_client = _read("src/llm/llm_client.py")
    assert set(re.findall(r'"reason": "([a-z_]+)"', llm_client)) == _policy_domain("llm.fallback", "reason")


def test_cost_status_values_in_code_equal_the_policy_domain():
    values = set(re.findall(r'"cost_status": "([a-z_]+)"', _read("src/observability/cost.py")))
    assert values == _policy_domain("request.completed", "cost_status")


def test_request_statuses_in_code_equal_the_policy_domain():
    statuses = set(re.findall(r'"request\.completed", step="request", status="([a-z]+)"', _read("api/routers.py")))
    assert statuses == _policy_domain("request.completed", "status")


# =============================================================================
# Meta: the sync tests themselves cannot silently go vacuous
# =============================================================================

def test_the_extracted_code_domains_are_not_empty():
    assert len(dim._PURPOSES) >= 8 and len(dim._ROUTES) == 3
    assert len(typing.get_args(__import__("src.safety.guardrails", fromlist=["GuardrailReason"]).GuardrailReason)) == 4
