# src/observability/dimensions.py
"""
Metric-dimension policy (M7.2) — the executable form of
docs/observability-contract.md §9.

Events answer "what happened during this request?" and may carry any
correlation field. Metrics answer "what pattern exists across requests?" and
may only be labelled by BOUNDED, non-sensitive values. This module decides,
per (event, key), which of those a field is:

  dimension    bounded, label-safe; value must come from a closed domain.
               Anything outside the domain is bucketed to "other" — it is
               never passed through, dropped, or allowed to grow a label.
  measure      numeric; aggregated (count/sum/histogram), never a label.
  correlation  event-level only (request_id, query_hash, timestamp).
  identity     names the series (event, step, schema_version); not a label.
  excluded     must not reach metrics at all (e.g. source_filter).

Classification is event + key scoped because the same key name carries
different domains in different events (guardrails.reason vs
retrieval.reason vs llm.fallback.reason).

This module is policy only. It defines no metric names, performs no
aggregation or export, and is not called from emit_event().
"""

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Tuple

DIMENSION = "dimension"
MEASURE = "measure"
CORRELATION = "correlation"
IDENTITY = "identity"
EXCLUDED = "excluded"

OTHER = "other"
NONE_VALUE = "none"

# Theoretical upper bound on label combinations per event if EVERY dimension
# of that event were used together (counting the error-class domain once —
# see series_upper_bound). A metric need not use all of them (M7.3
# chooses subsets); this only guards against an accidental explosion.
CARDINALITY_CAP = 2000

# Keys that must never appear in telemetry or become labels, in any event.
FORBIDDEN_KEYS: FrozenSet[str] = frozenset({
    "query", "raw_query", "sql", "prompt", "completion", "error", "error_text",
    "error_message", "message", "document_id", "document", "content",
    "source_name", "client_id", "client_ip", "ip",
})

# Exception class names allowed as an error label; anything else -> "other".
ERROR_TYPES: FrozenSet[str] = frozenset({
    "ConnectionError", "TimeoutError", "ValueError", "KeyError", "TypeError",
    "RuntimeError", "OSError", "AttributeError", "IndexError",
    "LLMExhaustedRetriesError", "APIConnectionError", "APITimeoutError",
    "RateLimitError", "OperationalError",
})


def allowed_models() -> FrozenSet[str]:
    """Models that may be a label value: priced models plus configured ones."""
    from src.llm.llm_client import DEFAULT_MODEL, FALLBACK_MODEL, PRICES_PER_1M

    return frozenset(PRICES_PER_1M) | {DEFAULT_MODEL, FALLBACK_MODEL}


_ERROR_DOMAIN = "error_types"
_MODEL_DOMAIN = "models"

_BOOL = frozenset({"true", "false"})
_ROUTES = frozenset({"/query", "/rag", "/analytics"})
_ROUTES_OR_OTHER = _ROUTES | {"other"}
_REQUEST_INTENTS = frozenset({"rag", "analytics", "reject"})
_ANALYTICS_INTENTS = frozenset({"timeseries", "merchant_rank", "category_rank", "generic", NONE_VALUE})
_INTENT_METHODS = frozenset({"heuristic", "llm"})
_PURPOSES = frozenset({
    "intent_classification", "language_detection", "translation", "query_rewrite",
    "rag_answer", "rag_insight", "analytics_nl_to_sql", "analytics_summary",
    "llm_rerank",
})
_GUARDRAIL_REASONS = frozenset({"too_short", "noise", "injection", "out_of_domain", NONE_VALUE})
_RETRIEVAL_METHODS = frozenset({"vector_rpc"})
_SKIP_REASONS = frozenset({"retriever_disabled", "no_embedding"})
_RERANKERS = frozenset({"hybrid"})
_FALLBACK_REASONS = frozenset({"budget_threshold"})
_COST_STATUS = frozenset({"not_applicable", "complete", "partial", "unknown"})


@dataclass(frozen=True)
class Rule:
    kind: str
    # frozenset of allowed normalized values, or _ERROR_DOMAIN / _MODEL_DOMAIN
    domain: Any = None


def _D(domain) -> Rule:
    return Rule(DIMENSION, domain if isinstance(domain, str) else frozenset(domain))


_M = Rule(MEASURE)
_C = Rule(CORRELATION)
_I = Rule(IDENTITY)
_X = Rule(EXCLUDED)

# Envelope fields (every event). `status` is a per-event dimension below.
ENVELOPE: Dict[str, Rule] = {
    "schema_version": _I,
    "event": _I,
    "step": _I,
    "timestamp": _C,
    "request_id": _C,
    "duration_ms": _M,
}

_SUCCESS = {"success"}
_FAILURE = {"failure"}

# (event, key) -> Rule. Envelope `status` and every metadata key.
POLICY: Dict[Tuple[str, str], Rule] = {}


def _event(name: str, status, **keys: Rule) -> None:
    POLICY[(name, "status")] = _D(status)
    for key, rule in keys.items():
        POLICY[(name, key)] = rule


_event("request.started", _SUCCESS, route=_D(_ROUTES))
# M8.4: emitted by the rate-limit middleware before any request context
# exists (request_id "-"); the route is already mapped to a bounded value.
_event("rate_limit.blocked", {"blocked"}, route=_D(_ROUTES_OR_OTHER))
_event("request.completed", {"success", "blocked", "error"},
       route=_D(_ROUTES), error_type=_D(_ERROR_DOMAIN),
       cost_status=_D(_COST_STATUS), cost_usd_total=_M)

_GUARDRAIL = dict(blocked=_D(_BOOL), reason=_D(_GUARDRAIL_REASONS),
                  query_length=_M, query_hash=_C)
_event("guardrails.completed", _SUCCESS, **_GUARDRAIL)
_event("guardrails.blocked", {"blocked"}, **_GUARDRAIL)

_event("language_detection.completed", _SUCCESS)
_event("language_detection.failed", _FAILURE)

_event("intent.completed", _SUCCESS, intent=_D(_REQUEST_INTENTS), confidence=_M,
       method=_D(_INTENT_METHODS), route=_D(_REQUEST_INTENTS))
_event("intent.failed", _FAILURE, error_type=_D(_ERROR_DOMAIN))

_event("retrieval.completed", _SUCCESS, retrieval_method=_D(_RETRIEVAL_METHODS),
       candidate_count=_M, source_filter=_X)
_event("retrieval.failed", _FAILURE, retrieval_method=_D(_RETRIEVAL_METHODS),
       error_type=_D(_ERROR_DOMAIN), source_filter=_X)
_event("retrieval.skipped", {"skipped"}, retrieval_method=_D(_RETRIEVAL_METHODS),
       reason=_D(_SKIP_REASONS), candidate_count=_M, selected_count=_M, source_filter=_X)

_event("ranking.completed", _SUCCESS, candidate_count=_M, selected_count=_M,
       reranker=_D(_RERANKERS))
_event("ranking.failed", _FAILURE, candidate_count=_M, error_type=_D(_ERROR_DOMAIN))
_event("ranking.skipped", {"skipped"}, candidate_count=_M, selected_count=_M)

_event("llm.completed", _SUCCESS, purpose=_D(_PURPOSES), model=_D(_MODEL_DOMAIN),
       prompt_tokens=_M, completion_tokens=_M, total_tokens=_M,
       estimated_cost_usd=_M, retry_count=_M)
_event("llm.failed", _FAILURE, purpose=_D(_PURPOSES), model=_D(_MODEL_DOMAIN),
       retry_count=_M, error_type=_D(_ERROR_DOMAIN))
_event("llm.fallback", _SUCCESS, purpose=_D(_PURPOSES), from_model=_D(_MODEL_DOMAIN),
       to_model=_D(_MODEL_DOMAIN), reason=_D(_FALLBACK_REASONS),
       cumulative_session_cost_usd=_M)

_event("analytics.sql.completed", _SUCCESS, intent=_D(_ANALYTICS_INTENTS),
       used_fallback_sql=_D(_BOOL), primary_error_type=_D(_ERROR_DOMAIN), row_count=_M)
_event("analytics.sql.failed", _FAILURE, intent=_D(_ANALYTICS_INTENTS),
       used_fallback_sql=_D(_BOOL), primary_error_type=_D(_ERROR_DOMAIN),
       error_type=_D(_ERROR_DOMAIN))
_event("analytics.completed", {"success", "failure"}, intent=_D(_ANALYTICS_INTENTS),
       confidence=_M, chart_generated=_D(_BOOL), error_type=_D(_ERROR_DOMAIN))


# =============================================================================
# NORMALIZATION & LOOKUP
# =============================================================================

def _normalize(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return NONE_VALUE
    return str(value)


def _domain_values(rule: Rule) -> FrozenSet[str]:
    if rule.domain == _ERROR_DOMAIN:
        return ERROR_TYPES | {NONE_VALUE}
    if rule.domain == _MODEL_DOMAIN:
        return allowed_models()
    return rule.domain


def classify(event: str, key: str) -> Rule:
    """
    The single classification for (event, key). Raises for forbidden or
    unclassified keys — an unknown key is a policy decision not yet made,
    never a default.
    """
    if key in FORBIDDEN_KEYS:
        raise ValueError(f"forbidden telemetry key: {key!r}")
    if key in ENVELOPE:
        return ENVELOPE[key]
    try:
        return POLICY[(event, key)]
    except KeyError:
        raise KeyError(f"unclassified metric field: {event}.{key}") from None


def dimension_value(event: str, key: str, value: Any) -> str:
    """
    Label-safe value for a dimension. Out-of-domain values become "other".
    Raises if (event, key) is not a dimension, so a measure, correlation id
    or excluded field can never be turned into a label by accident.
    """
    rule = classify(event, key)
    if rule.kind != DIMENSION:
        raise ValueError(f"{event}.{key} is a {rule.kind}, not a dimension")
    normalized = _normalize(value)
    return normalized if normalized in _domain_values(rule) else OTHER


def dimensions_for(event_dict: Dict[str, Any]) -> Dict[str, str]:
    """
    All label-safe dimensions of one emitted event: its envelope `status`
    plus every dimension-classified metadata key, each normalized.
    """
    name = event_dict["event"]
    dims = {"status": dimension_value(name, "status", event_dict.get("status"))}
    for key, value in (event_dict.get("metadata") or {}).items():
        if classify(name, key).kind == DIMENSION:
            dims[key] = dimension_value(name, key, value)
    return dims


def series_upper_bound(event: str) -> int:
    """Product of every dimension's domain size (+ the "other" bucket)."""
    from src.llm.llm_client import PRICES_PER_1M

    bound = 1
    error_dim_counted = False
    for (ev, _key), rule in POLICY.items():
        if ev != event or rule.kind != DIMENSION:
            continue
        if rule.domain == _ERROR_DOMAIN:
            # Policy: a metric uses at most ONE error-class dimension per
            # event (error_type and primary_error_type are never combined),
            # so the error domain is counted once.
            if not error_dim_counted:
                bound *= len(_domain_values(rule) | {OTHER})
                error_dim_counted = True
            continue
        if rule.domain == _MODEL_DOMAIN:
            # Worst case, independent of environment: every priced model plus
            # a distinct configured default and fallback, plus "other".
            bound *= len(PRICES_PER_1M) + 2 + 1
        else:
            bound *= len(_domain_values(rule) | {OTHER})
    return bound


def events() -> Tuple[str, ...]:
    return tuple(sorted({ev for ev, _ in POLICY}))


# =============================================================================
# HUMAN-READABLE RENDERING (docs/observability-contract.md §9)
# =============================================================================

def _describe(rule: Rule) -> str:
    if rule.kind != DIMENSION:
        return ""
    if rule.domain == _ERROR_DOMAIN:
        return "known exception classes, else `other`"
    if rule.domain == _MODEL_DOMAIN:
        return "priced or configured models, else `other`"
    return ", ".join(f"`{v}`" for v in sorted(rule.domain)) + "; else `other`"


def render_policy_markdown() -> str:
    """Deterministic markdown for the contract's §9 block (sync-tested)."""
    lines = ["| Event | Key | Class | Allowed values |", "|---|---|---|---|"]
    for (ev, key), rule in sorted(POLICY.items()):
        lines.append(f"| `{ev}` | `{key}` | {rule.kind} | {_describe(rule)} |")
    lines += ["", "| Event | Series upper bound |", "|---|---|"]
    for ev in events():
        lines.append(f"| `{ev}` | {series_upper_bound(ev)} |")
    lines.append("")
    lines.append(f"Cardinality cap per event: {CARDINALITY_CAP}.")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - regenerate the §9 block
    print(render_policy_markdown())
