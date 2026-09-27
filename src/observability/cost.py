# src/observability/cost.py
"""
Request-scoped LLM cost accumulation (P2 — M6 prerequisite).

Narrow purpose, per the M6 design review: make *request-level* cost
aggregation trustworthy under concurrency. This does NOT replace or alter
src/llm/llm_client.py's existing process-global SESSION_COST_USD /
MAX_COST_USD soft-budget mechanism — that's a deliberate whole-process
spend guard (policy), left untouched. Converting it to be request-scoped
instead would silently weaken it: a $0.10 default budget essentially never
trips within a single request, only across many requests over time, which
is the actual protection it's meant to provide.

This module is purely additive: a second, request-scoped accumulator,
isolated the same way request_id is (M3) — each request runs in its own
execution context, so this naturally starts at 0.0 for every new request
without needing an explicit reset, though reset_request_cost() is still
called explicitly at the API boundary for clarity, the same way
set_request_id() is.

Per-call LLM cost (from estimate_cost(), off the real API response) is
already accurate regardless of any of this — this module only fixes
*summing it up per request*, nothing about the per-call number itself.
"""

from contextvars import ContextVar

_request_cost_usd: ContextVar[float] = ContextVar("request_cost_usd", default=0.0)


def reset_request_cost() -> None:
    """Call once, at the API boundary, alongside set_request_id()."""
    _request_cost_usd.set(0.0)


def add_request_cost(amount_usd: float) -> None:
    """Called by LLMClient.run() after every successful call, in addition
    to (not instead of) the existing process-global SESSION_COST_USD."""
    _request_cost_usd.set(_request_cost_usd.get() + amount_usd)


def get_request_cost() -> float:
    return _request_cost_usd.get()
