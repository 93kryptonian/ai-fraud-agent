# src/observability/cost.py
"""
Request-scoped LLM cost accumulation (P2, extended in M7.1).

Narrow purpose: make *request-level* cost attribution trustworthy under
concurrency and honest about what is unknown. This does NOT replace or
alter src/llm/llm_client.py's process-global SESSION_COST_USD /
MAX_COST_USD soft-budget mechanism — that is a deliberate whole-process
spend guard (policy), and it only ever sees *known* cost. Converting it to
request-scoped would silently weaken it: a $0.10 default budget essentially
never trips within a single request, only across many requests over time.

M7.1 semantics (docs/observability-contract.md §8). Cost is derived from
*responses received*, never from attempts:

- priced response    : usage present and the model has a known price
- unpriced response  : usage missing, or the model has no known price
- a call that raises before any response contributes nothing and does not
  change completeness (reliability is reported by llm.failed instead)

cost_status:
  not_applicable  no response with cost relevance was received -> 0.0
  complete        every response received was priced           -> known total
  partial         >=1 priced and >=1 unpriced response         -> lower bound
  unknown         only unpriced responses                      -> None

Unknown is never represented as 0.0.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RequestCost:
    known_total_usd: float = 0.0
    priced_responses: int = 0
    unpriced_responses: int = 0


# Default None (not a shared mutable instance): each request gets its own
# RequestCost, created lazily or by reset_request_cost().
_request_cost: ContextVar[Optional[RequestCost]] = ContextVar("request_cost", default=None)


def _current() -> RequestCost:
    cost = _request_cost.get()
    if cost is None:
        cost = RequestCost()
        _request_cost.set(cost)
    return cost


def reset_request_cost() -> None:
    """Call once, at the API boundary, alongside set_request_id()."""
    _request_cost.set(RequestCost())


def add_request_cost(amount_usd: float) -> None:
    """Record one PRICED response (known cost)."""
    cost = _current()
    cost.known_total_usd += amount_usd
    cost.priced_responses += 1


def add_unpriced_response() -> None:
    """Record one response whose cost cannot be determined."""
    _current().unpriced_responses += 1


def get_request_cost() -> float:
    """Known (priced) cost so far. Excludes unpriced responses."""
    return _current().known_total_usd


def cost_metadata() -> Dict[str, Any]:
    """
    request.completed cost fields: {"cost_status", "cost_usd_total"}.
    """
    c = _current()
    if c.priced_responses == 0 and c.unpriced_responses == 0:
        return {"cost_status": "not_applicable", "cost_usd_total": 0.0}
    if c.unpriced_responses == 0:
        return {"cost_status": "complete", "cost_usd_total": round(c.known_total_usd, 6)}
    if c.priced_responses > 0:
        return {"cost_status": "partial", "cost_usd_total": round(c.known_total_usd, 6)}
    return {"cost_status": "unknown", "cost_usd_total": None}
