"""
Thin client for the fraud-intelligence API.

This module is the only place the live UI talks to the network, and it knows
nothing about the API's internals: it sends a question to POST /query and
turns whatever comes back into one `Outcome`. It imports nothing from the
API's own packages and holds no provider credentials.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

# A cold free-tier API can take about 45 s to wake, then the request itself runs.
DEFAULT_TIMEOUT_SECONDS = 90.0

ANALYTICS = "analytics"
RAG = "rag"
REJECTED = "rejected"
RATE_LIMITED = "rate_limited"
TIMEOUT = "timeout"
UNREACHABLE = "unreachable"
API_ERROR = "api_error"
UNEXPECTED = "unexpected"


@dataclass(frozen=True)
class Outcome:
    kind: str
    elapsed_seconds: float = 0.0
    answer: str = ""
    sql: Optional[str] = None
    rows: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    confidence: Optional[float] = None
    message: str = ""
    retry_after_seconds: Optional[int] = None
    status_code: Optional[int] = None


def _number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _parse_success(body: Any, elapsed: float) -> Outcome:
    if not isinstance(body, dict):
        return Outcome(UNEXPECTED, elapsed, message="The API returned a response this page does not understand.")

    if body.get("error"):
        return Outcome(API_ERROR, elapsed, message="The API reported an error while answering.")

    result = body.get("result")
    if not isinstance(result, dict):
        return Outcome(UNEXPECTED, elapsed, message="The API returned a response this page does not understand.")

    kind = result.get("type")

    if kind == "reject":
        return Outcome(REJECTED, elapsed, message=str(result.get("message") or "The question was rejected."))

    if kind == "analytics":
        analytics = result.get("analytics")
        analytics = analytics if isinstance(analytics, dict) else {}
        rows = analytics.get("data_points")
        return Outcome(
            ANALYTICS,
            elapsed,
            answer=str(analytics.get("answer") or ""),
            sql=analytics.get("sql") or None,
            rows=rows if isinstance(rows, list) else [],
            confidence=_number(analytics.get("confidence")),
        )

    if kind == "rag":
        citations = result.get("citations")
        status = result.get("retrieval_status")
        return Outcome(
            RAG,
            elapsed,
            answer=str(result.get("answer") or result.get("insight") or ""),
            citations=citations if isinstance(citations, list) else [],
            confidence=_number((result.get("score") or {}).get("final_score")),
            message="" if status in (None, "ok") else "Document search was not fully available for this answer.",
        )

    return Outcome(UNEXPECTED, elapsed, message="The API returned an answer type this page does not know.")


def ask(
    api_url: str,
    question: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    transport: Optional[httpx.BaseTransport] = None,
) -> Outcome:
    """Send one question to the API and classify the result. Never raises."""
    url = api_url.rstrip("/") + "/query"
    started = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - started

    try:
        with httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=10.0), transport=transport
        ) as client:
            response = client.post(url, json={"query": question})
    except httpx.TimeoutException:
        return Outcome(TIMEOUT, elapsed(), message=f"No answer within {int(timeout_seconds)} seconds.")
    except httpx.TransportError:
        return Outcome(UNREACHABLE, elapsed(), message="Could not reach the API.")

    status = response.status_code

    if status == 429:
        retry = None
        try:
            retry = int(response.headers.get("Retry-After", ""))
        except ValueError:
            try:
                retry = int(response.json().get("retry_after_seconds"))
            except Exception:
                retry = None
        return Outcome(RATE_LIMITED, elapsed(), retry_after_seconds=retry, status_code=status,
                       message="The API is rate limiting requests.")

    if status >= 500:
        return Outcome(API_ERROR, elapsed(), status_code=status, message="The API failed while answering.")

    if status >= 400:
        return Outcome(REJECTED, elapsed(), status_code=status, message="The API did not accept this question.")

    try:
        body = response.json()
    except ValueError:
        return Outcome(UNEXPECTED, elapsed(), status_code=status,
                       message="The API returned a response this page does not understand.")

    outcome = _parse_success(body, elapsed())
    return Outcome(**{**outcome.__dict__, "status_code": status})
