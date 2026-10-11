"""
Evaluation contract: turn one API response into an observation, and compare it
with what the case expects.

Pure functions, no network and no project imports, so the checks themselves can
be tested without credentials. The observation is read from the response shape
POST /query really returns:

  analytics  {"intent": "analytics", "result": {"type": "analytics", "analytics": {"answer", "sql", ...}}}
  rag        {"intent": "rag",       "result": {"type": "rag", "answer", "citations", "retrieval_status", ...}}
  rejection  {"intent": "reject",    "result": {"type": "reject", "message"}}

Nothing here decides a route. It only reads the route the API reported.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional

ANSWER = "answer"
REFUSAL = "refusal"
DEGRADED = "degraded"
ERROR = "error"
RATE_LIMITED = "rate_limited"

_ROUTES = {"analytics", "rag", "reject"}

_ID_WORDS = frozenset(
    "yang dan adalah untuk dengan dalam ini itu pada dari atau tidak dapat oleh sebagai karena akan juga "
    "terjadi penipuan kartu transaksi merupakan melalui terhadap memiliki kasus insiden tertinggi terendah "
    "jumlah sebuah lebih antara setelah ketika bahwa jika saat yaitu serta hanya sekitar telah sudah bisa "
    "mereka kami anda pedagang bank".split()
)
_EN_WORDS = frozenset(
    "the and of to is are in for that with this by on as from or not can be it an which has have was were "
    "their than more most highest lowest total rate cases".split()
)


@dataclass(frozen=True)
class Observed:
    route: Optional[str]
    behavior: str
    answer: str = ""
    has_sql: bool = False
    citations: int = 0
    note: str = ""


@dataclass
class CaseResult:
    id: str
    observed: Observed
    route_ok: bool
    behavior_ok: bool
    include_ok: bool
    language_ok: bool
    grounding_ok: bool
    failures: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all((self.route_ok, self.behavior_ok, self.include_ok, self.language_ok, self.grounding_ok))


def answer_language(text: str) -> str:
    """'en', 'id' or 'unknown' from function-word counts. Judges OUTPUT text, never routes a query."""
    words = re.findall(r"[a-z']+", (text or "").lower())
    id_hits = sum(w in _ID_WORDS for w in words)
    en_hits = sum(w in _EN_WORDS for w in words)
    if max(id_hits, en_hits) < 2 or id_hits == en_hits:
        return "unknown"
    return "id" if id_hits > en_hits else "en"


def observe(body: Any, status_code: int = 200) -> Observed:
    """Read route and behavior from a /query response. Anything unrecognized is an error, never a pass."""
    if status_code == 429:
        return Observed(None, RATE_LIMITED, note="HTTP 429")
    if status_code >= 400:
        return Observed(None, ERROR, note=f"HTTP {status_code}")
    if not isinstance(body, dict):
        return Observed(None, ERROR, note="response is not an object")

    route = body.get("intent")
    if body.get("error"):
        return Observed(route if route in _ROUTES else None, ERROR, note="API reported an error")

    result = body.get("result")
    if route not in _ROUTES or not isinstance(result, dict):
        return Observed(None, ERROR, note="no recognizable route or result")

    kind = result.get("type")
    if kind != route:
        return Observed(route, ERROR, note=f"route {route!r} but result type {kind!r}")

    if route == "reject":
        return Observed(route, REFUSAL, answer=str(result.get("message") or ""))

    if route == "analytics":
        analytics = result.get("analytics")
        analytics = analytics if isinstance(analytics, dict) else {}
        answer = str(analytics.get("answer") or "")
        return Observed(
            route,
            ANSWER if answer.strip() else ERROR,
            answer=answer,
            has_sql=bool(str(analytics.get("sql") or "").strip()),
            note="" if answer.strip() else "analytics returned no answer",
        )

    answer = str(result.get("answer") or "")
    citations = result.get("citations")
    status = result.get("retrieval_status")
    if status not in (None, "ok"):
        return Observed(route, DEGRADED, answer=answer, note=f"retrieval_status={status}")
    return Observed(
        route,
        ANSWER if answer.strip() else ERROR,
        answer=answer,
        citations=len(citations) if isinstance(citations, list) else 0,
        note="" if answer.strip() else "rag returned no answer",
    )


def check_case(case: dict, observed: Observed) -> CaseResult:
    """Compare one observation with the case's expectations. Every unmet expectation is a failure."""
    failures = []

    route_ok = observed.route == case["expected_route"]
    if not route_ok:
        failures.append(f"route: expected {case['expected_route']!r}, got {observed.route!r}")

    behavior_ok = observed.behavior == case["expected_behavior"]
    if not behavior_ok:
        why = f" ({observed.note})" if observed.note else ""
        failures.append(f"behavior: expected {case['expected_behavior']!r}, got {observed.behavior!r}{why}")

    include_ok = True
    for requirement in case.get("must_include") or []:
        # a string is one required term; a list is alternatives, any one of which satisfies it
        # (an answer in another language may translate the term)
        options = [requirement] if isinstance(requirement, str) else list(requirement)
        if observed.behavior != ANSWER or not any(o.lower() in observed.answer.lower() for o in options):
            include_ok = False
            failures.append(f"must_include: none of {options!r} in the answer")

    language_ok = True
    if case.get("language") and case["expected_behavior"] == ANSWER:
        got = answer_language(observed.answer) if observed.behavior == ANSWER else "none"
        language_ok = got == case["language"]
        if not language_ok:
            failures.append(f"language: expected {case['language']!r}, answer reads as {got!r}")

    grounding_ok = True
    if case.get("expect_sql") and not observed.has_sql:
        grounding_ok = False
        failures.append("grounding: analytics answer carries no SQL")
    if case.get("expect_citations") and observed.citations == 0:
        grounding_ok = False
        failures.append("grounding: rag answer carries no citations")

    return CaseResult(case["id"], observed, route_ok, behavior_ok, include_ok, language_ok, grounding_ok, failures)
