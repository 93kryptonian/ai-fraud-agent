# src/observability/events.py
"""
Structured events (M4).

This is a machine-readable event contract, not a logging convenience. It is
deliberately a separate stream from src/utils/logger.py (M3): that logger is
for humans reading a terminal; this module is for something that will later
parse, aggregate, or trace these records. Conflating the two would mean
either breaking the human-readable format or growing every human log line
a metadata blob nobody reads.

Scope, per docs/observability-contract.md and the M4 design review:
- request.started / request.completed (request lifecycle)
- guardrails.completed / guardrails.blocked (the decision boundary)
Nothing else yet. No timing (M5 owns duration_ms — always None here, never
faked as 0), no LLM/RAG telemetry (M6), no OpenTelemetry (explicit non-goal).

request.completed represents *normal* endpoint completion only. If a
request raises an unhandled exception before reaching that point, no
request.completed is emitted for it today — that's an M5/M6 concern
(request/stage failure semantics), not silently patched over here.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.observability.context import get_request_id

SCHEMA_VERSION = 1

# Dedicated logger/stream: raw JSON lines, no human-readable prefix, and
# explicitly NOT the same logger get_logger() configures for free-text logs.
_events_logger = logging.getLogger("observability.events")
if not _events_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _events_logger.addHandler(_handler)
    _events_logger.setLevel(logging.INFO)
    _events_logger.propagate = False


@dataclass
class Event:
    """
    The event envelope, per contract §3.

    `status` here is the STAGE-level enum (success | failure | blocked |
    skipped) — not the request-level outcome enum from contract §2
    (success | blocked | rate_limited | error). The two are related but
    distinct: request.completed's `status` field happens to be drawn from
    the request-level enum, while e.g. guardrails.completed/blocked draw
    from the stage-level one. This dataclass doesn't enforce either enum
    by type (both are plain str) precisely so it can represent both without
    conflating them into one Literal.
    """

    schema_version: int
    timestamp: str
    request_id: str
    event: str
    step: str
    status: str
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def query_hash(text: str) -> str:
    """
    Sha256, truncated — enough to correlate identical queries across events
    without being able to reverse it back to the query. Never put the raw
    query text itself into an event. See contract §6.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def emit_event(
    event: str,
    step: str,
    status: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Event:
    """
    Build and emit one structured event, tagged with the request_id bound
    to the current execution context (M3). Returns the Event for callers
    that want to inspect what was just emitted (e.g. tests) — emitting
    always happens as a side effect; this is not a query.
    """
    evt = Event(
        schema_version=SCHEMA_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id=get_request_id() or "-",
        event=event,
        step=step,
        status=status,
        duration_ms=None,
        metadata=metadata or {},
    )
    _events_logger.info(json.dumps(asdict(evt), default=str))
    return evt
