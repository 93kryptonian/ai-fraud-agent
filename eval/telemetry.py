"""
Per-case latency, cost and routing-layer facts, read from the system's own
telemetry events (the structured JSON events the API already emits).

Nothing is estimated here. A figure the events do not carry stays None and is
reported as missing, never as zero. In particular, the events carry no
embedding cost: only LLM calls are priced.
"""

import json
import logging


class EventCollector(logging.Handler):
    """Collects the parsed telemetry events emitted while it is attached."""

    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, record):
        try:
            self.events.append(json.loads(record.getMessage()))
        except ValueError:
            pass

    def reset(self):
        self.events = []

    def __enter__(self):
        logging.getLogger("observability.events").addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger("observability.events").removeHandler(self)


def summarize(events: list) -> dict:
    """One request's events -> the measurements the live report needs."""
    by_name = {}
    for e in events:
        by_name.setdefault(e.get("event"), []).append(e)

    completed = (by_name.get("request.completed") or [{}])[-1]
    meta = completed.get("metadata") or {}
    intent = ((by_name.get("intent.completed") or [{}])[-1]).get("metadata") or {}
    llm_events = (by_name.get("llm.completed") or []) + (by_name.get("llm.failed") or [])

    return {
        "request_status": completed.get("status"),
        "duration_ms": completed.get("duration_ms"),
        "cost_status": meta.get("cost_status"),
        "cost_usd": meta.get("cost_usd_total"),
        "intent_method": intent.get("method"),
        "intent_confidence": intent.get("confidence"),
        "llm_calls": len(llm_events),
        "llm_purposes": sorted({(e.get("metadata") or {}).get("purpose") for e in llm_events} - {None}),
        "llm_failures": len(by_name.get("llm.failed") or []),
        "guardrail_blocked": bool(by_name.get("guardrails.blocked")),
    }
