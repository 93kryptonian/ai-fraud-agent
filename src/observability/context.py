# src/observability/context.py
"""
Request context (M3).

Scope, deliberately narrow (see docs/observability-contract.md M3):
- Give every incoming request a unique request_id
- Make it readable from anywhere in the call stack without threading it
  through every function signature
- Guarantee concurrent requests never see each other's ID

This module does NOT emit events, measure timing, or touch the LLM/RAG
pipelines. That's M4 onward.

Why ContextVar and not a module-level global:
A plain global (`CURRENT_REQUEST_ID = None`) is shared process-wide, so
under concurrency (FastAPI serving requests A and B at the same time)
request B could overwrite the ID request A is still using, or A could read
B's ID. `contextvars.ContextVar` gives each execution context (each asyncio
Task, each thread) its own independent value — that's what actually
prevents cross-request leakage.
"""

import uuid
from contextvars import ContextVar
from typing import Optional

_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    """Generate a fresh request identifier."""
    return str(uuid.uuid4())


def set_request_id(request_id: str) -> None:
    """
    Bind a request_id to the current execution context.

    Call this once, at the API boundary, before any other work happens for
    that request (including guardrail validation) — see contract §2.
    """
    _request_id.set(request_id)


def get_request_id() -> Optional[str]:
    """
    Return the request_id bound to the current execution context, or None
    if called outside any request (e.g. at import time, in a script).
    """
    return _request_id.get()
