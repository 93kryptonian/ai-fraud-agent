# src/observability/live.py
"""
Live feed adapter (M8.1).

    emit_event()
        |-- JSONL StreamHandler            (unchanged)
        '-- optional SignalsHandler  --->  process Aggregator  --->  snapshot()
                                           (lock around mutation)

Consumes ONLY events already emitted through the `observability.events`
logger; nothing in events.py knows about this module. The pure Aggregator
(src/observability/signals.py) stays unaware of logging, threads, HTTP and
environment variables — the lock and the process-level ownership live here.

Properties:
- Opt-in and off by default (SIGNALS_ENABLED). When disabled there is no
  handler and no aggregator: zero collection overhead.
- Fail-open: a telemetry failure never becomes an application failure. The
  handler swallows every exception (and the logging module's own error
  path), counting it in `handler_errors` for diagnosis.
- State is process-local and non-durable: a restart or spin-down starts a
  fresh aggregator (`started_at_unix` records when). Counters are cumulative
  since then. M8.3 consumes `started_at_unix` for reset detection.

No endpoint, no export, no windows here — those are M8.2 / M8.3.
"""

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from src.observability.signals import Aggregator

EVENTS_LOGGER_NAME = "observability.events"


class SignalsHandler(logging.Handler):
    """Feeds each emitted event (a JSON line) into the owner's aggregator."""

    def __init__(self, owner: "LiveSignals"):
        super().__init__(level=logging.INFO)
        self._owner = owner

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._owner._ingest(record.getMessage())
        except Exception:  # fail-open: never raise into the request path
            self._owner._count_error()

    def handleError(self, record: logging.LogRecord) -> None:
        # Suppress logging's default stderr traceback; failures are counted.
        pass


class LiveSignals:
    """Owns the process-level Aggregator, its lock and its handler."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._aggregator = Aggregator()
        self._handler = SignalsHandler(self)
        self.started_at_unix = time.time()
        self.handler_errors = 0
        self.ignored_records = 0

    # ---- ingestion (called by the handler) --------------------------------

    def _ingest(self, message: str) -> None:
        try:
            event = json.loads(message)
        except ValueError:
            with self._lock:
                self.ignored_records += 1  # not a JSON event line
            return
        with self._lock:
            self._aggregator.observe(event)

    def _count_error(self) -> None:
        try:
            with self._lock:
                self.handler_errors += 1
        except Exception:
            pass

    # ---- reading -----------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = self._aggregator.snapshot()
            snap["meta"]["started_at_unix"] = self.started_at_unix
            snap["meta"]["handler_errors"] = self.handler_errors
            snap["meta"]["ignored_records"] = self.ignored_records
        return snap


# =============================================================================
# PROCESS-LEVEL OWNERSHIP
# =============================================================================

_live: Optional[LiveSignals] = None
_state_lock = threading.Lock()


def is_enabled() -> bool:
    return _live is not None


def get() -> Optional[LiveSignals]:
    """The live collector, or None when disabled (nothing was instantiated)."""
    return _live


def enable() -> LiveSignals:
    """Attach the handler and start a fresh aggregator. Idempotent."""
    global _live
    with _state_lock:
        if _live is None:
            live = LiveSignals()
            logging.getLogger(EVENTS_LOGGER_NAME).addHandler(live._handler)
            _live = live
        return _live


def disable() -> None:
    """Detach the handler and drop the aggregator (state is discarded)."""
    global _live
    with _state_lock:
        if _live is not None:
            logging.getLogger(EVENTS_LOGGER_NAME).removeHandler(_live._handler)
            _live = None


def configure_from_env() -> Optional[LiveSignals]:
    """Enable iff SIGNALS_ENABLED=true. Read at call time, not import time."""
    if os.getenv("SIGNALS_ENABLED", "false").lower() == "true":
        return enable()
    return None
