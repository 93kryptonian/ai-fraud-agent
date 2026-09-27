# src/observability/timing.py
"""
Timing (M5).

Two layers, deliberately kept separate:

- `elapsed_timer()`: a plain monotonic-clock measurement. Knows nothing
  about events. Used directly by call sites whose terminal event isn't a
  simple binary success/failure — today that's `guardrails`
  (completed/blocked) and `request` (success/blocked/error) in
  api/routers.py. Those already decide their own event name and status
  explicitly; they just needed a number to attach to it.

- `observe_step()`: the M5 design-review's "Option A" — the smallest
  possible execution observer. It emits exactly one terminal event based
  ONLY on whether the wrapped block raised, nothing else:
      no exception -> "{step}.completed", status="success"
      exception    -> "{step}.failed",    status="failure", re-raised
  It has no parameter for status, metadata, or outcome — on purpose. A
  stage with more than two outcomes (blocked, skipped, business-level
  degraded, ...) does not fit this shape and must not be forced through
  it; it emits its own event explicitly via elapsed_timer() instead, the
  same way guardrails/request already do. Used for M5's two new
  orchestrator stages: language_detection, intent.

perf_counter(), not datetime, backs every measurement here — monotonic,
immune to wall-clock adjustments. The event's own `timestamp` field
(src/observability/events.py) stays wall-clock for human/log correlation;
the two serve different purposes and are computed independently.

Timing infrastructure must never alter application behavior: observe_step
always re-raises whatever it catches, unchanged, after emitting.
"""

import time
from contextlib import contextmanager
from typing import Callable, Iterator

from src.observability.events import emit_event


@contextmanager
def elapsed_timer() -> Iterator[Callable[[], int]]:
    """
    Yield a zero-arg callable returning elapsed milliseconds (int) since
    entering this context, valid to call at any point inside the `with`
    block (or after it exits — the clock keeps running, callers just stop
    asking).
    """
    start = time.perf_counter()
    # round(), not int(): float subtraction on two perf_counter() reads can
    # land a hair under the "true" value (e.g. 4.9999999999954525 instead
    # of 5.0) due to ordinary floating-point representation error —
    # truncating with int() would silently under-report by 1ms in exactly
    # those cases. Confirmed empirically, not a theoretical worry.
    yield lambda: round((time.perf_counter() - start) * 1000)


@contextmanager
def observe_step(step: str):
    """
    The minimal binary execution observer (M5 design, Option A).

    with observe_step("intent"):
        result = detect_intent(...)

    Emits "intent.completed" (status=success) if the block completes
    normally, or "intent.failed" (status=failure) if it raises — and
    re-raises that same exception afterward. Does not know about
    business outcomes, blocked/skipped states, or take any status
    parameter — see module docstring for why.
    """
    with elapsed_timer() as elapsed:
        try:
            yield
        except Exception:
            emit_event(f"{step}.failed", step=step, status="failure", duration_ms=elapsed())
            raise
        else:
            emit_event(f"{step}.completed", step=step, status="success", duration_ms=elapsed())
