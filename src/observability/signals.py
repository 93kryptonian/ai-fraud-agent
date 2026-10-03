# src/observability/signals.py
"""
Operational signals derived from structured events (M7.3).

    event dicts -> Aggregator.observe() -> in-memory state -> snapshot()

A pure, replay-only aggregator. It is NOT attached to emit_event(), holds no
global state, starts no collector, and does no export or time-windowing —
live wiring, vendor integration and rates-over-time are M8. The CLI at the
bottom is a replay/debug tool: JSONL file -> replay -> snapshot on stdout.

Invariants (docs/observability-contract.md §9 / §13):
- Every metric label is a dimension-classified field of its source event
  (src/observability/dimensions.py) or the envelope `status`; label values go
  through dimension_value(), so out-of-domain values become "other". There
  are no derived label mappings.
- An optional dimension that an event does not carry is OMITTED from the
  series, never filled with a synthetic "none". (A key that is present with a
  null value is a real emitted value and normalizes to "none".)
- request_id is a grouping key for per-request signals only, never a label.
- Absence is not zero: only observed series appear; ratios are None when
  their denominator is 0.
- Unclassified input never crashes the aggregator: it is counted in
  signals_unclassified_total and skipped (strict=True raises, for tests).
"""

import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.observability import dimensions as dim

# Latency buckets in ms; an implicit +Inf bucket keeps >10s observations.
DURATION_BUCKETS_MS: Tuple[float, ...] = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)
# Calls-per-request buckets (per purpose).
CALL_COUNT_BUCKETS: Tuple[float, ...] = (1, 2, 3, 4, 5, 10, 20)

MAX_IN_FLIGHT_REQUESTS = 10_000

COUNTER = "counter"
SUM = "sum"
HISTOGRAM = "histogram"

_SERIES_NOTE = (
    "Cumulative since the aggregator started; no time windows. Instrumented "
    "application request volume is requests_total + rate_limited_total; it is "
    "not every possible HTTP request. Percentiles are bucket upper-bound "
    "estimates."
)


# =============================================================================
# METRIC INVENTORY
# =============================================================================

@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: str
    events: Tuple[str, ...]
    labels: Tuple[str, ...] = ()
    value: Optional[str] = None  # measure key (or "duration_ms") for sum/histogram
    when: Optional[Callable[[Dict[str, Any]], bool]] = None


def _md(event: Dict[str, Any]) -> Dict[str, Any]:
    return event.get("metadata") or {}


METRICS: Tuple[MetricSpec, ...] = (
    # --- request ---------------------------------------------------------
    MetricSpec("requests_total", COUNTER, ("request.completed",), ("route", "status")),
    MetricSpec("request_duration_ms", HISTOGRAM, ("request.completed",), ("route", "status"),
               value="duration_ms"),
    MetricSpec("request_cost_usd_total", SUM, ("request.completed",), ("route", "cost_status"),
               value="cost_usd_total"),
    # Rate-limited requests never reach the request lifecycle, so they have no
    # request.completed: this counter is what closes the volume gap.
    MetricSpec("rate_limited_total", COUNTER, ("rate_limit.blocked",), ("route",)),
    # --- llm -------------------------------------------------------------
    MetricSpec("llm_calls_total", COUNTER, ("llm.completed", "llm.failed"),
               ("purpose", "model", "status")),
    MetricSpec("llm_duration_ms", HISTOGRAM, ("llm.completed", "llm.failed"), ("purpose",),
               value="duration_ms"),
    MetricSpec("llm_retries_total", SUM, ("llm.completed", "llm.failed"), ("purpose",),
               value="retry_count"),
    MetricSpec("llm_retried_calls_total", COUNTER, ("llm.completed",), ("purpose",),
               when=lambda e: (_md(e).get("retry_count") or 0) > 0),
    MetricSpec("llm_failures_total", COUNTER, ("llm.failed",), ("purpose", "error_type")),
    # llm.fallback is a terminal event for the downgrade itself, not an LLM
    # call outcome: it is counted from its own event, never from llm_calls.
    MetricSpec("llm_fallbacks_total", COUNTER, ("llm.fallback",), ("purpose", "reason")),
    MetricSpec("llm_prompt_tokens_total", SUM, ("llm.completed",), ("purpose", "model"),
               value="prompt_tokens"),
    MetricSpec("llm_completion_tokens_total", SUM, ("llm.completed",), ("purpose", "model"),
               value="completion_tokens"),
    MetricSpec("llm_cost_usd_total", SUM, ("llm.completed",), ("purpose", "model"),
               value="estimated_cost_usd"),
    MetricSpec("llm_cost_unknown_total", COUNTER, ("llm.completed",), ("purpose", "model"),
               when=lambda e: "estimated_cost_usd" in _md(e) and _md(e)["estimated_cost_usd"] is None),
    # --- retrieval / ranking ---------------------------------------------
    MetricSpec("retrieval_total", COUNTER,
               ("retrieval.completed", "retrieval.failed", "retrieval.skipped"),
               ("status", "retrieval_method", "reason")),
    MetricSpec("retrieval_empty_total", COUNTER, ("retrieval.completed",), ("retrieval_method",),
               when=lambda e: _md(e).get("candidate_count") == 0),
    MetricSpec("retrieval_failures_total", COUNTER, ("retrieval.failed",), ("error_type",)),
    MetricSpec("ranking_total", COUNTER,
               ("ranking.completed", "ranking.failed", "ranking.skipped"), ("status", "reranker")),
    # --- analytics -------------------------------------------------------
    MetricSpec("analytics_total", COUNTER, ("analytics.completed",), ("status", "intent")),
    MetricSpec("analytics_sql_total", COUNTER,
               ("analytics.sql.completed", "analytics.sql.failed"),
               ("status", "intent", "used_fallback_sql")),
    MetricSpec("analytics_failures_total", COUNTER, ("analytics.completed",), ("error_type",),
               when=lambda e: e.get("status") == "failure"),
    # --- intent / guardrails / language ----------------------------------
    MetricSpec("intent_total", COUNTER, ("intent.completed", "intent.failed"),
               ("status", "intent", "method")),
    MetricSpec("guardrails_total", COUNTER, ("guardrails.completed", "guardrails.blocked"),
               ("status", "reason")),
    MetricSpec("language_detection_total", COUNTER,
               ("language_detection.completed", "language_detection.failed"), ("status",)),
)

# Per-request derived signal: LLM calls per purpose within one request.
PER_REQUEST_METRIC = "llm_calls_per_request"
_LLM_CALL_EVENTS = ("llm.completed", "llm.failed")


def validate_specs() -> None:
    """
    Label traceability: every label must be the envelope `status` or a
    dimension-classified field, and wherever a source event carries that
    field it must be classified as a dimension (an event that does not carry
    an optional label simply omits it). Every value must be a measure.
    Raises on any violation.
    """
    for spec in METRICS:
        for event in spec.events:
            if event not in dim.events():
                raise ValueError(f"{spec.name}: unknown event {event}")
        for label in spec.labels:
            carried = 0
            for event in spec.events:
                try:
                    rule = dim.classify(event, label)
                except KeyError:
                    continue  # optional label absent from this event
                carried += 1
                if rule.kind != dim.DIMENSION:
                    raise ValueError(f"{spec.name}: label {label!r} is a {rule.kind} of {event}")
            if carried == 0:
                raise ValueError(f"{spec.name}: label {label!r} is carried by none of its events")
        if spec.value is not None:
            for event in spec.events:
                if dim.classify(event, spec.value).kind != dim.MEASURE:
                    raise ValueError(f"{spec.name}: value {spec.value!r} is not a measure of {event}")


# =============================================================================
# SERIES STATE
# =============================================================================

class _Histogram:
    def __init__(self, buckets: Sequence[float]):
        self.buckets = tuple(buckets)
        self.counts = [0] * (len(self.buckets) + 1)  # last = +Inf
        self.count = 0
        self.sum = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.sum += value
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def percentile(self, q: float) -> Optional[float]:
        """Upper bound of the bucket holding the q-th observation (conservative)."""
        if self.count == 0:
            return None
        target = math.ceil(q * self.count)
        seen = 0
        for i, n in enumerate(self.counts):
            seen += n
            if seen >= target:
                return self.buckets[i] if i < len(self.buckets) else math.inf
        return math.inf  # pragma: no cover

    def to_dict(self) -> Dict[str, Any]:
        def fmt(v):
            return "+Inf" if v is not None and math.isinf(v) else v

        buckets = {str(b): n for b, n in zip(self.buckets, self.counts)}
        buckets["+Inf"] = self.counts[-1]
        return {
            "count": self.count, "sum": round(self.sum, 6), "buckets": buckets,
            "p50": fmt(self.percentile(0.50)), "p95": fmt(self.percentile(0.95)),
            "p99": fmt(self.percentile(0.99)),
        }


class _Sum:
    def __init__(self):
        self.total = 0.0
        self.count = 0       # non-null observations
        self.null_count = 0  # events whose value was null (unknown)

    def observe(self, value: Optional[float]) -> None:
        if value is None:
            self.null_count += 1
        else:
            self.total += value
            self.count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {"sum": round(self.total, 6), "count": self.count, "null_count": self.null_count}


# =============================================================================
# AGGREGATOR
# =============================================================================

LabelKey = Tuple[Tuple[str, str], ...]


class Aggregator:
    def __init__(self, strict: bool = False, max_in_flight: int = MAX_IN_FLIGHT_REQUESTS):
        validate_specs()
        self.strict = strict
        self.max_in_flight = max_in_flight
        self._by_event: Dict[str, List[MetricSpec]] = {}
        for spec in METRICS:
            for event in spec.events:
                self._by_event.setdefault(event, []).append(spec)
        self._series: Dict[str, Dict[LabelKey, Any]] = {s.name: {} for s in METRICS}
        self._per_request_series: Dict[LabelKey, _Histogram] = {}
        self._in_flight: Dict[str, Counter] = {}
        self._unclassified = 0
        self._dropped_requests = 0
        self._events_observed = 0

    # ------------------------------------------------------------------ input

    def observe(self, event: Any) -> None:
        try:
            name, metadata = self._validate(event)
        except (KeyError, ValueError, TypeError):
            if self.strict:
                raise
            self._unclassified += 1
            return

        self._events_observed += 1
        for spec in self._by_event.get(name, ()):
            if spec.when is not None and not spec.when(event):
                continue
            self._apply(spec, name, event, metadata)
        self._track_request(name, event, metadata)

    def observe_all(self, events: Iterable[Any]) -> "Aggregator":
        for event in events:
            self.observe(event)
        return self

    @staticmethod
    def _validate(event: Any) -> Tuple[str, Dict[str, Any]]:
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise TypeError("not an event")
        name = event["event"]
        if name not in dim.events():
            raise KeyError(f"unclassified event: {name}")
        metadata = event.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be an object")
        for key in event:
            if key != "metadata":
                dim.classify(name, key)
        for key in metadata:
            dim.classify(name, key)
        return name, metadata

    # --------------------------------------------------------------- counters

    @staticmethod
    def _labels(spec: MetricSpec, name: str, event: Dict[str, Any], metadata: Dict[str, Any]) -> LabelKey:
        pairs = []
        for label in spec.labels:
            if label == "status":
                raw = event.get("status")
            elif label in metadata:
                raw = metadata[label]
            else:
                continue  # optional dimension absent: omitted, never synthesized
            pairs.append((label, dim.dimension_value(name, label, raw)))
        return tuple(pairs)

    def _apply(self, spec: MetricSpec, name: str, event: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        key = self._labels(spec, name, event, metadata)
        series = self._series[spec.name]

        if spec.kind == COUNTER:
            series[key] = series.get(key, 0) + 1
            return

        raw = event.get(spec.value) if spec.value == "duration_ms" else metadata.get(spec.value)
        if spec.kind == SUM:
            series.setdefault(key, _Sum()).observe(raw)
        elif spec.kind == HISTOGRAM:
            if raw is None:
                return  # unmeasured: never treated as 0
            series.setdefault(key, _Histogram(DURATION_BUCKETS_MS)).observe(raw)

    # ------------------------------------------------------ per-request signal

    def _track_request(self, name: str, event: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        rid = event.get("request_id")
        if not rid or rid == "-":
            return  # not inside a request: does not participate

        if name in _LLM_CALL_EVENTS:
            if rid not in self._in_flight:
                if len(self._in_flight) >= self.max_in_flight:
                    self._in_flight.pop(next(iter(self._in_flight)))
                    self._dropped_requests += 1
                self._in_flight[rid] = Counter()
            purpose = dim.dimension_value(name, "purpose", metadata.get("purpose"))
            self._in_flight[rid][purpose] += 1
        elif name == "request.completed":
            counts = self._in_flight.pop(rid, None)
            for purpose, n in sorted((counts or {}).items()):
                key = (("purpose", purpose),)
                self._per_request_series.setdefault(key, _Histogram(CALL_COUNT_BUCKETS)).observe(n)

    # --------------------------------------------------------------- snapshot

    def _total(self, name: str, **match: str) -> int:
        total = 0
        for key, value in self._series[name].items():
            labels = dict(key)
            if all(labels.get(k) == v for k, v in match.items()):
                total += value if isinstance(value, int) else value.count
        return total

    @staticmethod
    def _ratio(num: int, den: int) -> Optional[float]:
        return None if den == 0 else round(num / den, 6)

    def rates(self) -> Dict[str, Optional[float]]:
        t = self._total
        return {
            "request_error_rate": self._ratio(t("requests_total", status="error"), t("requests_total")),
            "request_block_rate": self._ratio(t("requests_total", status="blocked"), t("requests_total")),
            # Instrumented application request volume = requests that reached
            # the request lifecycle + requests the limiter rejected before it.
            "request_rate_limited_ratio": self._ratio(
                t("rate_limited_total"), t("requests_total") + t("rate_limited_total")),
            "guardrail_block_rate": self._ratio(t("guardrails_total", status="blocked"), t("guardrails_total")),
            "llm_failure_rate": self._ratio(t("llm_calls_total", status="failure"), t("llm_calls_total")),
            "llm_retry_rate": self._ratio(t("llm_retried_calls_total"), t("llm_calls_total", status="success")),
            "llm_fallback_rate": self._ratio(t("llm_fallbacks_total"), t("llm_calls_total")),
            "retrieval_empty_rate": self._ratio(t("retrieval_empty_total"), t("retrieval_total", status="success")),
            "retrieval_skip_rate": self._ratio(t("retrieval_total", status="skipped"), t("retrieval_total")),
            "retrieval_failure_rate": self._ratio(t("retrieval_total", status="failure"), t("retrieval_total")),
            "ranking_failure_rate": self._ratio(t("ranking_total", status="failure"), t("ranking_total")),
            "analytics_failure_rate": self._ratio(t("analytics_failures_total"), t("analytics_total")),
            "analytics_sql_fallback_rate": self._ratio(
                t("analytics_sql_total", used_fallback_sql="true"), t("analytics_sql_total")),
        }

    @staticmethod
    def _render(series: Dict[LabelKey, Any]) -> List[Dict[str, Any]]:
        out = []
        for key in sorted(series):
            value = series[key]
            out.append({
                "labels": dict(key),
                "value": value.to_dict() if hasattr(value, "to_dict") else value,
            })
        return out

    def snapshot(self) -> Dict[str, Any]:
        metrics = {name: self._render(series) for name, series in sorted(self._series.items()) if series}
        if self._per_request_series:
            metrics[PER_REQUEST_METRIC] = self._render(self._per_request_series)
        # Meta counters describe the aggregator itself, so 0 is a known value.
        metrics["signals_unclassified_total"] = [{"labels": {}, "value": self._unclassified}]
        metrics["signals_dropped_requests_total"] = [{"labels": {}, "value": self._dropped_requests}]
        return {
            "metrics": dict(sorted(metrics.items())),
            "rates": self.rates(),
            "meta": {
                "events_observed": self._events_observed,
                "in_flight_requests": len(self._in_flight),
                "note": _SERIES_NOTE,
            },
        }


def replay(events: Iterable[Any], strict: bool = False) -> Dict[str, Any]:
    """Deterministic: the same event sequence always yields the same snapshot."""
    return Aggregator(strict=strict).observe_all(events).snapshot()


# =============================================================================
# REPLAY / DEBUG CLI  (JSONL file -> replay -> snapshot on stdout)
# =============================================================================

def _read_events(stream) -> Tuple[List[Dict[str, Any]], int]:
    events, skipped = [], 0
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            skipped += 1  # e.g. human-readable log lines mixed into the file
            continue
        if isinstance(obj, dict) and "event" in obj:
            events.append(obj)
        else:
            skipped += 1
    return events, skipped


def main(argv: Optional[Sequence[str]] = None, stdin=None, stdout=None, stderr=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    stdin, stdout, stderr = stdin or sys.stdin, stdout or sys.stdout, stderr or sys.stderr

    if argv and argv[0] in ("-h", "--help"):
        stdout.write("usage: python -m src.observability.signals [events.jsonl]  (default: stdin)\n")
        return 0

    if argv and argv[0] != "-":
        with open(argv[0], encoding="utf-8") as fh:
            events, skipped = _read_events(fh)
    else:
        events, skipped = _read_events(stdin)

    stdout.write(json.dumps(replay(events), indent=2, sort_keys=True) + "\n")
    if skipped:
        stderr.write(f"skipped {skipped} non-event line(s)\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
