# src/observability/exposition.py
"""
Prometheus text exposition (M8.3).

    M7.3 snapshot (analytical)  --render_prometheus-->  text format 0.0.4

M7.3 owns the analytical snapshot semantics; this module owns only a
REPRESENTATION of it. It never touches the aggregator, does no I/O, reads no
environment, starts no threads, and uses no prometheus_client / registry.
Rates and percentiles are NOT exported: consumers derive them from the
counters and histogram buckets (rate(), histogram_quantile()).

Mapping:
  counter   x_total                  -> TYPE counter
  sum       foo_total {sum,count,null_count}
                                     -> foo_total                (counter: the sum)
                                        foo_observations_total   (counter: non-null count)
                                        foo_unknown_total        (counter: null count)
  histogram x                        -> x_bucket{...,le=...} (CUMULATIVE, +Inf), x_sum, x_count
  meta                               -> signals_* counters and gauges

A family with no observed series is omitted (absence is not zero); only the
meta families, whose zero is a known value, are always present. Duration
metrics keep their `_ms` names and values (an intentional deviation from the
seconds convention; any seconds view would be an explicit mapping decision).

Text format 0.0.4 only: no OpenMetrics, no content negotiation. Output is
deterministic: families sorted by name, series sorted, label names sorted
with `le` last.
"""

import math
import re
from typing import Any, Dict, Iterable, List, Tuple

from src.observability import signals as sig

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

COUNTER = "counter"
GAUGE = "gauge"
HISTOGRAM = "histogram"

_METRIC_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Always-present meta families (zero is a known value) vs live-only ones.
_META_COUNTERS_FROM_METRICS = ("signals_unclassified_total", "signals_dropped_requests_total")

# --------------------------------------------------------------------------
# HELP text: static, one line, no dynamic content.
# --------------------------------------------------------------------------
HELP: Dict[str, str] = {
    "requests_total": "Completed API requests by route and request status.",
    "request_duration_ms": "Request duration in milliseconds.",
    "request_cost_usd_total": "Known estimated LLM cost in USD per request outcome; a lower bound when partial.",
    "rate_limited_total": "Requests rejected by the rate limiter before the request lifecycle, by route.",
    "llm_calls_total": "Terminal LLM calls by purpose, model and status.",
    "llm_duration_ms": "LLM call duration in milliseconds (whole retry loop).",
    "llm_retries_total": "Retries beyond the first attempt across LLM calls.",
    "llm_retried_calls_total": "Completed LLM calls that needed at least one retry.",
    "llm_failures_total": "LLM calls that exhausted retries, by error class.",
    "llm_fallbacks_total": "Budget-threshold model downgrades by purpose and reason.",
    "llm_prompt_tokens_total": "Prompt tokens reported by the provider.",
    "llm_completion_tokens_total": "Completion tokens reported by the provider.",
    "llm_cost_usd_total": "Known estimated LLM cost in USD (unpriced responses excluded).",
    "llm_cost_unknown_total": "Completed LLM calls whose cost could not be determined.",
    "retrieval_total": "Retrieval stage outcomes.",
    "retrieval_empty_total": "Successful retrievals that returned no candidates.",
    "retrieval_failures_total": "Failed retrievals by error class.",
    "ranking_total": "Ranking stage outcomes.",
    "analytics_total": "Analytics pipeline outcomes by intent.",
    "analytics_sql_total": "Analytics SQL resolution outcomes.",
    "analytics_failures_total": "Analytics pipeline failures by error class.",
    "intent_total": "Intent detection outcomes.",
    "guardrails_total": "Guardrail decisions by reason.",
    "language_detection_total": "Language detection stage outcomes.",
    "llm_calls_per_request": "LLM calls made by one request, per purpose.",
    "signals_unclassified_total": "Events the aggregator could not classify and skipped.",
    "signals_dropped_requests_total": "In-flight request states evicted by the cap.",
    "signals_events_observed_total": "Events observed by the aggregator since it started.",
    "signals_handler_errors_total": "Live-feed handler errors swallowed since start.",
    "signals_ignored_records_total": "Non-event records ignored by the live feed since start.",
    "signals_in_flight_requests": "Requests with LLM-call state currently tracked.",
    "signals_start_time_seconds": "Aggregator start time (Unix seconds); a change means counters reset.",
}

_SUM_SUFFIXES = (
    ("", "sum"),
    ("_observations_total", "count"),
    ("_unknown_total", "null_count"),
)


def _kinds() -> Dict[str, str]:
    kinds = {spec.name: spec.kind for spec in sig.METRICS}
    kinds[sig.PER_REQUEST_METRIC] = sig.HISTOGRAM
    for name in _META_COUNTERS_FROM_METRICS:
        kinds[name] = sig.COUNTER
    return kinds


def _sum_family_names(name: str) -> List[Tuple[str, str]]:
    base = name[: -len("_total")] if name.endswith("_total") else name
    return [(f"{base}_total" if suffix == "" else f"{base}{suffix}", field) for suffix, field in _SUM_SUFFIXES]


def family_names() -> List[str]:
    """Every exposition family this module can emit (for collision/coverage tests)."""
    names: List[str] = []
    for name, kind in _kinds().items():
        if kind == sig.SUM:
            names += [n for n, _ in _sum_family_names(name)]
        else:
            names.append(name)
    names += ["signals_events_observed_total", "signals_handler_errors_total",
              "signals_ignored_records_total", "signals_in_flight_requests",
              "signals_start_time_seconds"]
    return sorted(names)


def _help_for(family: str) -> str:
    if family in HELP:
        return HELP[family]
    if family.endswith("_observations_total"):
        return "Non-null observations behind the corresponding sum."
    if family.endswith("_unknown_total"):
        return "Observations whose value was unknown (null) for the corresponding sum."
    raise KeyError(f"no HELP text for family {family!r}")


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _escape_help(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(value)


def _check_names(name: str, labels: Dict[str, str], histogram_bucket: bool = False) -> None:
    if not _METRIC_NAME.match(name):
        raise ValueError(f"invalid metric name: {name!r}")
    for label in labels:
        if not _LABEL_NAME.match(label) or label.startswith("__"):
            raise ValueError(f"invalid label name: {label!r}")
        if label == "le":
            raise ValueError("label 'le' is reserved for histogram buckets")


def _sample(name: str, labels: Dict[str, str], value: Any, le: str = None) -> str:
    _check_names(name, labels)
    pairs = [f'{k}="{_escape_label_value(str(labels[k]))}"' for k in sorted(labels)]
    if le is not None:
        pairs.append(f'le="{_escape_label_value(le)}"')
    body = "{" + ",".join(pairs) + "}" if pairs else ""
    return f"{name}{body} {_fmt_value(value)}"


def _bound_key(key: str) -> float:
    return math.inf if key == "+Inf" else float(key)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

class _Family:
    def __init__(self, name: str, kind: str):
        self.name, self.kind = name, kind
        self.lines: List[str] = []

    def render(self) -> List[str]:
        return [
            f"# HELP {self.name} {_escape_help(_help_for(self.name))}",
            f"# TYPE {self.name} {self.kind}",
        ] + self.lines


def _series_sorted(series: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(series, key=lambda s: sorted(s["labels"].items()))


def _histogram_lines(name: str, series: Iterable[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for s in _series_sorted(series):
        labels, hist = s["labels"], s["value"]
        running = 0
        for key in sorted(hist["buckets"], key=_bound_key):
            running += hist["buckets"][key]
            lines.append(_sample(f"{name}_bucket", labels, running, le=key))
        if running != hist["count"]:
            raise ValueError(f"{name}: bucket counts ({running}) do not match count ({hist['count']})")
        lines.append(_sample(f"{name}_sum", labels, hist["sum"]))
        lines.append(_sample(f"{name}_count", labels, hist["count"]))
    return lines


def render_prometheus(snapshot: Dict[str, Any]) -> str:
    """Render an M7.3 snapshot as Prometheus text format 0.0.4 (pure, deterministic)."""
    kinds = _kinds()
    families: Dict[str, _Family] = {}

    def family(name: str, kind: str) -> _Family:
        if name in families:
            raise ValueError(f"duplicate exposition family: {name}")
        families[name] = _Family(name, kind)
        return families[name]

    for name, series in snapshot.get("metrics", {}).items():
        if not series:
            continue  # absence is not zero
        kind = kinds.get(name)
        if kind is None:
            raise KeyError(f"unknown metric in snapshot: {name!r}")
        ordered = _series_sorted(series)

        if kind == sig.COUNTER:
            fam = family(name, COUNTER)
            fam.lines = [_sample(name, s["labels"], s["value"]) for s in ordered]
        elif kind == sig.SUM:
            for fam_name, field in _sum_family_names(name):
                fam = family(fam_name, COUNTER)
                fam.lines = [_sample(fam_name, s["labels"], s["value"][field]) for s in ordered]
        elif kind == sig.HISTOGRAM:
            fam = family(name, HISTOGRAM)
            fam.lines = _histogram_lines(name, ordered)

    meta = snapshot.get("meta", {})
    meta_families = (
        ("signals_events_observed_total", COUNTER, meta.get("events_observed")),
        ("signals_handler_errors_total", COUNTER, meta.get("handler_errors")),
        ("signals_ignored_records_total", COUNTER, meta.get("ignored_records")),
        ("signals_in_flight_requests", GAUGE, meta.get("in_flight_requests")),
        ("signals_start_time_seconds", GAUGE, meta.get("started_at_unix")),
    )
    for name, kind, value in meta_families:
        if value is None:
            continue  # live-only fields are absent from a plain replay snapshot
        fam = family(name, kind)
        fam.lines = [_sample(name, {}, value)]

    out: List[str] = []
    for name in sorted(families):
        out += families[name].render()
    return "\n".join(out) + "\n"
