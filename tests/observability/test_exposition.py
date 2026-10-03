# tests/observability/test_exposition.py
"""
Tests for M8.3: Prometheus text exposition (src/observability/exposition.py).

M7.3 owns the analytical snapshot; M8.3 owns only its representation. The key
invariant is round-trip conservation:

    runtime events -> M7.3 snapshot -> Prometheus text -> parsed -> == snapshot
"""

import ast
import math
import pathlib
import re

import pytest

from src.observability import dimensions as dim
from src.observability import exposition as ex
from src.observability import signals as sig
from tests.observability.test_signals import (  # noqa: F401  (fixture + helpers reused)
    healthy_rag, llm_ok, request_done, ev, runtime_events,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]


# =============================================================================
# A small grammar parser for text format 0.0.4 (test-side, independent of the
# renderer) that also enforces the structural rules.
# =============================================================================

_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _parse_labels(text, i):
    """Parse '{k="v",...}' starting at text[i]=='{'. Returns (labels, next_index)."""
    labels, i = {}, i + 1
    while text[i] != "}":
        j = text.index("=", i)
        key = text[i:j]
        assert _LABEL.match(key), f"bad label name {key!r}"
        assert text[j + 1] == '"', "label value must be quoted"
        k, value = j + 2, []
        while text[k] != '"':
            if text[k] == "\\":
                nxt = text[k + 1]
                assert nxt in ('\\', '"', "n"), f"bad escape \\{nxt}"
                value.append({"\\": "\\", '"': '"', "n": "\n"}[nxt])
                k += 2
            else:
                assert text[k] != "\n", "raw newline inside label value"
                value.append(text[k])
                k += 1
        assert key not in labels, f"duplicate label {key}"
        labels[key] = "".join(value)
        i = k + 1
        if text[i] == ",":
            i += 1
    return labels, i + 1


def parse(text):
    """-> {family: {"type", "help", "samples": [(sample_name, labels, value)]}}"""
    assert text.endswith("\n")
    families, current = {}, None
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith("# HELP "):
            name, _, help_text = line[7:].partition(" ")
            assert _NAME.match(name)
            assert name not in families, f"HELP for {name} declared twice"
            families[name] = {"type": None, "help": help_text, "samples": []}
            current = name
        elif line.startswith("# TYPE "):
            name, _, kind = line[7:].partition(" ")
            assert name == current, "TYPE must follow its HELP"
            assert kind in ("counter", "gauge", "histogram"), kind
            families[name]["type"] = kind
        else:
            assert not line.startswith("#"), f"unexpected comment {line!r}"
            m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)", line)
            sample_name, i = m.group(1), m.end()
            labels = {}
            if line[i:i + 1] == "{":
                labels, i = _parse_labels(line, i)
            assert line[i] == " ", f"bad sample line {line!r}"
            raw = line[i + 1:]
            value = math.inf if raw == "+Inf" else float(raw)
            assert current is not None and families[current]["type"], "sample before TYPE"
            fam = families[current]
            allowed = {current}
            if fam["type"] == "histogram":
                allowed = {f"{current}_bucket", f"{current}_sum", f"{current}_count"}
            assert sample_name in allowed, f"{sample_name} not part of family {current}"
            fam["samples"].append((sample_name, labels, value))
    return families


def validate(families):
    for name, fam in families.items():
        keys = [(s, tuple(sorted(lbl.items()))) for s, lbl, _ in fam["samples"]]
        assert len(keys) == len(set(keys)), f"duplicate series in {name}"
        if fam["type"] == "counter":
            assert all(v >= 0 for _, _, v in fam["samples"]), f"negative counter in {name}"
        if fam["type"] == "histogram":
            by_labels = {}
            for sample, labels, value in fam["samples"]:
                base = tuple(sorted((k, v) for k, v in labels.items() if k != "le"))
                by_labels.setdefault(base, {"b": [], "sum": None, "count": None})
                if sample.endswith("_bucket"):
                    by_labels[base]["b"].append((labels["le"], value))
                elif sample.endswith("_sum"):
                    by_labels[base]["sum"] = value
                else:
                    by_labels[base]["count"] = value
            for base, h in by_labels.items():
                assert h["sum"] is not None and h["count"] is not None, f"{name}: missing _sum/_count"
                les = [math.inf if le == "+Inf" else float(le) for le, _ in h["b"]]
                assert les == sorted(les) and les[-1] == math.inf, f"{name}: le order / +Inf"
                counts = [v for _, v in h["b"]]
                assert counts == sorted(counts), f"{name}: buckets must be cumulative"
                assert counts[-1] == h["count"], f"{name}: +Inf bucket must equal _count"


def samples_of(fams, family, sample=None):
    sample = sample or family
    return {tuple(sorted(lbl.items())): v for s, lbl, v in fams[family]["samples"] if s == sample}


def render(events, **meta):
    snap = sig.replay(events)
    snap["meta"].update(meta)
    return snap, ex.render_prometheus(snap)


def rich_events():
    """Synthetic stream touching counters, sums (with nulls), histograms and +Inf."""
    events = healthy_rag("a") + healthy_rag("b")
    events += [
        llm_ok("language_detection", rid="c"), llm_ok("language_detection", rid="c"),
        llm_ok("language_detection", cost=None, rid="c"),
        request_done(rid="c", ms=12000, cost="unknown", total=None),
        ev("llm.fallback", metadata={"purpose": "rag_answer", "from_model": "gpt-4o",
                                     "to_model": "gpt-4o-mini", "reason": "budget_threshold",
                                     "cumulative_session_cost_usd": 0.2}),
    ]
    return events


# =============================================================================
# Format validity
# =============================================================================

def test_rendered_output_is_valid_text_format():
    snap, text = render(rich_events(), started_at_unix=1.7e9, handler_errors=0, ignored_records=0)
    fams = parse(text)
    validate(fams)
    assert all(f["help"] and f["type"] for f in fams.values())


def test_label_values_are_escaped_and_round_trip():
    nasty = 'a"b\\c\nd'
    snap = {"metrics": {"requests_total": [
        {"labels": {"route": nasty, "status": "success"}, "value": 3}]},
        "rates": {}, "meta": {"events_observed": 1, "in_flight_requests": 0}}
    text = ex.render_prometheus(snap)

    assert "\n" not in text.split("requests_total{")[1].split("}")[0]   # newline escaped
    fams = parse(text)
    assert samples_of(fams, "requests_total") == {(("route", nasty), ("status", "success")): 3}


def test_help_text_is_escaped(monkeypatch):
    monkeypatch.setitem(ex.HELP, "requests_total", "line1\nline2 \\ back")
    snap = sig.replay([request_done()])
    text = ex.render_prometheus(snap)
    assert "# HELP requests_total line1\\nline2 \\\\ back\n" in text


@pytest.mark.parametrize("labels", [{"1bad": "x"}, {"has-dash": "x"}, {"le": "x"}, {"__reserved": "x"}])
def test_invalid_or_reserved_label_names_are_rejected(labels):
    snap = {"metrics": {"requests_total": [{"labels": labels, "value": 1}]}, "rates": {},
            "meta": {"events_observed": 0, "in_flight_requests": 0}}
    with pytest.raises(ValueError):
        ex.render_prometheus(snap)


def test_no_policy_dimension_is_named_le_or_reserved():
    keys = {key for (_ev, key), rule in dim.POLICY.items() if rule.kind == dim.DIMENSION}
    assert "le" not in keys
    assert not any(k.startswith("__") for k in keys)
    assert all(_LABEL.match(k) for k in keys)


def test_unknown_metric_in_snapshot_is_an_error_not_silently_dropped():
    snap = {"metrics": {"mystery_total": [{"labels": {}, "value": 1}]}, "rates": {},
            "meta": {"events_observed": 0, "in_flight_requests": 0}}
    with pytest.raises(KeyError):
        ex.render_prometheus(snap)


def test_inconsistent_histogram_is_rejected_not_exposed():
    snap = sig.replay([request_done(ms=3)])
    snap["metrics"]["request_duration_ms"][0]["value"]["count"] = 99
    with pytest.raises(ValueError):
        ex.render_prometheus(snap)


# =============================================================================
# Cumulative histograms
# =============================================================================

def test_histogram_buckets_are_cumulative_with_inf_sum_count():
    snap, text = render([request_done(ms=3), request_done(ms=8), request_done(ms=90),
                         request_done(ms=12000)])
    fams = parse(text)
    validate(fams)
    key = (("route", "/query"), ("status", "success"))

    buckets = {dict(k)["le"]: v for k, v in
               {tuple(sorted(lbl.items())): v for s, lbl, v in fams["request_duration_ms"]["samples"]
                if s == "request_duration_ms_bucket"}.items()}
    assert buckets["5"] == 1 and buckets["10"] == 2 and buckets["100"] == 3
    assert buckets["10000"] == 3 and buckets["+Inf"] == 4          # >10s kept, cumulative
    assert samples_of(fams, "request_duration_ms", "request_duration_ms_count")[key] == 4
    assert samples_of(fams, "request_duration_ms", "request_duration_ms_sum")[key] == 12101


def test_bounds_render_as_stored():
    _, text = render([request_done(ms=3)])
    assert 'le="1"' in text and 'le="10000"' in text and 'le="+Inf"' in text
    assert 'le="1.0"' not in text


# =============================================================================
# Sum expansion
# =============================================================================

def test_sums_expand_into_total_observations_and_unknown():
    snap, text = render([llm_ok(cost=0.001), llm_ok(cost=0.002), llm_ok(cost=None)])
    fams = parse(text)
    key = (("model", "gpt-4o-mini"), ("purpose", "rag_answer"))

    assert samples_of(fams, "llm_cost_usd_total")[key] == pytest.approx(0.003)
    assert samples_of(fams, "llm_cost_usd_observations_total")[key] == 2
    assert samples_of(fams, "llm_cost_usd_unknown_total")[key] == 1
    # the separate M7.3 counter overlaps by design; both exist, neither collides
    assert samples_of(fams, "llm_cost_unknown_total")[key] == 1
    assert all(fams[n]["type"] == "counter" for n in
               ("llm_cost_usd_total", "llm_cost_usd_observations_total", "llm_cost_usd_unknown_total"))


def test_family_names_never_collide():
    names = ex.family_names()
    assert len(names) == len(set(names))
    assert all(_NAME.match(n) for n in names)


# =============================================================================
# Round-trip conservation (the key invariant)
# =============================================================================

def _assert_conserved(snap, text):
    fams = parse(text)
    validate(fams)
    kinds = ex._kinds()

    for name, series in snap["metrics"].items():
        kind = kinds[name]
        if kind == sig.COUNTER:
            got = samples_of(fams, name)
            assert got == {tuple(sorted(s["labels"].items())): s["value"] for s in series}, name
        elif kind == sig.SUM:
            for fam_name, field in ex._sum_family_names(name):
                got = samples_of(fams, fam_name)
                want = {tuple(sorted(s["labels"].items())): s["value"][field] for s in series}
                assert got.keys() == want.keys(), fam_name
                for k in want:
                    assert got[k] == pytest.approx(want[k]), (fam_name, k)
        elif kind == sig.HISTOGRAM:
            counts = samples_of(fams, name, f"{name}_count")
            sums = samples_of(fams, name, f"{name}_sum")
            for s in series:
                key = tuple(sorted(s["labels"].items()))
                assert counts[key] == s["value"]["count"]
                assert sums[key] == pytest.approx(s["value"]["sum"])
                # de-cumulate and compare with the analytical (non-cumulative) buckets
                cumulative = sorted(
                    ((math.inf if dict(k)["le"] == "+Inf" else float(dict(k)["le"]), v)
                     for k, v in samples_of(fams, name, f"{name}_bucket").items()
                     if tuple(x for x in k if x[0] != "le") == key))
                prev, plain = 0, {}
                for le, v in cumulative:
                    plain["+Inf" if math.isinf(le) else str(int(le))] = int(v - prev)
                    prev = v
                assert plain == s["value"]["buckets"], (name, key)

    meta = snap["meta"]
    assert samples_of(fams, "signals_events_observed_total") == {(): meta["events_observed"]}
    assert samples_of(fams, "signals_in_flight_requests") == {(): meta["in_flight_requests"]}
    return fams


def test_round_trip_conserves_a_synthetic_snapshot():
    snap, text = render(rich_events())
    _assert_conserved(snap, text)


def test_round_trip_conserves_runtime_captured_events(runtime_events):   # noqa: F811
    """runtime events -> M7.3 snapshot -> Prometheus text -> parsed == snapshot"""
    snap = sig.replay(runtime_events, strict=True)
    fams = _assert_conserved(snap, ex.render_prometheus(snap))
    assert "requests_total" in fams and "llm_calls_total" in fams
    assert "analytics_sql_total" in fams and "retrieval_total" in fams


# =============================================================================
# Absence is not zero; derived values are not exported
# =============================================================================

def test_unobserved_families_are_omitted():
    _, text = render(healthy_rag())
    fams = parse(text)
    for absent in ("llm_fallbacks_total", "retrieval_empty_total", "llm_failures_total",
                   "analytics_total", "llm_retries_total_unknown_total"):
        assert absent not in fams
    assert "requests_total" in fams


def test_empty_snapshot_has_only_meta_families():
    _, text = render([])
    fams = parse(text)
    assert set(fams) == {"signals_unclassified_total", "signals_dropped_requests_total",
                         "signals_events_observed_total", "signals_in_flight_requests"}
    assert samples_of(fams, "signals_unclassified_total") == {(): 0}      # zero is known here
    assert samples_of(fams, "signals_events_observed_total") == {(): 0}


def test_live_meta_and_start_time_appear_only_when_present():
    _, replayed = render(healthy_rag())
    assert "signals_start_time_seconds" not in parse(replayed)

    _, live_text = render(healthy_rag(), started_at_unix=1700000000.5, handler_errors=2,
                          ignored_records=1)
    fams = parse(live_text)
    assert fams["signals_start_time_seconds"]["type"] == "gauge"
    assert samples_of(fams, "signals_start_time_seconds") == {(): 1700000000.5}
    assert fams["signals_in_flight_requests"]["type"] == "gauge"
    assert samples_of(fams, "signals_handler_errors_total") == {(): 2}
    assert samples_of(fams, "signals_ignored_records_total") == {(): 1}


def test_rates_and_percentiles_are_not_exported():
    snap, text = render(rich_events())
    assert snap["rates"] and any(isinstance(m["value"], dict) and "p95" in m["value"]
                                 for series in snap["metrics"].values() for m in series
                                 if isinstance(m["value"], dict))
    for rate_name in snap["rates"]:
        assert rate_name not in text
    assert "p50" not in text and "p95" not in text and "p99" not in text


def test_units_keep_milliseconds_names():
    _, text = render(rich_events())
    fams = parse(text)
    assert "request_duration_ms" in fams and "llm_duration_ms" in fams
    assert not [n for n in fams if n.endswith("_seconds") and n != "signals_start_time_seconds"]


# =============================================================================
# Determinism
# =============================================================================

def test_rendering_is_deterministic_and_independent_of_input_order():
    snap, text = render(rich_events(), started_at_unix=1.0)
    assert ex.render_prometheus(snap) == text

    shuffled = {
        "metrics": {k: list(reversed(v)) for k, v in reversed(list(snap["metrics"].items()))},
        "rates": snap["rates"],
        "meta": dict(reversed(list(snap["meta"].items()))),
    }
    assert ex.render_prometheus(shuffled) == text


def test_families_are_sorted_by_name_and_le_is_last():
    _, text = render(rich_events())
    names = re.findall(r"^# TYPE (\S+) ", text, re.M)
    assert names == sorted(names)
    assert re.search(r'request_duration_ms_bucket\{route="[^"]+",status="[^"]+",le="1"\}', text)


# =============================================================================
# Coverage: every metric is documented; inventory changes cannot skip HELP
# =============================================================================

def test_every_metric_in_the_inventory_has_help_and_type():
    names = [spec.name for spec in sig.METRICS] + [sig.PER_REQUEST_METRIC,
             "signals_unclassified_total", "signals_dropped_requests_total"]
    for name in names:
        assert name in ex.HELP, f"{name} has no HELP text"
    for family in ex.family_names():
        assert ex._help_for(family)


def test_every_family_the_exporter_can_emit_is_a_valid_name_with_one_line_help():
    for family in ex.family_names():
        assert _NAME.match(family)
        assert "\n" not in ex._help_for(family)


# =============================================================================
# Purity / boundary
# =============================================================================

def test_exporter_is_pure_and_independent_of_the_aggregator_internals():
    tree = ast.parse((ROOT / "src" / "observability" / "exposition.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"os", "logging", "threading", "fastapi", "starlette", "prometheus_client",
                           "opentelemetry", "json"}
    # it reads only the public snapshot shape + the metric inventory
    text = (ROOT / "src" / "observability" / "exposition.py").read_text(encoding="utf-8")
    assert "sig.Aggregator" not in text and "Aggregator(" not in text and "._series" not in text
