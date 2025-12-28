# Observability Design

This document explains how observability is designed in the Fraud Agents Enhanced system.

The goal is **not dashboards or tools**, but the ability to answer a simple operational question at any time:

> “What happened for this query?”

This layer focuses on **traceability, cost awareness, and failure understanding** without premature infrastructure complexity.

## Design Philosophy

Observability is treated as a **system design concern**, not an infrastructure add-on.

Principles:
- Log **structured events**, not free-form text
- Correlate all steps using a **single query identifier**
- Treat **cost, latency, and confidence** as first-class signals
- Prefer clarity over volume
- Avoid premature observability tooling

## Core Questions the System Can Answer

For any request, the system can answer:

- Which route was selected (RAG vs Analytics)
- Which model was used
- How long each step took
- How much the request cost
- Whether fallbacks were triggered
- Whether the response was best-effort or confident
- Why a failure or refusal occurred (if any)

If the system cannot answer these questions, observability is considered insufficient.

## Query-Level Tracing

Each incoming request is assigned a unique `query_id` at the orchestrator entry point.

This `query_id` is propagated across all internal steps:
- Guardrails
- Language detection
- Intent classification
- Routing
- RAG or Analytics execution
- Insight generation
- Scoring
- Final response assembly

The `query_id` is the backbone of tracing and correlation.

## Structured Event Logging

Each major step emits exactly one structured log event.

Each event includes:
- query_id
- step_name
- success (true / false)
- latency_ms
- metadata (optional)

Examples of step names:
- guardrail_checked
- intent_classified
- route_selected
- rag_executed
- analytics_executed
- insight_generated
- response_finalized

Logs are emitted as structured records (JSON-style), even when written to stdout.

## Cost Observability

Cost is treated as an operational metric, not an afterthought.

The system tracks:
- Token usage per LLM call
- Estimated cost per call
- Accumulated cost per query

Cost data is:
- Logged per step
- Aggregated at the query level
- Used to enforce fallback behavior when budget limits are exceeded

This enables debugging, budgeting, and responsible model usage.

## Latency Tracking

Latency is measured per step, not only end-to-end.

Each event records:
- Step-level latency in milliseconds

This allows:
- Identifying slow components
- Understanding routing tradeoffs
- Detecting regressions after changes

## Failure and Degradation Tracking

Failures are classified and logged explicitly.

Examples:
- Guardrail rejection
- Missing retrieval context
- Fallback answer triggered
- LLM retry exhaustion
- Low confidence score

Each failure records a `failure_reason`, allowing post-mortem analysis instead of silent degradation.

## What Is Intentionally Not Implemented

The system intentionally does NOT include:
- OpenTelemetry
- Prometheus metrics
- Distributed tracing infrastructure
- Dashboards or alerting systems

Reason:
Observability requirements must be clear before tooling is added.
Premature infrastructure often hides poor observability design.

## Mapping to Real Production Systems

This observability design maps cleanly to production setups:

- query_id → trace_id
- structured logs → log aggregation systems
- cost logs → billing & budget monitors
- step latency → performance monitoring

The system is designed so that observability tooling can be added later without architectural changes.


## Summary

Observability in this system is designed to answer:
- What happened?
- Why did it happen?
- How much did it cost?
- How confident is the result?

This layer prioritizes **traceability, cost awareness, and operational clarity**
over dashboards and tooling, reflecting real-world enterprise AI system needs.
