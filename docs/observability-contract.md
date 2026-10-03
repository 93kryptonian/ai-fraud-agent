# Observability Contract (reconciled after M6, through M7.3)

Status: **reconciled after M6 and updated through M8.4.** M6 is implemented
and frozen at `513f4b4`; M7.1-M7.3 and M8.1-M8.4 are implemented on top of it.
Every section describes what the code emits or does today: request-level cost
(§8), the metric dimension policy (§9), operational signals (§13), the live
feed, `/signals`, `/metrics` and `rate_limit.blocked` (§14), and the operating
notes (§15). No section is design-only any more; a section that describes
something not yet built must say so explicitly.

This document began as the M2 contract. It has been reconciled against the
actual code: events the M2 text promised but the code never emitted were
removed, event names that differ from the implementation were corrected, and
post-M6 changes are recorded in §10. It does not rewrite history; where the
original intent differs from the implementation, §11 says so.

Milestone sequence:

```text
M1 baseline freeze
→ M2 observability contract
→ M3 request context
→ M4 structured events
→ M5 timing/lifecycle
→ M6 pipeline instrumentation
→ M7 operational signals
→ M8 exposure/export
```

The target capability, unchanged since M2:

> Given a `request_id`, reconstruct the full execution path of that request —
> every stage it passed through, in order, with latency, status, and enough
> metadata to explain *why* it produced the answer it did — without exposing
> the user's query text, document contents, or LLM prompt/completion text.

---

## 1. Scope: the real request lifecycle [implemented]

Three entry points; they do not share the same stages:

```
POST /query      → validate_query() → run_query()      [orchestrator: full pipeline]
POST /rag        → validate_query() → run_rag()          [RAG only, no intent routing]
POST /analytics  → validate_query() → run_analytics()     [analytics only]
```

`run_query()` does, in order:

```
sanitize_input
  → detect_language                        (observed stage: language_detection)
    → detect_intent                        (heuristic; LLM if heuristic confidence < 0.80)
      → [reject]    → done
      → [analytics] → run_analytics(...)   → done
      → [rag]       → process_query (rewrite/translate)
                    → run_rag(...)         [retrieve (+rank) → build_context → llm.run]
                    → score_answer         (not instrumented)
                    → [low-confidence fallback, final_score < 0.12] → done
                    → generate_insight     (LLM purpose rag_insight)
                    → translate_en_to_id   (if user_lang == "id")   → done
```

Routing note: ranking/aggregation questions over the transaction dataset
(for example "which merchants have the highest fraud incidence") route to
**analytics**; conceptual and report-based questions route to **RAG**.
Ambiguous queries (heuristic confidence < 0.80) are decided by the LLM.

`retrieve_top_k` is one function with two observable stages: retrieval
(embed query → `match_documents` RPC) and ranking (`rerank_chunks`).
`run_rag()` never calls them separately, so the stage boundary lives inside
`src/rag/retriever_direct.py`.

`run_analytics()` does: `classify_analytics_intent` → `nl_to_sql` (template,
or LLM-generated for the `generic` intent only) → `execute_sql` → summarize
→ optional `refine_summary_with_llm`.

The LLM client (`llm.run`) is called from several sites, and one request can
invoke it multiple times. `llm.completed` is therefore a repeatable event
tagged with `purpose`, never a once-per-request event.

Rate-limited requests (HTTP 429) are rejected in middleware **before** the
router runs. They get no `request_id`, have no `request.completed`, and emit a
single bounded `rate_limit.blocked` event (M8.4, §4, §14).

---

## 2. Request-level record [implemented, narrower than M2 intended; cost added in M7.1]

One `request.completed` event per request, emitted by the router handler
(exactly one, including on an unhandled exception).

| Field | Where | Notes |
|---|---|---|
| `request_id` | envelope | uuid4, assigned in `api/routers.py` before `validate_query()`, so even a guardrail rejection is correlatable. |
| `status` | envelope | Request-level: `success`, `blocked`, `error`. A rate-limited request has no `request.completed`; the `rate_limited` request status is not emitted, and the rejection is recorded by `rate_limit.blocked` instead (§4, §14). `success` means the handler completed normally; it does **not** assert the business operation succeeded — a pipeline that catches its own exception and returns an `error` dict still yields `success` here. |
| `duration_ms` | envelope | Real handler duration, integer ms. |
| `metadata.route` | metadata | `/query`, `/rag`, `/analytics`. |
| `metadata.error_type` | metadata | Present only when `status == "error"`. Exception class name only. |
| `metadata.cost_status` | metadata | M7.1. `not_applicable \| complete \| partial \| unknown`, on every `request.completed` (success, blocked and error). See §8. |
| `metadata.cost_usd_total` | metadata | M7.1. `float` (rounded to 6 decimals), or `null` when `cost_status == "unknown"`. See §8. |

Fields the M2 contract listed for this record that are **not emitted today**:
`intent`, `lang`, `started_at`/`completed_at`, `error` (message),
`fallback_used`, `fallback_reason`. The `intent` is available from
`intent.completed`. (Request-level cost was in this list until M7.1.)

---

## 3. Event envelope [implemented]

```json
{
  "schema_version": 1,
  "timestamp": "...",
  "request_id": "...",
  "event": "retrieval.completed",
  "step": "retrieval",
  "status": "success",
  "duration_ms": 87,
  "metadata": { "...": "stage-specific, see §4" }
}
```

- `schema_version` increments only on a breaking change (removing or
  renaming a field, or changing what an existing field means). Adding an
  optional `metadata` key is not breaking.
- `event` is `"{step}.{outcome}"`.
- Stage-level `status` is one of `success | failure | blocked | skipped`.
  Request-level `status` (§2) is a different enum; the two are not unified.
- `duration_ms` is an integer in milliseconds. It is `null` for start
  markers and unmeasured events, never a fabricated `0`. A measured stage
  faster than the rounding resolution can legitimately report `0`; do not
  force a minimum of `1`.
- Parent/child durations are **not additive**; do not treat a parent's
  duration as the sum of its children.
- A stage that starts emits exactly one terminal event.

---

## 4. Event catalog [implemented]

Only events the code actually emits are listed.

| Event | Emitted from | metadata |
|---|---|---|
| `request.started` | `api/routers.py` | `route` |
| `guardrails.completed` / `guardrails.blocked` | `api/routers.py` (decision from `validate_query`) | `blocked`, `reason` (`too_short\|noise\|injection\|out_of_domain\|null`), `query_length`, `query_hash` |
| `language_detection.completed` / `.failed` | `src/orchestrator.py` via `observe_step` | none (`observe_step` carries no metadata by design) |
| `intent.completed` / `intent.failed` | `src/orchestrator.py` | completed: `intent`, `confidence` (`null` when decided by the LLM), `method` (`heuristic\|llm`), `route`. failed: `error_type` |
| `retrieval.completed` | `src/rag/retriever_direct.py` | `retrieval_method` (`vector_rpc`), `candidate_count`, `source_filter`. A successful retrieval with zero candidates is still `completed` with `candidate_count=0`. |
| `retrieval.skipped` | same | `retrieval_method`, `reason` (`retriever_disabled\|no_embedding`), `candidate_count=0`, `selected_count=0`, `source_filter` |
| `retrieval.failed` | same | `retrieval_method`, `error_type`, `source_filter` |
| `ranking.completed` | same | `candidate_count`, `selected_count`, `reranker` (`hybrid`; production always uses `use_llm=False`) |
| `ranking.skipped` | same | `candidate_count=0`, `selected_count=0` (retrieval succeeded with nothing to rank) |
| `ranking.failed` | same | `candidate_count`, `error_type` |
| `llm.completed` | `src/llm/llm_client.py::LLMClient.run` | `purpose`, `model`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `estimated_cost_usd`, `retry_count`. Token and cost fields are `null` when the provider returned no usage. One terminal event per `run()` call, covering the whole retry loop. |
| `llm.failed` | same | `purpose`, `model`, `retry_count`, `error_type`. Emitted once when retries are exhausted; `run()` then raises `LLMExhaustedRetriesError`. |
| `llm.fallback` | same | `purpose`, `from_model`, `to_model`, `reason` (`budget_threshold`, the only implemented reason), `cumulative_session_cost_usd` |
| `analytics.sql.completed` | `src/analytics/fraud_analytics.py` | `intent`, `used_fallback_sql`, `primary_error_type`, `row_count` |
| `analytics.sql.failed` | same | `intent`, `used_fallback_sql`, `primary_error_type`, `error_type` |
| `analytics.completed` | same | success: `intent`, `confidence`, `chart_generated`. failure: `error_type`. An honest "insufficient data" answer is `success`; only an unexpected internal failure is `failure`. |
| `request.completed` | `api/routers.py` | `route`, `cost_status`, `cost_usd_total`; plus `error_type` on error (§2) |
| `rate_limit.blocked` | `src/safety/rate_limit.py` (middleware, M8.4) | `route` only: `/query`, `/rag`, `/analytics`, else `other`. `status=blocked`, `duration_ms=null`, `request_id` is `-` (no request context exists yet). Never a raw path or any client identifier. Not emitted for `/signals` and `/metrics`. |

`llm.purpose` values emitted by code: `intent_classification`,
`language_detection`, `translation`, `query_rewrite`, `rag_answer`,
`rag_insight`, `analytics_nl_to_sql`, `analytics_summary`, `llm_rerank`
(`llm_rerank` exists as a call site but is not reached in production,
because ranking runs with `use_llm=False`).

**Removed from the catalog (never emitted by the code):**

| Removed | Why / current equivalent |
|---|---|
| `analytics.sql_executed` | Replaced by `analytics.sql.completed` / `analytics.sql.failed`. |
| `retrieval.empty` | There is no such event. Zero results after a successful retrieval is `retrieval.completed` with `candidate_count=0`; a retrieval that did not run is `retrieval.skipped`. |
| `scoring.completed` | `score_answer` is not instrumented. |
| `fallback.low_confidence` | The `final_score < 0.12` fallback in the orchestrator emits no event. |

Re-adding any of these is an explicit decision, not an assumed catalog entry.
(`rate_limit.blocked` was in this list until M8.4 added it deliberately.)

Not instrumented individually (by design): `sanitize_input`, the internal
steps of `process_query`, `build_context`, `to_chart_data`.

---

## 5. Known gaps [status after M6]

Fixed:
1. **`LLMClient.run` never raising** — fixed. Exhausted retries raise
   `LLMExhaustedRetriesError` and emit `llm.failed`; no sentinel string.
2. **Per-request cost accumulation** — fixed as a prerequisite. A
   request-scoped accumulator (`src/observability/cost.py`) sums per-call
   estimated cost. Exposed on `request.completed` as of M7.1 (§8).
3. **`details=str(e)` information leak** — fixed after M6 (§10).

Still open:
4. **Pipelines catch broad `Exception`** and return `error` dicts, so
   `request.completed(status=success)` does not imply a successful business
   outcome. Documented in §2; a business-outcome field is not part of this
   contract.
5. **Process-global budget guard** (`SESSION_COST_USD`) is intentionally
   global and separate from request attribution (§8).
6. **Rate-limited requests** were invisible to the event stream until M8.4; they are now counted via `rate_limit.blocked` (§14). Instrumented application request volume is `requests_total + rate_limited_total`; it is still not every possible HTTP request to the server.

---

## 6. Privacy rules [implemented, non-negotiable]

Under UU PDP's data-minimization principle (exact articles to be confirmed
with legal/compliance), telemetry that is not needed to answer "what
happened?" must not exist.

- Never put raw query text in a structured event. Use `query_hash` (sha256,
  first 12 hex chars) and `query_length`.
- Never put document or chunk content in an event. Only counts and
  identifiers-free aggregates are emitted today.
- Never put prompts or completions in an event. Token counts and cost only.
- Never put exception messages in an event. Use `error_type` (class name).
- **Executed SQL is returned to the API caller in the analytics response
  (`sql`), but is excluded from telemetry.** The response field grounds the
  answer; events carry only `intent`, `used_fallback_sql`, `row_count` and
  error types.
- Client IP is held in memory by the rate limiter only. It is never emitted
  in an event and, as of M8.4, never written to a log line: the former
  `Blocked ip=...` warning now logs only the bounded route and the window
  count. `rate_limit.blocked` carries the route and nothing that identifies a
  client.

---

## 7. Failure semantics [implemented]

- Instrumentation emits the failure event and then re-raises (or returns the
  same value it would have without instrumentation). Observability never
  changes what the caller receives.
- A stage that did not run emits `skipped`, not silence.
- "No exception seen" is not "succeeded" in this codebase: see §5 item 4.
- The retriever fails closed (empty result) rather than raising. The RAG
  result now carries `retrieval_status` (`ok|empty|unavailable`) so callers
  can tell an outage from a genuine no-result; the event stream already
  distinguishes the two via `retrieval.failed`/`.skipped` vs
  `retrieval.completed` with `candidate_count=0`.

---

## 8. Request-level cost contract [implemented in M7.1]

`request.completed` carries `cost_status` and `cost_usd_total` on every
outcome (success, blocked, error) of every route. A blocked request can
carry cost: guardrails may call the LLM for language detection before it
blocks.

Cost is derived from **responses received**, never from attempts:

| Observation | Effect |
|---|---|
| Response with usage, model priced | known cost added (priced response) |
| Response with usage, model has no price | unpriced response |
| Response without usage | unpriced response |
| Attempt raised before any response | nothing: no cost, no unknown count |
| Retry then success | every response with usage counts |
| Fallback model | both models' costs sum into the request |

```text
cost_status:
  not_applicable   no response was received (including: every call failed
                   before a response) -> cost_usd_total = 0.0
  complete         every response received was priced -> known total
  partial          >=1 priced and >=1 unpriced response -> known lower bound
  unknown          only unpriced responses -> null
```

`complete` means every LLM response received by the request had an
attributable, known price. Calls that fail before a response containing
usage is received do not affect cost completeness; their reliability is
reported by `llm.failed`, not by cost. A request whose only LLM calls all
failed is therefore `not_applicable`, not `complete`.

Rules:
- **Unknown is never `$0.00`.** `estimate_cost()` returns `None` for a model
  with no price, and `llm.completed.estimated_cost_usd` is `null` in that
  case (and when the provider returns no usage). The per-call event carries
  no `cost_status`; completeness is a request-level aggregation.
- Input and output tokens are priced separately
  (`PRICES_PER_1M = (input, output)` per model). Model matching is **exact**
  on the requested name: an alias such as `gpt-4o-mini` does not imply its
  dated snapshots, which may be priced differently.
- Usage is recorded the moment a response arrives, before any parsing, so a
  billable response whose parsing then fails (and is retried) still counts.
- Cached-input discounts are not modelled; the estimate is an upper bound
  for cached prompts.
- Request attribution is **separate** from the process-global budget guard.
  The guard (`SESSION_COST_USD >= MAX_COST_USD`) is a **known-cost guard,
  not a total-spend guard**: unpriced usage cannot contribute to its
  threshold. A one-time warning is logged per unpriced model, since the
  guard's accounting coverage is degraded while it is in use.
- `DEFAULT_MODEL` and `FALLBACK_MODEL` default to the same model, so the
  budget downgrade (`llm.fallback`) is not reachable under default
  configuration. This is a configuration fact, not a defect.
- `schema_version` stays `1`: the cost fields are additive optional metadata.

---

## 9. Metric dimension policy [implemented in M7.2]

Events answer "what happened during this request?"; metrics answer "what
pattern exists across requests?". Metrics are derived from events; they do
not replace them, and event metadata is **not** a bag of metric labels.

The executable policy is `src/observability/dimensions.py`; the table below is
generated from it and a test fails if the two drift. To regenerate:
`python -m src.observability.dimensions`.

Every key observed in an emitted event has exactly one classification **for
that event type** (the same key name can carry different domains in different
events, e.g. `guardrails.reason`, `retrieval.reason`, `llm.fallback.reason`):

- **dimension**: bounded, label-safe; the value must come from a closed set.
- **measure**: numeric; aggregated (count / sum / histogram), never a label.
- **correlation**: event-level only (`request_id`, `query_hash`, `timestamp`).
- **identity**: names the series (`event`, `step`, `schema_version`).
- **excluded**: must not reach metrics at all (`source_filter`).

Rules:
- **Unknown values become `other`.** A value outside a dimension's domain is
  bucketed to the literal `other`; it is never passed through or dropped.
- **`error_type` and `primary_error_type`** are dimension-safe only after
  allowlist normalization (known exception class, else `other`; `none` when
  absent). Raw exception class names must never be emitted as metric label
  values. A metric uses at most one error-class dimension per event.
- **`model`, `from_model`, `to_model`** are bounded to priced or configured
  models, else `other`.
- **Forbidden keys** (raw query, SQL, prompts/completions, error text,
  document content or IDs, source names, client identifiers) are rejected
  outright and can never be classified as anything.
- **Not dimensions today:** `lang` (emitted by no event) and request-level
  `intent` (not emitted on `request.completed`). Request latency is sliced by
  `route` only. Adding either field is an event-schema change, out of scope
  for M7.2.
- The policy defines **no metric names** and is not wired into `emit_event()`.
  Metric selection, aggregation and export are M7.3 / M8.
- Series upper bounds below are theoretical (every dimension of the event
  used together, error domain counted once); a metric need not use them all.
  Each must stay at or under the cap.

<!-- dimensions:start -->
| Event | Key | Class | Allowed values |
|---|---|---|---|
| `analytics.completed` | `chart_generated` | dimension | `false`, `true`; else `other` |
| `analytics.completed` | `confidence` | measure |  |
| `analytics.completed` | `error_type` | dimension | known exception classes, else `other` |
| `analytics.completed` | `intent` | dimension | `category_rank`, `generic`, `merchant_rank`, `none`, `timeseries`; else `other` |
| `analytics.completed` | `status` | dimension | `failure`, `success`; else `other` |
| `analytics.sql.completed` | `intent` | dimension | `category_rank`, `generic`, `merchant_rank`, `none`, `timeseries`; else `other` |
| `analytics.sql.completed` | `primary_error_type` | dimension | known exception classes, else `other` |
| `analytics.sql.completed` | `row_count` | measure |  |
| `analytics.sql.completed` | `status` | dimension | `success`; else `other` |
| `analytics.sql.completed` | `used_fallback_sql` | dimension | `false`, `true`; else `other` |
| `analytics.sql.failed` | `error_type` | dimension | known exception classes, else `other` |
| `analytics.sql.failed` | `intent` | dimension | `category_rank`, `generic`, `merchant_rank`, `none`, `timeseries`; else `other` |
| `analytics.sql.failed` | `primary_error_type` | dimension | known exception classes, else `other` |
| `analytics.sql.failed` | `status` | dimension | `failure`; else `other` |
| `analytics.sql.failed` | `used_fallback_sql` | dimension | `false`, `true`; else `other` |
| `guardrails.blocked` | `blocked` | dimension | `false`, `true`; else `other` |
| `guardrails.blocked` | `query_hash` | correlation |  |
| `guardrails.blocked` | `query_length` | measure |  |
| `guardrails.blocked` | `reason` | dimension | `injection`, `noise`, `none`, `out_of_domain`, `too_short`; else `other` |
| `guardrails.blocked` | `status` | dimension | `blocked`; else `other` |
| `guardrails.completed` | `blocked` | dimension | `false`, `true`; else `other` |
| `guardrails.completed` | `query_hash` | correlation |  |
| `guardrails.completed` | `query_length` | measure |  |
| `guardrails.completed` | `reason` | dimension | `injection`, `noise`, `none`, `out_of_domain`, `too_short`; else `other` |
| `guardrails.completed` | `status` | dimension | `success`; else `other` |
| `intent.completed` | `confidence` | measure |  |
| `intent.completed` | `intent` | dimension | `analytics`, `rag`, `reject`; else `other` |
| `intent.completed` | `method` | dimension | `heuristic`, `llm`; else `other` |
| `intent.completed` | `route` | dimension | `analytics`, `rag`, `reject`; else `other` |
| `intent.completed` | `status` | dimension | `success`; else `other` |
| `intent.failed` | `error_type` | dimension | known exception classes, else `other` |
| `intent.failed` | `status` | dimension | `failure`; else `other` |
| `language_detection.completed` | `status` | dimension | `success`; else `other` |
| `language_detection.failed` | `status` | dimension | `failure`; else `other` |
| `llm.completed` | `completion_tokens` | measure |  |
| `llm.completed` | `estimated_cost_usd` | measure |  |
| `llm.completed` | `model` | dimension | priced or configured models, else `other` |
| `llm.completed` | `prompt_tokens` | measure |  |
| `llm.completed` | `purpose` | dimension | `analytics_nl_to_sql`, `analytics_summary`, `intent_classification`, `language_detection`, `llm_rerank`, `query_rewrite`, `rag_answer`, `rag_insight`, `translation`; else `other` |
| `llm.completed` | `retry_count` | measure |  |
| `llm.completed` | `status` | dimension | `success`; else `other` |
| `llm.completed` | `total_tokens` | measure |  |
| `llm.failed` | `error_type` | dimension | known exception classes, else `other` |
| `llm.failed` | `model` | dimension | priced or configured models, else `other` |
| `llm.failed` | `purpose` | dimension | `analytics_nl_to_sql`, `analytics_summary`, `intent_classification`, `language_detection`, `llm_rerank`, `query_rewrite`, `rag_answer`, `rag_insight`, `translation`; else `other` |
| `llm.failed` | `retry_count` | measure |  |
| `llm.failed` | `status` | dimension | `failure`; else `other` |
| `llm.fallback` | `cumulative_session_cost_usd` | measure |  |
| `llm.fallback` | `from_model` | dimension | priced or configured models, else `other` |
| `llm.fallback` | `purpose` | dimension | `analytics_nl_to_sql`, `analytics_summary`, `intent_classification`, `language_detection`, `llm_rerank`, `query_rewrite`, `rag_answer`, `rag_insight`, `translation`; else `other` |
| `llm.fallback` | `reason` | dimension | `budget_threshold`; else `other` |
| `llm.fallback` | `status` | dimension | `success`; else `other` |
| `llm.fallback` | `to_model` | dimension | priced or configured models, else `other` |
| `ranking.completed` | `candidate_count` | measure |  |
| `ranking.completed` | `reranker` | dimension | `hybrid`; else `other` |
| `ranking.completed` | `selected_count` | measure |  |
| `ranking.completed` | `status` | dimension | `success`; else `other` |
| `ranking.failed` | `candidate_count` | measure |  |
| `ranking.failed` | `error_type` | dimension | known exception classes, else `other` |
| `ranking.failed` | `status` | dimension | `failure`; else `other` |
| `ranking.skipped` | `candidate_count` | measure |  |
| `ranking.skipped` | `selected_count` | measure |  |
| `ranking.skipped` | `status` | dimension | `skipped`; else `other` |
| `rate_limit.blocked` | `route` | dimension | `/analytics`, `/query`, `/rag`, `other`; else `other` |
| `rate_limit.blocked` | `status` | dimension | `blocked`; else `other` |
| `request.completed` | `cost_status` | dimension | `complete`, `not_applicable`, `partial`, `unknown`; else `other` |
| `request.completed` | `cost_usd_total` | measure |  |
| `request.completed` | `error_type` | dimension | known exception classes, else `other` |
| `request.completed` | `route` | dimension | `/analytics`, `/query`, `/rag`; else `other` |
| `request.completed` | `status` | dimension | `blocked`, `error`, `success`; else `other` |
| `request.started` | `route` | dimension | `/analytics`, `/query`, `/rag`; else `other` |
| `request.started` | `status` | dimension | `success`; else `other` |
| `retrieval.completed` | `candidate_count` | measure |  |
| `retrieval.completed` | `retrieval_method` | dimension | `vector_rpc`; else `other` |
| `retrieval.completed` | `source_filter` | excluded |  |
| `retrieval.completed` | `status` | dimension | `success`; else `other` |
| `retrieval.failed` | `error_type` | dimension | known exception classes, else `other` |
| `retrieval.failed` | `retrieval_method` | dimension | `vector_rpc`; else `other` |
| `retrieval.failed` | `source_filter` | excluded |  |
| `retrieval.failed` | `status` | dimension | `failure`; else `other` |
| `retrieval.skipped` | `candidate_count` | measure |  |
| `retrieval.skipped` | `reason` | dimension | `no_embedding`, `retriever_disabled`; else `other` |
| `retrieval.skipped` | `retrieval_method` | dimension | `vector_rpc`; else `other` |
| `retrieval.skipped` | `selected_count` | measure |  |
| `retrieval.skipped` | `source_filter` | excluded |  |
| `retrieval.skipped` | `status` | dimension | `skipped`; else `other` |

| Event | Series upper bound |
|---|---|
| `analytics.completed` | 864 |
| `analytics.sql.completed` | 576 |
| `analytics.sql.failed` | 576 |
| `guardrails.blocked` | 36 |
| `guardrails.completed` | 36 |
| `intent.completed` | 96 |
| `intent.failed` | 32 |
| `language_detection.completed` | 2 |
| `language_detection.failed` | 2 |
| `llm.completed` | 100 |
| `llm.failed` | 1600 |
| `llm.fallback` | 1000 |
| `ranking.completed` | 4 |
| `ranking.failed` | 32 |
| `ranking.skipped` | 2 |
| `rate_limit.blocked` | 8 |
| `request.completed` | 1280 |
| `request.started` | 8 |
| `retrieval.completed` | 4 |
| `retrieval.failed` | 64 |
| `retrieval.skipped` | 12 |

Cardinality cap per event: 2000.
<!-- dimensions:end -->

Operational questions the policy must support: request error and block rate
and latency by `route`; LLM failure, retry and fallback rate, tokens and cost
by `purpose` and `model`; retrieval success, skip and failure rate;
ranking failure rate; analytics SQL fallback and failure rate; and how many
LLM calls each `purpose` makes per request (answered in §13 by `llm_calls_per_request`).

Derived, not re-emitted: latency percentiles come from
`request.completed.duration_ms`; rates come from terminal event counts.

---

## 10. Post-M6 changes reconciled

Applied after the M6 freeze (`513f4b4`):

- **`merchant_inference_mode` removed.** Merchant/category ranking questions
  are answered by analytics SQL. The `merchant_inference` retrieval method
  and LLM purpose no longer exist. If SQL cannot answer, the system says so
  rather than falling back to a document-wide LLM read (see
  `docs/failure_modes.md` §4.3).
- **Analytics responses include `sql`** (the exact executed statement).
  Excluded from telemetry (§6).
- **Retriever exposes `retrieval_status`** (`ok|empty|unavailable`) in the
  RAG result. No new event; the `retrieval.*` events already distinguish the
  cases.
- **`details=str(e)` leak fixed.** Analytics error responses and the
  orchestrator's `error` field no longer carry exception text; unsafe-SQL
  rejection no longer echoes the SQL.
- **LLM exhaustion and request-scoped cost prerequisites** are fixed (§5).

---

## 11. Observed M6 traces — test/mock traces, not production traffic

These were captured from M6 runs against mocked/test dependencies. They are
condensed: only event names and key metadata are shown. Request IDs, token
counts and cost values are omitted here rather than reconstructed.

**RAG `/query`** (request `duration_ms` = 4):

```text
request.started
  llm.completed        purpose=language_detection
  guardrails.completed
  llm.completed        purpose=language_detection
  language_detection.completed
  intent.completed     intent=rag confidence=0.90 method=heuristic
  llm.completed        purpose=language_detection
  retrieval.completed  candidate_count=2
  ranking.completed    candidate_count=2 selected_count=2
  llm.completed        purpose=rag_answer
  llm.completed        purpose=rag_insight
request.completed
```

**Analytics `/query`** (request `duration_ms` = 9):

```text
request.started
  guardrails.completed
  language_detection.completed
  intent.completed     intent=analytics confidence=0.95 method=heuristic
  analytics.sql.completed  intent=timeseries used_fallback_sql=false row_count=3   (3 ms)
  analytics.completed      confidence=0.8602 chart_generated=true                  (6 ms)
request.completed
```

Observations (descriptive, not contract):
- The RAG trace contains **three** `llm.completed` events with
  `purpose=language_detection` but only **one** `language_detection.completed`.
  The LLM-call events and the stage event measure different things: the
  stage event is the orchestrator's language-detection stage; the LLM events
  are every call made for that purpose.
- The analytics trace shows the intended hierarchy (intent → sql →
  analytics) with `analytics.sql` nested inside `analytics`. Durations of
  children are not additive and must not be asserted as a contract.

### Finding: repeated language detection

`detect_language()` is called three times along the `/query` path:

- `src/safety/guardrails.py::validate_query`
- `src/orchestrator.py::run_query` (ignores the `detected_lang` already
  passed in by the router)
- `src/rag/question_rewrite.py::process_query`

Each call tries a keyword heuristic first and makes an LLM call
(`purpose=language_detection`) **only when no keyword matches**. The traces in
this section show three LLM calls because their text matched no keyword.

This is recorded as an **observability finding**. M7 measures it
(`llm_calls_per_request{purpose="language_detection"}`, §13); consolidation
is outside M7 and is not part of its scope.

### Finding: guardrails can call the LLM before rejecting

`validate_query` calls `detect_language()` **before** the injection and domain
checks. A query that matches none of the keywords (for example an injection
string, gibberish, or non-English text) therefore costs an LLM call and its
retry backoff before being rejected, even though the module describes the
guardrails as deterministic. This was observed against a live server: a
blocked `/query` took about 12 s (four failed attempts, no API key
configured). With a key, each such request is a billable call. The per-IP rate
limiter bounds it but does not remove it.

Recorded here as an observed behaviour only. Changing it alters guardrail
runtime behaviour, cost and latency, so it needs its own design pass and is
**not** part of M7 or M8.

---

## 12. Non-goals

Never in scope: OpenTelemetry or any vendor SDK in application code, an event
database or broker, dashboards, alerting, sampling or backpressure for the
event stream, persistence of signal state, and the language-detection and
guardrail refactors recorded in §11.

What changed from the M2/M7.0 wording: a Prometheus *text exposition* (M8.3) is
implemented, hand-rendered with no `prometheus_client`. The application still
never calls a vendor SDK; an external scraper pulls from `/metrics`.

---

## 13. Operational signals [implemented in M7.3]

`src/observability/signals.py` turns the event stream into operational
signals:

```text
event dicts -> Aggregator.observe() -> in-memory state -> snapshot()
```

It is a **pure, replay-only** aggregator. It is not attached to
`emit_event()`, has no global state or collector, and does no export or
time-windowing; live wiring, vendor integration and rates over time are M8.
The CLI is a replay/debug tool only:

```text
python -m src.observability.signals events.jsonl     # or stdin
```

Rules:
- **Label traceability.** Every metric label is the envelope `status` or a
  dimension-classified field of its source event (§9). Values go through
  `dimension_value()`, so out-of-domain values become `other`. There are no
  derived label mappings; `validate_specs()` enforces this.
- **Absence is not zero.** Only observed series appear. An optional
  dimension an event does not carry is omitted from the series, never filled
  with a synthetic `none`. (A key present with a null value is a real emitted
  value and normalizes to `none`.) Ratios are `null` when the denominator is
  0.
- **`request_id` is a grouping key only**, never a label.
- **Unclassified input** (unknown event, unknown or forbidden key, malformed
  input) is counted in `signals_unclassified_total` and skipped; it never
  crashes the aggregator. `strict=True` raises, for tests.
- **Cumulative only.** No time windows. Instrumented application request volume is
  `requests_total + rate_limited_total` (§14); it is still not every possible
  HTTP request.
- **`llm.fallback` is a terminal event for the downgrade, not an LLM call
  outcome.** `llm_fallbacks_total` counts it from its own event;
  `llm_calls_total` counts only `llm.completed` / `llm.failed`.

| Metric | Kind | Labels |
|---|---|---|
| `requests_total` | counter | `route`, `status` |
| `request_duration_ms` | histogram | `route`, `status` |
| `request_cost_usd_total` | sum | `route`, `cost_status` |
| `rate_limited_total` | counter (`rate_limit.blocked`) | `route` |
| `llm_calls_total` | counter | `purpose`, `model`, `status` |
| `llm_duration_ms` | histogram | `purpose` |
| `llm_retries_total` | sum of `retry_count` | `purpose` |
| `llm_retried_calls_total` | counter (completed, `retry_count` > 0) | `purpose` |
| `llm_failures_total` | counter | `purpose`, `error_type` |
| `llm_fallbacks_total` | counter | `purpose`, `reason` |
| `llm_prompt_tokens_total`, `llm_completion_tokens_total` | sum | `purpose`, `model` |
| `llm_cost_usd_total` | sum (non-null costs) | `purpose`, `model` |
| `llm_cost_unknown_total` | counter (null cost) | `purpose`, `model` |
| `retrieval_total` | counter | `status`, `retrieval_method`, `reason` (skipped only) |
| `retrieval_empty_total` | counter (completed, `candidate_count` = 0) | `retrieval_method` |
| `retrieval_failures_total` | counter | `error_type` |
| `ranking_total` | counter | `status`, `reranker` (completed only) |
| `analytics_total` | counter | `status`, `intent` |
| `analytics_sql_total` | counter | `status`, `intent`, `used_fallback_sql` |
| `analytics_failures_total` | counter (analytics failure) | `error_type` |
| `intent_total` | counter | `status`, `intent`, `method` |
| `guardrails_total` | counter | `status`, `reason` |
| `language_detection_total` | counter | `status` |
| `llm_calls_per_request` | histogram (per-request derived) | `purpose` |
| `signals_unclassified_total`, `signals_dropped_requests_total` | counter (meta; `0` is a known value) | none |

Sums report `{sum, count, null_count}`: a null value (unknown cost) is
counted in `null_count` and never added as `0`.

**Per-request signal.** `llm_calls_per_request{purpose}` is the number of LLM
calls (`llm.completed` + `llm.failed`) a request made for one purpose,
observed once per request per purpose. State is accumulated by `request_id`
while the request is in flight and flushed on `request.completed`, after
which it is removed. In-flight state is capped (10,000 requests); eviction is
counted in `signals_dropped_requests_total`. Events with `request_id` `-`
do not participate. This is the signal that makes the repeated
language-detection finding (§11) queryable.

**Histograms.** Fixed buckets in ms: 1, 5, 10, 25, 50, 100, 250, 500, 1000,
2500, 5000, 10000, plus an implicit `+Inf` bucket so slower observations are
kept. A null `duration_ms` is skipped, never treated as 0. Percentiles
(`p50`, `p95`, `p99`) are **bucket upper-bound estimates**, not exact values;
a percentile landing in the overflow bucket reports `+Inf`.

**Rates** (cumulative, whole-snapshot ratios; `null` when undefined):
`request_error_rate`, `request_block_rate`, `request_rate_limited_ratio`
(`rate_limited_total / (requests_total + rate_limited_total)`; `null` when both
are 0), `guardrail_block_rate`,
`llm_failure_rate`, `llm_retry_rate` (retried completed calls / completed
calls), `llm_fallback_rate` (fallbacks / LLM calls), `retrieval_empty_rate`
(empty / successful retrievals), `retrieval_skip_rate`,
`retrieval_failure_rate`, `ranking_failure_rate`, `analytics_failure_rate`,
`analytics_sql_fallback_rate`.

---

## 14. Exposure design (M8) [M8.1-M8.4 implemented]

M8 exposes the M7.3 signals without coupling to a vendor. Target: **pull-based,
in-process** signals; JSONL replay (§13 CLI) remains the offline path. Push to
a vendor backend, OpenTelemetry, dashboards and alerting are out of M8.

```text
emit_event()
    |-- JSONL StreamHandler                 (unchanged)
    '-- optional SignalsHandler  [M8.1]
              |
              v
        process Aggregator  (lock around mutation, in the adapter)
              |
              v
          snapshot()
              |-- M8.2  authenticated JSON endpoint
              '-- M8.3  Prometheus text exposition  -> external scraper
```

**M8.1: live feed adapter [implemented]** (`src/observability/live.py`).
- Consumes only events emitted through the `observability.events` logger;
  `events.py` knows nothing about it.
- The pure `Aggregator` stays unaware of logging, threads, HTTP, environment
  variables and vendors; the lock and process-level ownership live in the
  adapter.
- **Opt-in and off by default**: `SIGNALS_ENABLED=true` enables it (read when
  the app is created). Disabled means no handler and no aggregator
  instantiated: no collection overhead.
- **Fail-open**: the handler swallows every exception (including logging's
  own error path) and counts it in `meta.handler_errors`; non-JSON records on
  the events logger are counted in `meta.ignored_records`. A telemetry
  failure never becomes an application failure.
- **State is process-local and non-durable.** A restart or spin-down (the
  Render free tier spins down when idle) starts a fresh aggregator;
  `meta.started_at_unix` records when, so a consumer can detect the reset.
  Counters are cumulative since then. Scaling out gives per-instance state.
- No endpoint exists yet (M8.2).

**M8.2: JSON snapshot endpoint [implemented]** (`api/signals.py`).
`GET /signals` with `Authorization: Bearer <SIGNALS_TOKEN>` returns the M7.3
snapshot (`metrics`, `rates`, `meta`, including rates and percentiles as
debugging conveniences).
- **Registered only when it can serve**: the live feed is enabled
  (`SIGNALS_ENABLED=true`) **and** `SIGNALS_TOKEN` is non-empty. Otherwise the
  route does not exist: plain `404`, and `/signals` is absent from the OpenAPI
  schema. Enabled without a token fails closed (a warning is logged at
  startup); there is no "no token means open" mode. The handler re-checks both
  conditions at request time.
- Enabled and credentials missing, malformed, or wrong -> `401`, body
  `{"error": "Unauthorized"}`, `WWW-Authenticate: Bearer`. The scheme is
  case-insensitive. There is no redacted public variant.
- The token is compared in constant time (`hmac.compare_digest`) and never
  appears in a response, a log line, or an event; a presented wrong
  credential is never echoed.
- A snapshot failure returns a generic `503` with no internal detail.
- Responses carry `Cache-Control: no-store`.
- **The endpoint does not observe itself.** The handler emits no request
  events and sets no `request_id`, so `/signals` calls never contribute to
  `requests_total`, request latency, or `events_observed`. This is an explicit,
  tested rule: a frequent scraper must not dominate the request signals.
- The existing rate limiter and CORS middleware apply as for every route (the
  default is 20 requests per minute per client; size a scrape interval
  accordingly).
- Only bounded dimensions and measures are exposed (no `request_id`,
  `query_hash`, or content). `meta.started_at_unix`, `meta.handler_errors`,
  `meta.ignored_records`, `meta.events_observed` and `meta.in_flight_requests`
  are plain numbers, never labels or dimensions.

**M8.3: Prometheus text exposition [implemented]**
(`src/observability/exposition.py`, `GET /metrics` in `api/signals.py`).
M7.3 owns the analytical snapshot; M8.3 owns only a representation of it. The
exporter is a pure function (`render_prometheus(snapshot) -> str`): no I/O, no
environment, no threads, no `prometheus_client`, no global registry, and it
never touches the aggregator.
- **Endpoint:** `GET /metrics` is separate from `/signals` (which stays the
  JSON snapshot). It has the identical gate and access rules: registered only
  when the feed is enabled and `SIGNALS_TOKEN` is set (else 404 and absent
  from OpenAPI), bearer auth with 401, constant-time comparison, generic 503,
  `Cache-Control: no-store`, no token leakage, self-exclusion (scraping never
  contributes to the signals), and the existing rate limiter and CORS.
  `Content-Type: text/plain; version=0.0.4; charset=utf-8`.
- **Lock boundary:** the endpoint takes one snapshot (the aggregator lock is
  held only inside `snapshot()`) and renders the text **outside** that lock;
  sorting, cumulative conversion and string building never run inside the
  collection critical section.
- **Format:** Prometheus text 0.0.4 only; no OpenMetrics, no content
  negotiation. Every family has `# HELP` and `# TYPE`; names and label names
  are validated; label values escape `\`, `"` and newline; `le` is reserved
  (no policy dimension may be named `le`); output is deterministic (families
  sorted by name, series sorted, label names sorted, `le` last).
- **Mapping:**

| Snapshot | Exposition |
|---|---|
| counter `x_total` | `counter` |
| sum `foo_total` `{sum,count,null_count}` | `foo_total` (the sum), `foo_observations_total` (non-null count), `foo_unknown_total` (null count), all `counter` |
| histogram `x` | `histogram`: **cumulative** `x_bucket{...,le="..."}` including `+Inf`, then `x_sum`, `x_count` (the aggregator stores non-cumulative counts; conversion is the exporter's job) |
| `llm_calls_per_request` | histogram labelled by `purpose` |
| `signals_unclassified_total`, `signals_dropped_requests_total` | `counter` |
| `meta.events_observed`, `handler_errors`, `ignored_records` | `signals_events_observed_total`, `signals_handler_errors_total`, `signals_ignored_records_total` (`counter`) |
| `meta.in_flight_requests`, `started_at_unix` | `signals_in_flight_requests`, `signals_start_time_seconds` (`gauge`) |
| `rates`, `p50`/`p95`/`p99` | **not exported**: consumers derive them (`rate()`, `histogram_quantile()`) |

- Histogram bounds render as stored (`1`, `5`, ..., `10000`, `+Inf`).
- The sum expansion is generic: `llm_cost_usd_unknown_total` and the separate
  M7.3 counter `llm_cost_unknown_total` carry overlapping data under different
  names. That overlap is accepted rather than special-cased; a test asserts no
  two exposition families share a name.
- **Absence is not zero:** a family with no observed series is omitted
  entirely; nothing is initialized merely because it exists in the inventory.
  Only the meta families (known process-local semantics, zero included) are
  always present; `signals_handler_errors_total`,
  `signals_ignored_records_total` and `signals_start_time_seconds` come from
  the live collector and are absent from a plain replay snapshot.
- Duration metrics keep `_ms` names and values: an intentional deviation from
  the seconds convention (`signals_start_time_seconds` is a Unix timestamp). A
  seconds view would be an explicit mapping decision, never a silent unit
  change.
- Every metric in the inventory must have a static one-line `HELP` text; a
  test fails if one is added without it. The invariant that proves the
  boundary is round-trip conservation: runtime events -> M7.3 snapshot ->
  Prometheus text -> parsed text equals the snapshot's totals.
- **Reset semantics:** counters are process-local and start at zero after a
  restart or spin-down; `signals_start_time_seconds` changes when that
  happens, and a scraper's counter-reset handling (`rate()` / `increase()`)
  covers the rest.

Example configuration only (not a requirement imposed by the application; the
limiter allows 20 requests/minute per client by default, so a 30 s interval
leaves ample headroom):

```yaml
scrape_configs:
  - job_name: ai-fraud-agent
    scheme: https
    metrics_path: /metrics
    scrape_interval: 30s
    authorization:
      type: Bearer
      credentials: <SIGNALS_TOKEN>
    static_configs:
      - targets: ["<service-host>"]
```

Example queries only (derived by the consumer, not shipped as rules):

```text
# request error ratio over 5 minutes
sum(rate(requests_total{status="error"}[5m])) / sum(rate(requests_total[5m]))

# p95 request latency in ms
histogram_quantile(0.95, sum by (le) (rate(request_duration_ms_bucket[5m])))

# LLM calls one request makes for language detection (mean)
sum(rate(llm_calls_per_request_sum{purpose="language_detection"}[5m]))
  / sum(rate(llm_calls_per_request_count{purpose="language_detection"}[5m]))
```

**M8.4: `rate_limit.blocked` event [implemented]** (`src/safety/rate_limit.py`).
The middleware emits one bounded event for each request it rejects, before the
router runs.
- **Metadata is `route` only.** The raw path is mapped to a bounded value
  *before the event is built* (`/query`, `/rag`, `/analytics`, trailing slash
  tolerated; everything else `other`), so no event ever carries a raw path. No
  `count`, `limit`, IP, hash, or other client identifier.
- `status=blocked`, `duration_ms=null` (unmeasured, never a fake 0), and
  `request_id` is `-`: the middleware runs before any request context exists,
  so the per-request signal ignores it.
- **Control-plane exclusion:** `/signals` and `/metrics` emit no event (they
  are not application traffic; a throttled scraper must not inflate the
  `other` bucket). The 429 is unchanged and the throttle is still logged
  (`route=control_plane`).
- **Telemetry never changes the decision.** Recording a block has its own
  guard; if logging or emission fails the request is still answered 429 (the
  existing outer fail-open still applies to genuine limiter failures).
- **Privacy:** the raw client IP is no longer written to the rate-limit
  warning (it logs route and window count only), consistent with §6.
- Classified in the M7.2 policy (`route` and `status` as dimensions; the route
  domain is the three routes plus `other`). Signals: `rate_limited_total{route}`
  and `request_rate_limited_ratio` (§13); exposition: `rate_limited_total`
  counter with `HELP` (§14 M8.3).
- Volume semantics: `requests_total` = requests that reached the request
  lifecycle; `rate_limited_total` = requests the limiter rejected;
  instrumented application volume = their sum. That is still not every
  possible HTTP request to the server.

---

## 15. Operating notes [implemented]

How the M7/M8 signals behave when this service is actually run. Nothing here
is enforced by code beyond what §13 and §14 already describe.

**Enabling.** The feed and both endpoints are off by default and nothing in
`render.yaml` turns them on. To use them set, in the host's environment (on
Render: the service's environment settings, with the token as a secret and
never committed):

| Variable | Purpose |
|---|---|
| `SIGNALS_ENABLED=true` | start the live feed |
| `SIGNALS_TOKEN=<secret>` | bearer token; without it `/signals` and `/metrics` stay closed (404) |
| `TRUST_FORWARDED_FOR=true` | already set in `render.yaml`; per-client rate limiting behind Render's proxy |
| `RATE_LIMIT_PER_MINUTE` | default `20` per client IP, shared by every route including `/metrics` |

Verify with `GET /health`, then `GET /metrics` with
`Authorization: Bearer <token>`. A 404 means the feed is disabled or the token
is unset; 401 means the credential is wrong.

**One process.** The aggregator is per process. The Docker image runs a single
uvicorn worker. With more than one worker (or instance) each has its own
counters, and a scrape reaches an arbitrary one; do not run multiple workers
and trust the totals.

**Counters reset.** State is in memory only. A restart or spin-down resets
every counter to zero; `signals_start_time_seconds` changes when it happens,
and a scraper's counter-reset handling covers it. Nothing is persisted.

**Scraping and the free tier.** A Render free-tier web service spins down when
idle. A scraper polling `/metrics` (for example every 30 s) counts as traffic,
so it keeps the instance awake and consumes the monthly free instance hours.
Choose the interval, or scrape only on demand, with that in mind. The scraper
also shares the per-IP rate-limit window with any other traffic from the same
address.

**Volume and cost caveats.**
- `requests_total + rate_limited_total` is the instrumented application
  volume, not every possible HTTP request (§14, M8.4).
- A blocked-request flood emits one `rate_limit.blocked` event per blocked
  request. Exact counting was chosen over sampling; sampling would be a
  separate design.
- Each emitted event is also parsed by the live handler (about 13 us per event
  in a local benchmark; not a guarantee on other hardware).
- Guardrails can make an LLM call before rejecting a query (§11).

**Privacy.** Events and signals carry bounded dimensions only; the client IP is
never emitted or logged by the rate limiter (§6).

---

## Relationship to `docs/observability.md`

That document is aspirational and predates M3–M6 (it describes `query_id`
and cost/latency tracking that this contract and the M3–M6 code now define
precisely). `request_id` is the canonical field name. Rewriting
`docs/observability.md` to match reality is deferred to the documentation
milestone.
