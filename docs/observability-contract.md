# Observability Contract (reconciled after M6, through M7.3)

Status: **reconciled after M6, updated for M7.1.** M6 is implemented and
frozen at `513f4b4`. Sections marked **[implemented]** describe what the code
emits today (M7.1 request-level cost is implemented, §8). Sections marked
**[design-only]** describe M7 design direction; nothing in them is
implemented yet. (M7.2 dimension policy, §9, M7.3 operational signals, §13, the M8.1 live feed and the M8.2 `/signals` endpoint, §14, are implemented.)

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
router runs. They get no `request_id` and emit no events today (see §4).

---

## 2. Request-level record [implemented, narrower than M2 intended; cost added in M7.1]

One `request.completed` event per request, emitted by the router handler
(exactly one, including on an unhandled exception).

| Field | Where | Notes |
|---|---|---|
| `request_id` | envelope | uuid4, assigned in `api/routers.py` before `validate_query()`, so even a guardrail rejection is correlatable. |
| `status` | envelope | Request-level: `success`, `blocked`, `error`. `rate_limited` is **not** emitted (§4). `success` means the handler completed normally; it does **not** assert the business operation succeeded — a pipeline that catches its own exception and returns an `error` dict still yields `success` here. |
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
| `rate_limit.blocked` | The middleware returns 429 and writes a log line only. |

Re-adding any of these is an explicit decision, not an assumed catalog entry.

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
6. **Rate-limited requests are invisible** to the event stream (§4).

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
- Client IP is held in memory by the rate limiter only and is never emitted.
  If rate-limit events are ever added, hash or truncate the IP first.

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

`detect_language()` currently executes three times along the `/query` path:

- `src/safety/guardrails.py::validate_query`
- `src/orchestrator.py::run_query` (ignores the `detected_lang` already
  passed in by the router)
- `src/rag/question_rewrite.py::process_query`

This is recorded as an **observability finding**. M7 measures it
(`llm_calls_per_request{purpose="language_detection"}`, §13); consolidation
is outside M7 and is not part of its scope.

---

## 12. Non-goals

For M7.0 and M7 generally: no Prometheus implementation, no OpenTelemetry,
no Grafana or dashboards, no event database or broker, no alerting, and no
language-detection refactor. A later exposure/export layer (M8) translates
*from* these events; the application never calls a vendor SDK directly.

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
- **Cumulative only.** No time windows. Request volume is not HTTP volume
  (rate-limited requests emit no events, §5).
- **`llm.fallback` is a terminal event for the downgrade, not an LLM call
  outcome.** `llm_fallbacks_total` counts it from its own event;
  `llm_calls_total` counts only `llm.completed` / `llm.failed`.

| Metric | Kind | Labels |
|---|---|---|
| `requests_total` | counter | `route`, `status` |
| `request_duration_ms` | histogram | `route`, `status` |
| `request_cost_usd_total` | sum | `route`, `cost_status` |
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
`request_error_rate`, `request_block_rate`, `guardrail_block_rate`,
`llm_failure_rate`, `llm_retry_rate` (retried completed calls / completed
calls), `llm_fallback_rate` (fallbacks / LLM calls), `retrieval_empty_rate`
(empty / successful retrievals), `retrieval_skip_rate`,
`retrieval_failure_rate`, `ranking_failure_rate`, `analytics_failure_rate`,
`analytics_sql_fallback_rate`.

---

## 14. Exposure design (M8) [M8.1 and M8.2 implemented; M8.3-M8.4 design-locked, not implemented]

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

**M8.3: Prometheus text exposition [design-locked].** A separate mapping from
the analytical snapshot; the exposition format is not the internal model.
- Hand-rendered; no `prometheus_client`, no global registry.
- Must emit `# HELP` / `# TYPE`, cumulative histogram buckets with `+Inf`,
  `_sum` and `_count`, label-value escaping (`\`, `"`, newline), valid metric
  and label names, and deterministic ordering.
- Counters map to counters; a sum becomes a monotonic value plus separate
  observation and unknown counters; histograms become cumulative buckets
  (the aggregator stores non-cumulative counts).
- Rates and percentiles are **not** exported; consumers derive them from the
  counters. A start-time metric supports reset detection.
- Duration metrics keep their `_ms` names and values. This intentionally
  deviates from the seconds convention; any seconds representation would be an
  explicit mapping decision, never an accidental unit change.

**M8.4: `rate_limit.blocked` event [design-locked].** The middleware emits a
bounded event for requests rejected before the router.
- Labelled by `route` only: `/query`, `/rag`, `/analytics`, anything else
  `other`. No raw path is ever a label. It has no `request_id`.
- Requires classifying the event in the M7.2 policy and updating the runtime
  completeness test.
- Afterwards, "instrumented application request volume" is approximately
  `request.completed` plus `rate_limit.blocked`; it is still not every
  possible HTTP request to the server.

---

## Relationship to `docs/observability.md`

That document is aspirational and predates M3–M6 (it describes `query_id`
and cost/latency tracking that this contract and the M3–M6 code now define
precisely). `request_id` is the canonical field name. Rewriting
`docs/observability.md` to match reality is deferred to the documentation
milestone.
