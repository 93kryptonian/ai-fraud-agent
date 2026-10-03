# Observability Contract (reconciled after M6)

Status: **reconciled after M6.** M6 is implemented and frozen at `513f4b4`.
Sections marked **[implemented]** describe what the code emits today.
Sections marked **[design-only]** describe M7 design direction; nothing in
them is implemented yet.

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

## 2. Request-level record [implemented, narrower than M2 intended]

One `request.completed` event per request, emitted by the router handler
(exactly one, including on an unhandled exception).

| Field | Where | Notes |
|---|---|---|
| `request_id` | envelope | uuid4, assigned in `api/routers.py` before `validate_query()`, so even a guardrail rejection is correlatable. |
| `status` | envelope | Request-level: `success`, `blocked`, `error`. `rate_limited` is **not** emitted (§4). `success` means the handler completed normally; it does **not** assert the business operation succeeded — a pipeline that catches its own exception and returns an `error` dict still yields `success` here. |
| `duration_ms` | envelope | Real handler duration, integer ms. |
| `metadata.route` | metadata | `/query`, `/rag`, `/analytics`. |
| `metadata.error_type` | metadata | Present only when `status == "error"`. Exception class name only. |

Fields the M2 contract listed for this record that are **not emitted today**:
`intent`, `lang`, `started_at`/`completed_at`, `error` (message),
`cost_usd_total`, `fallback_used`, `fallback_reason`. The `intent` is
available from `intent.completed`. Request-level cost is M7 (§8).

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
| `request.completed` | `api/routers.py` | see §2 |

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
   estimated cost. It is **not yet exposed** in any event (§8).
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

## 8. M7 cost contract [design-only, not yet implemented]

Request-level cost is not emitted today. M7.1 will add it to
`request.completed`.

```text
cost_status:
  not_applicable   no LLM call occurred in the request
  complete         every LLM call has attributable cost
  partial          at least one call's cost is unknown, others known
  unknown          an LLM call occurred and no cost is attributable

cost_usd_total:
  0.0              when not_applicable
  known total      when complete
  known partial    when partial (a lower bound)
  null             when unknown
```

Rules:
- **Unknown is never `$0.00`.** Missing provider usage and an unpriced model
  are both "unknown". Today `estimate_cost()` returns `0.0` for an unknown
  model; M7.1 must change that.
- Input and output tokens are priced distinctly in M7.1. Today a single
  per-1K price is applied to the sum of both.
- Fallback-model calls count toward request cost (both models' costs are
  summed).
- A request that errors or hits the low-confidence fallback still reports
  the cost of the LLM calls that already completed.
- Attribution for failed or retried calls (billable attempts that produced
  no usable response) needs explicit semantics in M7.1; until then a call
  that exhausted retries adds no cost.
- Request attribution is **separate** from the process-global budget guard.
  `SESSION_COST_USD >= MAX_COST_USD` (downgrade policy across many
  requests) must not be merged with per-request cost.

---

## 9. M7 metric cardinality policy [design-only, not yet implemented]

Metrics are derived from events; they do not replace them. Every metric
dimension must be bounded and non-sensitive.

Allowed dimensions (bounded):

```text
route, status, intent, lang
llm: purpose, model, outcome
fallback: reason
retrieval: retrieval_method, outcome
ranking: reranker, outcome
analytics: intent (timeseries|merchant_rank|category_rank|generic),
           used_fallback_sql, outcome
```

Prohibited as metric dimensions (high-cardinality or sensitive; fine as
event-level correlation fields where they already exist, never as labels):

```text
request_id, query_hash, raw query, SQL, error text, prompt/completion,
document content or IDs, source names, client identifiers, timestamps
```

Operational questions M7 must be able to answer: request error and block
rate and latency; LLM failure, retry and fallback rate, tokens and cost;
retrieval success, skip, failure and empty-result rate; ranking failure
rate; analytics SQL fallback and failure rate; and how many LLM calls each
`purpose` makes per request.

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
(`llm_calls_total{purpose="language_detection"}` per request); consolidation
is outside M7.0 and is not part of M7's scope.

---

## 12. Non-goals

For M7.0 and M7 generally: no Prometheus implementation, no OpenTelemetry,
no Grafana or dashboards, no event database or broker, no alerting, and no
language-detection refactor. A later exposure/export layer (M8) translates
*from* these events; the application never calls a vendor SDK directly.

---

## Relationship to `docs/observability.md`

That document is aspirational and predates M3–M6 (it describes `query_id`
and cost/latency tracking that this contract and the M3–M6 code now define
precisely). `request_id` is the canonical field name. Rewriting
`docs/observability.md` to match reality is deferred to the documentation
milestone.
