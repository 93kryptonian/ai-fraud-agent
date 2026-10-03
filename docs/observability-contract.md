# Observability Contract (M2)

Status: **contract only — nothing in this document is implemented yet.**
No code changes accompany this file. See "Relationship to `docs/observability.md`"
at the end for how this replaces that document once M3–M9 land.

This is Milestone 2 of the observability upgrade (M1 baseline freeze → **M2
this contract** → M3 request context → M4 structured events → M5 timing +
orchestrator instrumentation → M6 RAG/LLM/fallback instrumentation → M7 tests
→ M8 trace viewer + metrics → M9 docs).

The target capability, stated once, precisely:

> Given a `request_id`, reconstruct the full execution path of that request —
> every stage it passed through, in order, with latency, status, and enough
> metadata to explain *why* it produced the answer it did — without exposing
> the user's query text, document contents, or LLM prompt/completion text.

---

## 1. Scope: the real request lifecycle

This contract is written against the code as it actually exists, not an
idealized pipeline. There are **three entry points**, and they do not all go
through the same stages:

```
POST /query      → validate_query() → run_query()      [orchestrator: full pipeline]
POST /rag        → validate_query() → run_rag()          [RAG only, no intent routing]
POST /analytics  → validate_query() → run_analytics()     [analytics only]
```

`run_query()` (the only path with real branching) does, in order:

```
sanitize_input
  → detect_language
    → detect_intent (heuristic, then LLM if heuristic confidence < 0.80)
      → [reject]  → done
      → [analytics] → run_analytics(...)                → done
      → [rag]     → process_query (rewrite/translate)
                  → run_rag(...)  [retrieve_top_k → build_context → llm.run]
                  → score_answer
                  → [low-confidence fallback, if final_score < 0.12] → done
                  → generate_insight
                  → translate_en_to_id (if user_lang == "id")        → done
```

`run_rag()`'s `retrieve_top_k` call already does its own sub-pipeline
(embed query → `match_documents` RPC → `rerank_chunks`). (A second path,
`merchant_inference_mode`, existed historically and was removed; merchant
ranking questions are now answered by analytics SQL.)

`run_analytics()` does: `classify_analytics_intent` → `nl_to_sql` (template
or LLM-generated) → `execute_sql` → summarize → optional
`refine_summary_with_llm`.

**The LLM client (`llm.run`) is called from at least 8 different call sites**
(intent classification, language-detection fallback, translation ×2, query
rewrite, RAG answer generation, analytics NL→SQL,
analytics summary refinement, LLM reranking) — a single request can invoke it
multiple times. Every event schema below assumes this: `llm.completed` is a
repeatable event tagged with a `purpose`, not a once-per-request event.

---

## 2. Request-level record

One record per incoming HTTP request, emitted at `request.completed`:

| Field | Type | Notes |
|---|---|---|
| `request_id` | uuid4 string | Generated once, in `api/routers.py`, **before** `validate_query()` runs, so even a guardrail rejection gets a `request_id`. |
| `route` | `"query" \| "rag" \| "analytics"` | Which endpoint. |
| `intent` | `"rag" \| "analytics" \| "reject" \| null` | Only set on the `/query` route; null for `/rag` and `/analytics` (they don't route). |
| `lang` | `"en" \| "id"` | |
| `started_at`, `completed_at`, `duration_ms` | ISO8601, ISO8601, int | |
| `status` | `"success" \| "blocked" \| "rate_limited" \| "error"` | **Outcome only.** A request that hit the low-confidence fallback and still returned a usable answer is `status=success` — see `fallback_used`/`fallback_reason` below. Don't fold a business-logic path (fallback) into the same enum as a hard outcome (blocked/rate-limited/error); it makes "did this request succeed?" ambiguous to query later. |
| `error` | string \| null | Present when `status == "error"`. |
| `cost_usd_total` | float | Sum of this request's `llm.completed` events. **See §5 known gap** — today's cost tracking is process-global, not per-request; this field cannot be trusted until that's fixed. |
| `fallback_used` | bool | True if `llm.fallback` or the low-confidence fallback fired anywhere in this request, independent of `status`. |
| `fallback_reason` | `null \| "low_confidence" \| "budget_threshold"` | Which fallback fired, if any. Only these two exist in code today (orchestrator's `final_score < 0.12` branch, and `LLMClient`'s cost-threshold model downgrade) — not a placeholder for reasons that aren't implemented. |

---

## 3. Event envelope

Every stage emits **one event on success, one on failure**, never both,
never zero (a stage that starts must emit exactly one terminal event):

```json
{
  "schema_version": 1,
  "timestamp": "2026-09-26T18:24:00.123Z",
  "request_id": "8d9f2a41-...",
  "event": "retrieval.completed",
  "step": "retrieval",
  "status": "success",
  "duration_ms": 87,
  "metadata": { "...": "stage-specific, see §4" }
}
```

`schema_version` starts at `1` and only increments on a breaking change to
this envelope or an existing field's meaning (adding a new optional
`metadata` key is not breaking; removing/renaming a field, or changing what
an existing field means, is). This is what lets a future trace viewer or
metrics job know which shape it's reading without guessing.

`event` is always `"{step}.{outcome}"`. `status` is one of `success | failure
| blocked | skipped`. `skipped` covers stages that don't run for a given
request (e.g. `ranking.skipped` when retrieval returned zero candidates).
(Note: this is the *stage*-level `status` enum, distinct from the
*request*-level `status` enum in §2 — a stage can be `blocked` while the
request's own status is still `success`, e.g. guardrails never fired but one
LLM call inside the request retried and failed before a fallback recovered.)

---

## 4. Stage catalog

Grounded in the actual function that owns each stage, so an implementer
knows exactly where the instrumentation call goes.

| Event | Owning code | metadata |
|---|---|---|
| `request.started` | `api/routers.py`, before `validate_query` | `route` |
| `guardrails.completed` / `guardrails.blocked` | `src/safety/guardrails.py::validate_query` | `blocked` (bool), `reason` (`too_short\|noise\|injection\|out_of_domain\|null`), `query_length`, `query_hash` (see §6 — never raw text) |
| `language_detection.completed` | `src/rag/question_rewrite.py::detect_language` | `lang`, `method` (`heuristic\|llm_fallback`) |
| `intent.completed` | `src/orchestrator.py::detect_intent` | `intent`, `confidence`, `method` (`heuristic\|llm`), `route` |
| `retrieval.completed` / `retrieval.empty` | `src/rag/retriever_direct.py::retrieve_top_k` | `retrieval_method` (`vector_rpc\|disabled`), `candidate_count`, `selected_count`, `source_filter` |
| `ranking.completed` / `ranking.skipped` | `src/rag/ranking.py::rerank_chunks` | `candidate_count`, `selected_count`, `reranker` (`hybrid\|hybrid+llm`), `top_result_score`, `embeddings_available` (bool) |
| `llm.completed` / `llm.failed` | `src/llm/llm_client.py::LLMClient.run` | `purpose` (`intent_classification\|language_detection\|translation\|query_rewrite\|rag_answer\|analytics_nl_to_sql\|analytics_summary\|llm_rerank`), `model`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `estimated_cost_usd`, `retry_count`, `fallback_triggered` (bool) |
| `llm.fallback` | `LLMClient.run`, budget-downgrade branch | `from_model`, `to_model`, `reason` (`budget_threshold` — the only reason implemented today; `failure`/`timeout` are not distinguished by current code, see §5), `cumulative_session_cost_usd` |
| `analytics.sql_executed` | `src/analytics/fraud_analytics.py::execute_sql` | `template` (`merchant_rank\|category_rank\|timeseries\|llm_generated`), `row_count`, `truncated` (bool) |
| `analytics.completed` | `run_analytics` return | `intent`, `confidence`, `chart_generated` (bool) |
| `scoring.completed` | `src/llm/scoring.py::score_answer` | `final_score`, `gate` (`heuristic_early_exit\|full_ensemble`), `used_llm_judge` (bool) |
| `fallback.low_confidence` | `src/orchestrator.py`, `final_score < 0.12` branch | `score`, `threshold` |
| `rate_limit.blocked` | `src/safety/rate_limit.py::RateLimitMiddleware` | `client_id_hash`, `count`, `limit` |
| `request.completed` | end of the router handler | `route`, `intent`, `status`, `total_cost_usd`, `fallback_used` |

Stages **not** instrumented individually (per the plan's "meaningful
boundaries, not every helper" principle): `sanitize_input`, `process_query`'s
internal rewrite/translate sub-steps, `build_context`, `to_chart_data`. These
are cheap, deterministic, and already covered by the stage that calls them.

---

## 5. Known gaps this contract depends on (must fix before M6, not work around)

Writing this contract surfaced two real defects in the current code that
would make the telemetry **lie** if instrumentation were bolted on as-is:

1. **`LLMClient.run` never raises.** After `MAX_RETRIES` failed attempts it
   returns the literal string `"LLM failed after retries."` as if it were a
   valid answer. An `llm.completed` event built on top of this today would
   report `status=success` for a call that actually failed 4 times. Before
   M6, `run()` needs to either raise on exhaustion or return a typed
   failure the caller can check — the observability layer should not paper
   over this by string-matching the sentinel text.
2. **Cost tracking is process-global (`SESSION_COST_USD`), not per-request.**
   Under concurrent requests, `cost_usd_total` per request and
   `llm.fallback`'s budget-threshold trigger are both attributing one
   shared counter to whichever request happens to be running. This contract
   defines `cost_usd_total` as a per-request field on the assumption this
   gets fixed; until then, treat that field as approximate under concurrency.
3. **The orchestrator and pipeline functions already catch broad
   `Exception`** and return `{"error": str(e)}` dicts instead of letting
   exceptions propagate. The instrumentation wrapper (M5) must therefore
   check the returned `error` field, not rely solely on catching exceptions
   at the boundary, or failures will be silently recorded as `success`.

None of these are fixed by this document. They're listed here because they
block M6 from producing *correct* telemetry, not just present telemetry.

---

## 6. Privacy rules (non-negotiable, not just "prefer")

This is a fraud-intelligence system; queries and documents can plausibly
contain personal data. Under UU PDP's data-minimization principle, telemetry
that isn't needed to answer "what happened?" must not exist. (Exact article
numbers to confirm with legal/compliance — not guessing them here.)

- **Never** log raw query text in any structured event. Use `query_hash`
  (sha256, first 12 hex chars — enough to correlate identical queries
  without reversing them) and `query_length` instead.
- **Never** log full document/chunk content. Use `document_id` / `source_name`
  / `page` / `rank` / `score` only (already the plan's own guidance, and
  matches what `build_citations` already exposes).
- **Never** log full LLM prompts or completions in structured events — token
  counts and cost only. Free-text `logger.debug(...)` calls may still exist
  for local debugging, but must never run at `INFO` in a deployed
  environment and must be excluded from whatever aggregates/exports events.
- **Client IP** (used by the rate limiter) may be held in-memory as raw IP
  since it never leaves the process today. If rate-limit events are ever
  exported/persisted, hash or truncate the IP first — an IP is personal data
  under UU PDP once it's retained.
- This closes an existing gap, not just a future rule: current code already
  logs raw query text at `INFO` in `src/orchestrator.py`,
  `src/rag/retriever_direct.py`, and `src/analytics/fraud_analytics.py`
  (e.g. `f"query={query!r}"`). Implementing this contract means changing
  those call sites, not just adding new ones alongside them.

---

## 7. Failure semantics

- An instrumentation wrapper emits the failure event **and then re-raises**
  (or returns the same error it would have without instrumentation).
  Observability must never change what the caller receives.
- A stage that didn't run emits `skipped`, not silence — a missing event for
  an expected stage should itself be a visible anomaly, not ambiguous with
  "wasn't recorded."
- See §5 for why "no exception seen" ≠ "succeeded" in this codebase today.

---

## 8. Example trace (illustrative — not from a real run; no code exists yet)

```
request_id: 8d9f2a41-...
route: query · intent: rag · lang: en · status: success
total: 1,842 ms · cost: $0.00042 · fallback_used: false · fallback_reason: null

  0 ms   request.started            route=query
  4 ms   guardrails.completed       blocked=false
 18 ms   language_detection.completed  lang=en method=heuristic
 32 ms   intent.completed           intent=rag confidence=0.90 method=heuristic
 72 ms   retrieval.completed        method=vector_rpc candidates=42 selected=5
160 ms   ranking.completed          reranker=hybrid top_result_score=0.87
1020 ms  llm.completed              purpose=rag_answer model=gpt-4o-mini
                                    tokens=1058 cost=$0.00018 retries=0
1650 ms  scoring.completed          final_score=0.88 gate=full_ensemble
1842 ms  request.completed          status=success
```

---

## 9. Non-goals (for now)

No OpenTelemetry, no Prometheus, no external backend. This contract is the
internal abstraction; a later OTel exporter translates *from* these events,
the application never calls into a vendor SDK directly. Metrics (M8) are
derived from these events after M3–M7 land, not designed in parallel with
them.

---

## Relationship to `docs/observability.md`

That document already describes this target state in places (`query_id`
propagation, structured events, cost/latency tracking) — but it's aspirational:
today's logger (`src/utils/logger.py`) emits plain-text lines with no
`query_id`/`request_id` at all, and none of the described events exist in
code. This contract is the concrete, implementable version of that same
intent, using `request_id` as the canonical field name (synonymous with that
document's `query_id` — Phase 20 of the plan is to reconcile terminology and
rewrite `docs/observability.md` to describe the system as it actually is,
once M3–M9 are built).
