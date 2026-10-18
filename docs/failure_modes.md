# Failure Modes & Safeguards

This document defines **known failure modes**, how they are **detected**, and how the
system **responds safely**.

The goal is not to eliminate failure, but to ensure failures are:
- Expected
- Observable
- Contained

---

## 1. Failure Philosophy

This system assumes that:
- LLMs are unreliable
- Data may be missing or incomplete
- User input may be ambiguous or adversarial

Failure is treated as a **first-class design concern**, not an edge case.

---

## 2. Failure Mode Categories

### 2.1 Input-Level Failures

#### A. Out-of-Domain Queries
**Example:**
- “How to detect crypto rug pulls?”

**Detection:**
- Domain keyword filtering
- Intent classification guardrails

**Response:**
- Explicit rejection
- Clear explanation
- No partial answers

---

#### B. Prompt Injection Attempts
**Example:**
- “Ignore previous instructions and…”

**Detection:**
- Regex-based injection patterns
- Structural markers (`system`, `assistant`, code blocks)

**Response:**
- Immediate rejection
- No LLM execution

---

#### C. Low-Signal / Noise Input
**Example:**
- Empty string
- Random characters

**Detection:**
- Length checks
- Noise-only patterns

**Response:**
- Validation error
- Request rejected before routing

---

## 3. Routing-Level Failures

### 3.1 Ambiguous Intent

**Scenario:**
- Query could be analytics or RAG

**Detection:**
- Low confidence from heuristic routing

**Response:**
- Conservative default to RAG
- Avoids unsafe or incorrect computation

---

### 3.2 Misrouted Query

**Scenario:**
- Analytics question routed to RAG or vice versa

**Detection:**
- Shape mismatch in downstream pipeline
- Empty results or invalid outputs

**Response:**
- Graceful fallback
- User-facing insufficiency message

---

## 4. Data-Level Failures

### 4.1 Missing or Sparse Data

**Scenario:**
- SQL query returns no rows
- Required fields not present

**Detection:**
- Empty dataframe checks
- Schema validation

**Response:**
- Explicit “data insufficient” message
- No fabricated insights

---

### 4.2 Partial Coverage in Documents

**Scenario:**
- Question not fully answered by available documents

**Detection:**
- Low retrieval confidence
- Low answer score

**Response:**
- Conservative answer
- Optional fallback explanation

---

### 4.3 Merchant / Category Questions Without a SQL Answer

**Scenario:**
- Ranking questions ("which merchants have the highest fraud incidence?")
  are routed to analytics and answered from SQL over `fraud_transactions`.
  The former document-wide LLM "merchant inference" path was removed
  (token-expensive and not verifiable). If SQL returns no rows or fails,
  there is intentionally no LLM fallback.

**Detection:**
- `analytics.sql.failed` / empty result (`row_count = 0`)

**Response:**
- "Insufficient data" answer (confidence 0.0), with the attempted SQL
  returned in `sql` when one was executed
- No fabricated answer from documents
- A cheaper document-based approach may be added later if needed

---

### 4.4 Retriever Unavailable vs. Nothing Relevant

**Scenario:**
- Vector search is disabled/failing, or simply finds nothing

**Detection:**
- `retrieval_status` in the RAG result: `ok | empty | unavailable`
  (plus the `retrieval.*` events)

**Response:**
- No LLM call is made without context
- `unavailable` → "search temporarily unavailable"; `empty` → "documents
  do not provide enough information"

---

## 5. Model-Level Failures

### 5.1 Hallucination Risk

**Scenario:**
- LLM produces plausible but unsupported claims

**Detection:**
- Retrieval grounding checks
- Answer confidence scoring
- Numeric consistency checks (analytics)

**Response:**
- Suppress hallucinated content
- Fallback to safe response

---

### 5.2 Low Confidence Output

**Scenario:**
- Weak semantic overlap with sources

**Detection:**
- Heuristic + embedding-based scoring
- Optional LLM judge (Phoenix)

**Response:**
- Confidence-aware fallback
- No authoritative tone

---

## 6. Cost & Resource Failures

### 6.1 Cost Overrun Risk

**Scenario:**
- Excessive LLM usage

**Detection:**
- Session-level cost tracking

**Response:**
- Model downgrade
- Execution short-circuit

---

### 6.2 External Dependency Failure

**Scenario:**
- OpenAI / Supabase unavailable

**Detection:**
- Runtime exceptions

**Response:**
- Graceful error response
- No cascading failures

---

## 7. CI & Test-Time Safeguards

- LLM execution disabled in CI
- Embeddings return deterministic dummy vectors
- Retriever can be disabled via feature flags

This ensures:
- Deterministic tests
- No external dependency coupling

---

## 8. Failure Summary

This system is designed so that:

- Failures are visible
- Unsafe behavior is blocked
- Incorrect answers are suppressed
- Users are informed, not misled

Failure handling is **intentional**, not incidental.

---

## 9. Failure containment contract

**The defect this section exists for.** The `/query`, `/rag` and `/analytics`
handlers were `async def` but called blocking code (LLM calls, retry backoff,
PostgREST, psycopg2). A blocking call inside an `async def` handler runs ON the
event loop, so one slow request froze every other request on the worker,
including `/health`. A request the guardrail *rejects* could do it, because
language detection can make an LLM call before the injection check, and a hung
upstream could hold the loop for as long as the client library allowed (the
OpenAI SDK default read timeout is 600 s).

**The contract.**

1. **Liveness isolation.** `/health` and `/` are `async def` and do no blocking
   work, so they are served on the event loop and never wait behind request work.
2. **No blocking work on the event loop.** API handlers are plain `def`; FastAPI
   runs them in its worker threadpool (anyio default: 40 threads).
3. **Every external call has an explicit deadline** (table below), so a hung
   upstream occupies a worker thread for a bounded time.
4. **One retry policy.** `LLMClient.run` is the only retry loop for LLM calls:
   the SDK's own retries are disabled, so the worst case is knowable and
   `retry_count` in the telemetry reports what actually happened. There is no
   backoff sleep after the final attempt.
5. **Shared state is guarded under concurrency.** The process-wide cost total,
   the "warn once per unpriced model" set and the lazy client singletons are
   protected, because running requests in threads made them genuinely concurrent.
   Per-request context (request id, cost) stays isolated because each request
   runs in its own copied context.

| Knob | Default | Bounds |
|---|---|---|
| `LLM_TIMEOUT_SECONDS` | `30` | one LLM attempt (SDK default was 600 s) |
| `EMBEDDING_TIMEOUT_SECONDS` | `10` | one embedding attempt; the SDK's 2 retries are kept, since a failed embedding degrades to a dummy vector |
| `SUPABASE_TIMEOUT_SECONDS` | `15` | one PostgREST call (client default was 120 s) |
| `SQL_TIMEOUT_MS` | `15000` | one analytics statement (unchanged); the connection attempt is bounded at 10 s |

Worst case for one LLM request (`llm_call_worst_case_seconds()`): 4 attempts x
30 s + 1.2 x (1 + 2 + 3) s of backoff = **127.2 s** at the defaults. A request
makes several such calls in the worst case, so the client can still wait a long
time when an upstream is hung. The deadlines bound the occupancy of a worker
thread; they do not shorten what the client sees.

**Measured** with `scripts/containment_probe.py` (local, one worker, no API key,
so the LLM call fails and backs off; the same probe before and after):

| Scenario | Before | After |
|---|---|---|
| 4 slow requests at once (wall time) | 49.0 s, serialized | 7.5 s |
| `/health` while they ran | timed out at 30 s | median 3 ms, max 5 ms |
| a fast request while they ran | waited 18.2 s | median 3 ms, max 4 ms |
| 20 slow requests at once (wall time) | not measured | 7.5 s; `/health` max 9 ms |
| `/health` during a hung upstream (30 s) | 5 of 5 probes timed out | 30 of 30 answered, max 56 ms |
| the hung request itself | not finished after 30 s (SDK default 600 s) | ended after 19.6 s with `LLM_TIMEOUT_SECONDS=3` (computed bound 19.2 s), logged as `APITimeoutError`, `retry_count=3` |

The no-key failure path went from 12 s to 7.2 s because the trailing backoff
sleep after the last attempt was removed.

**Reproduced vs inferred.** The stall, the hung-upstream freeze and the deadline
were all reproduced against a real server. Only the no-key and tarpit cases were
run: a *real* provider timeout, and a real LLM call's normal duration, were not
measured. The 600 s and 120 s library defaults are from the libraries'
documented values (observed here only for the first 30 s).

**Known limits, deliberately not addressed here.**
- *Threadpool saturation.* With 60 concurrent slow requests the 40-thread pool
  saturates: wall time doubled to 14.9 s (two waves), `/health` still answered
  in 5 ms, but a fast request waited up to 6.9 s for a free thread. A concurrency
  cap or load shedding (`503` + `Retry-After`) is a separate design decision.
- *No per-request deadline.* A request waits for its upstream calls' own
  deadlines; it is not cancelled at a total budget.
- *Guardrail ordering is unchanged.* A keyword-less rejected input still makes an
  LLM call first. It can no longer freeze the service, but it still costs a call
  and a worker thread.
- *Single worker, single process,* as elsewhere in this project.

---

## 10. Findings from the evaluation (2026-10-11)

Found by running the evaluation set ([evaluation.md](evaluation.md)). All are fixed; the last two are limits of the fixes.

| Finding | Where | Status |
|---|---|---|
| A scalar or generic analytics result lost its value ("Analysis processed N rows and M columns.") | `summarize_generic`, `src/analytics/fraud_analytics.py` | **Fixed**: the summary states the values |
| Prompt-injection patterns were English only | `FORBIDDEN_PATTERNS`, `src/safety/guardrails.py` | **Fixed**: Indonesian patterns added |
| A rejected input could cost an LLM call (language detection ran first) | `validate_query`, `src/safety/guardrails.py` | **Fixed**: rejections use an LLM-free language guess |
| A failed intent classifier was reported as an LLM decision | `detect_intent`, `src/orchestrator.py` | **Fixed**: reported as `method=heuristic`, confidence 0.40 |
| A question needing two jobs got one path | orchestrator routing | **Fixed**: the `hybrid` route runs both pipelines (see below) |
| Embedding cost was not in the telemetry | `src/embeddings/embedder.py` | **Fixed**: `embedding.completed` / `embedding.failed`, priced and counted in the request cost |
| An Indonesian "the documents do not say" refusal scored 0.415 and was shown as a confident answer | `heuristic_score`, `src/llm/scoring.py` | **Fixed**: refusal detection in both languages (`src/llm/refusal.py`) |

**Limits of the hybrid route.**

- It is decided by keyword rules only. The LLM classifier still chooses among `rag`, `analytics` and `reject`, so a compound question phrased without the rule keywords goes down one path.
- The two pipelines run one after the other, so latency adds (measured 4.6 to 12.0 s for the four evaluation cases).
- The document half is asked the explanation clause when it can stand alone (at least five words), otherwise the whole question. Given the whole question, the document model declines when the data half is not in the documents, even if the explanation half is.
- The corpus (Bhatla, EBA/ECB report) has nothing on merchant-category patterns, so the documents cannot explain them: those compound questions are correctly answered as `degraded` (numbers with SQL, and an honest "not enough information" for the explanation).

**Embedding cost.** One query embedding is about 7 to 17 tokens, about $0.0000002 to $0.0000003. Over the 24 evaluation cases (11 embedding calls, 120 tokens) it was about $0.0000024, about 0.025% of the LLM spend (a fortieth of one percent). It is below the 6-decimal resolution of a request's `cost_usd_total`, so it is recorded per call (9 decimals) and in `embedding_cost_usd_total`.

---

## 11. Contract boundary probe (2026-10-18)

Ten boundary questions run once against the live code with a real LLM, to see which clauses of
[system_contract.md](system_contract.md) are enforced and by what. One machine, one day: a probe,
not a benchmark. The result is the enforcement ledger in section 9 of the contract. Recorded, not fixed.

| Finding | Effect |
|---|---|
| Crypto fraud, insurance fraud, general programming, legal advice and accusing a named merchant are refused only by the LLM classifier | Refusal costs 1 to 3 LLM calls and is not a deterministic guarantee: the guardrail domain allowlist lets any input containing "fraud" through |
| Languages other than English and Indonesian are not rejected | A French question was processed with 5 LLM calls and answered in English with "not enough information" |
| A two-part question is routed as one | "What is card-not-present fraud? Also, which merchants have the highest fraud incidence?" answered only the second half, silently |
| "Predict future fraud" is a stated non-goal but nothing enforces it | "Will card fraud rise next year?" was routed to analytics and ended in "Insufficient data" by accident |
| AML is routed to RAG, not rejected | The documents do not cover it, so RAG refuses after 5 LLM calls: consistent with "unless covered by the documents", but costly |

