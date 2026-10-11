# Evaluation & Regression Strategy

How the system's behavior is checked, what that does and does not establish, and
where the checks run. This document describes what is implemented. Anything
planned or not measured is labelled as such.

The goal is not perfect answers. It is **controlled correctness and safe failure**:
answer when the evidence supports it, refuse when it should, and notice when that changes.

---

## 1. What exists

| Piece | Where | Runs |
|---|---|---|
| Evaluation cases | `eval/cases.yaml` | |
| Checks (route, behavior, terms, language, grounding) | `eval/evaluator.py` | |
| **Offline routing gate**: no credentials, no network | `tests/test_eval_offline_routing.py` | every CI run |
| Contract tests for the checks themselves | `tests/test_eval_contract.py`, `tests/test_eval_cases.py`, `tests/test_eval_telemetry.py` | every CI run |
| **Live evaluation**: real LLM, real database | `python -m eval.run_eval --out results.json` | by hand |

---

## 2. The case set

<!-- eval-cases: 21 cases, 20 scored -->
**21 cases (20 scored, 1 known limitation)**: 16 English, 5 Indonesian.

| Category | Cases | What it checks |
|---|---|---|
| analytics | 8 | data questions routed to SQL; the answer must carry its SQL |
| rag | 6 | document questions; the answer must carry citations |
| reject | 4 | prompt injection and out-of-domain questions must be refused |
| ambiguous | 2 | no keyword rule applies; the LLM classifier decides |
| compound | 1 | **known limitation, not scored** (see below) |

Each case records what the system *should* do and which layer is expected to decide the
route: `heuristic` (keyword rules), `guardrail` (input guardrails) or `llm` (classifier).
Cases describe intended behavior; a failing live case is a finding, not a reason to edit the case.

**Compound question.** "Which merchant categories have the highest fraud rates, and what might
explain the pattern?" needs two jobs (rank with SQL, explain from documents). The system picks
one path, so it answers only the first. The case records that behavior; there is no hybrid
workflow and none is claimed.

---

## 3. What is checked

For every case, from the response `POST /query` really returns:

- **Route**: the route the API reported (`analytics`, `rag` or `reject`).
- **Behavior**: `answer`, `refusal`, `degraded` (retrieval unavailable) or `error`. A malformed
  response, an HTTP error or a rate limit is never a pass.
- **Required terms**: terms that must appear in the answer. A list means alternatives (an answer
  in another language may translate the term). This checks that expected terms are present; it
  does **not** establish that the answer is complete or correct.
- **Answer language**: judged from the answer text by a function-word count. It is a heuristic
  and returns "unknown" for text too short to judge, which fails the check.
- **Grounding**: analytics answers must carry SQL; document answers must carry citations.

---

## 4. The two kinds of check, and what each proves

**Offline routing gate (CI).** Calls the production `detect_intent` and `validate_query`
themselves, with the LLM replaced by a recorder that fails any test that reaches it. It proves:
the keyword rules and guardrails route each labelled case as `cases.yaml` says, and cases
labelled as needing the LLM really defer to it. Seven deliberate regressions injected into the
routing code (a dropped keyword, a changed threshold, a weakened guard) each made it fail.

It does **not** prove that the real LLM classifies ambiguous questions correctly, that retrieval
finds anything, or that any answer is right.

**Live evaluation (manual).** Sends every case through the real app (guardrails, router,
orchestrator) with the real LLM and a read-only database, and records latency, cost and which layer
decided the route from the system's telemetry events. It is an integration check, run by hand.
Its result is a measurement on one machine on one day, not a benchmark.

### Known gaps: recorded, enforced, and now empty

A case may record a `known_gap` in `cases.yaml`. The offline gate then expects that case to fail,
strictly: the day the gap is fixed the test fails until the record is removed, so a record cannot
go stale. The first gap found this way was the Indonesian injection case (X2): all six injection
patterns were English. Fixing the patterns exposed the next layer (language detection made an LLM
call before the rejection), and fixing that emptied the list. See section 5.

### The 0.40 fallback

A question matching no keyword rule gets a weak RAG guess at confidence 0.40 and is handed to the
LLM classifier. If the classifier gives no usable label (call failed, bad JSON, unknown intent),
the route stays that weak guess. It used to be reported as an LLM decision, which made a failed
classifier indistinguishable from a real RAG decision. It is now reported as `method=heuristic`
with confidence 0.40, which a real heuristic decision never has (those are at least 0.80).

---

## 5. Live results (2026-10-11, one machine, real OpenAI, read-only Supabase)

The 21 cases, run four times as the checks and the system were corrected. Every verdict below
was made after reading the actual answers.

| Run | System | Checks | Scored cases passed |
|---|---|---|---|
| 1 | original | first version | 15 / 20 |
| 2 | original | corrected (4 evaluator defects fixed) | 19 / 20 |
| 3 | fixed | corrected | 19 / 20 (the remaining failure was the evaluator's language judge) |
| 4 | fixed | corrected | **20 / 20** |

**What the failures were.** Run 1 had five failures: four were defects in the evaluator itself (a
language judge too small to recognise two correct Indonesian answers, and two expectations that
hard-coded a surface form) and one was a real system defect (A5). Fixing the evaluator did not
change the system; run 2 shows the same system with one failure left.

**Findings fixed in the system, each with tests that fail without the fix:**

- **A5, a value dropped from the answer.** "How many fraudulent transactions are there in total?"
  ran correct SQL but answered "Analysis processed 1 rows and 1 columns." Any analytics result that
  was not a time series or a ranking got a shape-only summary. It now states the values
  ("Result, total fraudulent transactions: 7,506.").
- **Indonesian prompt injection was not detected.** All six patterns were English. Indonesian
  equivalents were added, with benign Indonesian questions (including one that only a word boundary
  protects) that must not be flagged.
- **A rejected input could cost an LLM call.** Language detection ran before the rejection checks
  and fell back to the LLM for text its keywords did not recognise. A rejection now uses an
  LLM-free language guess; the authoritative detection runs only for accepted queries.
- **A failed intent classifier looked like a decision.** See the 0.40 fallback above.

**Measured effect (run 2, before, vs run 4, after):** the Indonesian injection (X2) went from three
LLM calls and about 2.0 s (rejected late, by the LLM classifier) to zero calls and under 5 ms
(rejected by the guardrails). Total LLM calls over the 21 cases went from 30 to 27.

**Measurements (run 4; client machine to the cloud services):**

- Request duration from telemetry: median about 3.0 s, max about 9.3 s. Template analytics answers
  took about 2.4 to 4.0 s; document answers 3.8 to 9.3 s; Indonesian document answers (with a
  translation step) 8.7 to 9.3 s; guardrail rejections a few milliseconds.
- LLM cost from telemetry: about $0.0075 for the whole run (27 priced LLM calls, 0 failures).
  Template analytics answers made no LLM call; each document answer cost about $0.001.
- Route decided by: keyword rules 15, LLM classifier 2, guardrails 4 (of 21 requests).

**Not fixed, by design or scope:**

- **A compound question gets one path.** C1 records this; there is no hybrid workflow.
- **Embedding cost is not in the telemetry.** Only LLM calls are priced. Pricing embeddings would
  change the cost semantics in `docs/observability-contract.md`, so it needs its own decision.

**Not measured:** answer quality beyond the checks above; latency from other locations; variance
beyond four runs on one day (runs 1 to 4 made the same routing decisions except where a fix changed them).

---

## 6. Deployment gating: what is and is not enforced

- CI (GitHub Actions) runs the offline routing gate on every push and pull request.
- The repository does **not** make that result block a deploy. Render deploys from GitHub on
  its own, and the repository's deploy workflow does nothing (its Render credentials resolve to
  empty values). Whether a failing CI run blocks a Render deploy depends on a Render setting
  (Auto-Deploy "After CI Checks Pass"), which is not controlled or verified from this repository.
- The live evaluation is not run automatically and gates nothing.

An earlier version of this document said behavioral regressions block deployment. That was not true.

---

## 7. What is not evaluated

- **Hallucination rate**: not measured.
- **Answer completeness and correctness**: only required-term presence is checked.
- **Refusal usefulness** and narrative quality: not automated, not reviewed systematically.
- **Retrieval quality**: not evaluated independently of the final answer.
- **Variance across runs and models**: four runs on one day.
