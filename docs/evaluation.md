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

### Known gap, recorded and enforced

All six prompt-injection patterns in the guardrails are English. The Indonesian injection case
(X2) is therefore not rejected by the guardrails. It is a strict expected failure in the offline
gate: if the guardrails are fixed, that test fails until the record is removed. In the live run
X2 was still refused, because the LLM intent classifier rejected it, at the cost of an LLM call.

### The 0.40 fallback

A question matching no keyword rule gets a weak RAG guess at confidence 0.40 and is handed to
the LLM classifier. If that classifier call fails, the result silently becomes `rag`, which cannot
be told apart from a real RAG decision. A test documents this behavior; it is a hazard, not a design goal.

---

## 5. Live results (2026-10-11, one machine, real OpenAI, read-only Supabase)

Two runs of the 21 cases. Between them, four defects in the *evaluator* were fixed after the
answers were read: the language judge was too small (two correct Indonesian answers read as
"unknown"), and two expectations hard-coded a surface form. Each fix has tests, including cases
where the wrong-language answer must still fail.

| Run | Scored cases passed | Notes |
|---|---|---|
| 1 (first checks) | 15 / 20 | 1 real system defect, 4 evaluator defects |
| 2 (corrected checks) | 19 / 20 | the real defect remains |

**The failing case (A5, a real defect).** "How many fraudulent transactions are there in total?"
produced SQL and ran it, but the summary was "Analysis processed 1 rows and 1 columns." The value
is dropped: any analytics result that is not a time series or a ranking falls back to a generic
summary (`summarize_generic` in `src/analytics/fraud_analytics.py`). Not fixed here.

**Measurements (run 2; client-side machine to the cloud services):**

- Request duration from telemetry: median ~3.0 s, p90 ~6.3 s, max ~9.1 s. Template analytics
  answers took about 2.4 to 4.0 s; document answers 3.8 to 9.1 s; Indonesian document answers
  (with a translation step) 8.7 to 9.1 s; guardrail rejections took a few milliseconds.
- LLM cost from telemetry: about $0.0076 for the whole run (all LLM calls priced, 0 failures).
  Template analytics answers made no LLM call; each document answer cost about $0.001.
- Route decided by: keyword rules 15, LLM classifier 3, guardrails 3 (of 21 requests).

**Not measured:** embedding cost (the telemetry events carry only LLM cost); anything about
answer quality beyond the checks above; latency from other locations; any repeat-run variance
beyond two runs that made identical routing decisions.

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
- **Variance across runs and models**: two runs only.
