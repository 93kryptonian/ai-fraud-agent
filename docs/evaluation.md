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

<!-- eval-cases: 24 cases, 24 scored -->
**24 cases (all scored)**: 17 English, 7 Indonesian.

| Category | Cases | What it checks |
|---|---|---|
| analytics | 8 | data questions routed to SQL; the answer must carry its SQL |
| rag | 6 | document questions; the answer must carry citations |
| reject | 4 | prompt injection and out-of-domain questions must be refused |
| ambiguous | 2 | no keyword rule applies; the LLM classifier decides |
| compound | 4 | numbers and an explanation in one question: the `hybrid` route |

Each case records what the system *should* do and which layer is expected to decide the
route: `heuristic` (keyword rules), `guardrail` (input guardrails) or `llm` (classifier).
Cases describe intended behavior; a failing live case is a finding, not a reason to edit the case.

**Compound questions.** A question like "which merchants have the highest fraud incidence, and why
does card-not-present fraud occur?" needs two jobs, so it takes the `hybrid` route: analytics (with
its SQL) and RAG (with its citations) both run. The expectation depends on the corpus. Where the
documents can explain (C3, C4) the answer must be a full `answer` carrying both kinds of evidence.
Where they cannot (C1, C2: the documents say nothing about merchant-category patterns) the correct
outcome is `degraded`: the numbers are answered with their SQL and the explanation honestly says
there is not enough information. A corpus gap is reported as `degraded`, never as a pass.

---

## 3. What is checked

For every case, from the response `POST /query` really returns:

- **Route**: the route the API reported (`analytics`, `rag`, `hybrid` or `reject`).
- **Behavior**: `answer`, `refusal`, `degraded` (retrieval unavailable, or one half of a hybrid answer unsupported) or `error`. A malformed
  response, an HTTP error or a rate limit is never a pass.
- **Required terms**: terms that must appear in the answer. A list means alternatives (an answer
  in another language may translate the term). This checks that expected terms are present; it
  does **not** establish that the answer is complete or correct.
- **Answer language**: judged from the answer text by a function-word count. It is a heuristic
  and returns "unknown" for text too short to judge, which fails the check.
- **Grounding**: analytics answers must carry SQL; document answers must carry citations; a hybrid answer must carry both.

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

The cases, run six times as the checks and the system were corrected. Every verdict below was
made after reading the actual answers.

| Run | System | Cases | Scored cases passed |
|---|---|---|---|
| 1 | original | 21 | 15 / 20 |
| 2 | original, corrected checks | 21 | 19 / 20 |
| 3 | after the first four fixes | 21 | 19 / 20 (the failure was the evaluator's language judge) |
| 4 | after the first four fixes | 21 | 20 / 20 |
| 5 | hybrid route and embedding cost added | 22 | 21 / 22 (one case passed that should not have: see below) |
| 6 | all fixes | 24 | 23 / 24 |

**What the failures were.** Run 1 had five failed cases: four were caused by the evaluator itself,
from three distinct defects (a language judge too small to recognise two correct Indonesian answers,
and two expectations that hard-coded a surface form), and one was a real system defect (A5). A
fourth evaluator defect surfaced after A5 was fixed: the judge did not know the system's new
generic answer templates (run 3). Run 5 is the instructive one: its Indonesian
compound case passed, but its document half was an apology ("dokumen yang tersedia tidak memberikan
cukup informasi"), which the system scored 0.415 and the evaluator accepted. The refusal check knew
only the English sentence. Fixing it made that case honestly `degraded`.

**Run 6's one failure (R1)** is model variance: "what are the main fraud typologies..." failed its
required-term check ("fraud" not in the answer) once, after passing in every earlier run; four
re-asks all passed. A single live run is a sample, not a proof, and a required term in generated
text can be flaky.

**Findings fixed in the system, each with tests that fail without the fix:**

- **A5, a value dropped from the answer.** "How many fraudulent transactions are there in total?"
  ran correct SQL but answered "Analysis processed 1 rows and 1 columns." It now states the values.
- **Indonesian prompt injection was not detected.** All six patterns were English; Indonesian ones
  were added, with benign Indonesian questions that must not be flagged.
- **A rejected input could cost an LLM call.** Language detection ran before the rejection checks.
- **A failed intent classifier looked like a decision.** It is now reported as `method=heuristic`
  with confidence 0.40, which a real heuristic decision never has.
- **A compound question got one path.** The `hybrid` route runs both pipelines (below).
- **Embedding cost was not in the telemetry.** Embedding calls are now events, priced from usage
  and counted in the request cost (below).
- **An Indonesian refusal scored as a confident answer.** Refusal detection now covers both languages.

**Measured effects:**

- *Indonesian injection (X2):* from three LLM calls and about 2.0 s (rejected late, by the LLM
  classifier) to no LLM calls and under 5 ms (rejected by the guardrails).
- *Hybrid:* the four compound cases took 4.6 to 12.0 s (the two pipelines run one after the other)
  and made 2 to 5 LLM calls and one embedding call each. C3 and C4 returned full answers with SQL
  and citations; C1 and C2 returned `degraded`, correctly, for the corpus reason above.
- *Embedding cost:* 11 embedding calls, 120 tokens, about $0.0000024 in total, about 0.025% of
  the roughly $0.0096 of LLM spend in the same run (a fortieth of one percent). It is below the 6-decimal
  resolution of a request's `cost_usd_total`, so it is recorded per call and in
  `embedding_cost_usd_total`.

**Measurements (run 6; client machine to the cloud services):**

- Request duration from telemetry: median about 3.3 s, max about 12.0 s. Template analytics answers
  took about 2.3 to 3.9 s; document answers 3.2 to 8.8 s; guardrail rejections a few milliseconds.
- LLM calls: 37 over 24 requests, 0 failures. Route decided by: keyword rules 18, LLM classifier 2,
  guardrails 4.

**Limits that remain:** `hybrid` is decided by keyword rules only; the LLM classifier cannot
choose it, so a compound question phrased without the rule keywords still goes down one path. The
corpus has no evidence on merchant-category patterns.

**Not measured:** answer quality beyond the checks above; latency from other locations; variance
beyond six runs on one day.

---

## 6. Deployment gating: what is and is not enforced

- CI (GitHub Actions) runs the offline routing gate on pushes to `main`, `dev` and `feature/**`, and
  on pull requests. A branch with any other name (for example `fix/...`) gets no CI run.
- The repository does **not** make that result block a deploy. Render is connected to
  GitHub and builds new commits itself, and the repository's deploy workflow does nothing (its
  Render credentials resolve to empty values), so nothing in the repository makes a deploy wait
  for CI. Whether a failing CI run blocks a Render deploy depends on a Render setting (Auto-Deploy
  "After CI Checks Pass"), which is not controlled or verified from this repository.
- The live evaluation is not run automatically and gates nothing.

An earlier version of this document said behavioral regressions block deployment. That was not true.

---

## 7. What is not evaluated

- **Hallucination rate**: not measured.
- **Answer completeness and correctness**: only required-term presence is checked.
- **Refusal usefulness** and narrative quality: not automated, not reviewed systematically.
- **Retrieval quality**: not evaluated independently of the final answer.
- **Variance across runs and models**: six runs on one day; one required-term check failed once for no system reason.
