# System Contract & Determinism Boundaries

This document defines the **explicit contract, decision logic, and determinism boundaries**
of the **AI Fraud Agents** system.

Its purpose is to make system behavior **predictable, auditable, and reviewable**.
This is not marketing documentation — it is a statement of truth.

---

## 1. System Scope

The system is an **AI-powered fraud intelligence engine** designed to:

- Answer **fraud-related knowledge questions** using document-grounded retrieval (RAG)
- Perform **fraud analytics** directly from transaction datasets
- Route queries deterministically between analytical and retrieval pipelines
- Enforce strict domain and safety constraints

The system is **not** a general-purpose chatbot.

---

## 2. Inputs Accepted

The system accepts **natural language queries** that meet all of the following criteria:

- Language:
  - English (`en`)
  - Indonesian (`id`)
- Domain:
  - Card fraud
  - Payment fraud
  - Merchant fraud
- Intent:
  - Analytical questions requiring computation
  - Knowledge questions answerable from supported documents
- Format:
  - Plain text
  - Single query per request

**Examples (Accepted):**
- “How does the monthly fraud rate change over time?”
- “Which merchant categories show the highest fraud incidence?”
- “Apa itu card-not-present fraud?”

---

## 3. Inputs Explicitly Rejected

The system explicitly rejects queries that are:

### 3.1 Out of Domain
- Crypto fraud
- Insurance fraud
- AML / KYC (unless explicitly covered by documents)
- General AI, programming, or math questions

### 3.2 Unsafe or Manipulative
- Prompt-injection attempts
- System override requests
- Role manipulation (e.g. “act as system”)

### 3.3 Low-Signal Input
- Empty or near-empty queries
- Noise-only input
- Non-semantic text

Rejected inputs result in **explicit refusal**, not silent failure.

---

## 4. Outputs Guaranteed

For all valid requests, the system guarantees:

- A structured response containing:
  - Intent classification (`rag`, `analytics`, `hybrid`, or `reject`). `hybrid` is a question that needs both the numbers (SQL) and an explanation from the documents: both pipelines run, and the answer says which half could not be supported
  - A grounded answer or rejection message
- Language preservation:
  - Output matches the user’s input language
- Safety enforcement, with the limits listed in section 9:
  - Answers are built from retrieved passages or from executed SQL and carry that evidence
    (citations, the SQL). A hallucination rate has **not been measured**, so "no hallucinated
    facts" is an intent, not a verified guarantee.
  - Out-of-domain input is refused: deterministically by the guardrails when it has no fraud or
    finance keyword, and by the LLM classifier when it does (for example crypto or insurance fraud).
- Deterministic routing for the questions the keyword rules and the guardrails decide.
  Questions that reach the LLM classifier are not guaranteed to route identically.

---

## 5. Outputs Best-Effort (Not Guaranteed)

The following outputs are best-effort and may degrade gracefully:

- Insight depth and narrative quality
- Confidence scoring precision
- Answer completeness when data or documents are sparse
- Natural language fluency (LLM-dependent)

In all cases, degradation is **explicit**, never fabricated.

---

## 6. Explicit Non-Goals

The system does **not** attempt to:

- Predict future fraud events
- Perform real-time transaction scoring
- Replace production fraud detection engines
- Act as a general conversational assistant
- Provide legal or regulatory advice beyond document content

These exclusions are intentional to maintain correctness and trust.

---

## 7. Routing & Decision Logic

Routing is determined through a hybrid strategy:

| Condition | Route | Behavior |
|--------|------|---------|
| Statistical trends or rankings requested | Analytics | SQL execution + aggregation |
| Conceptual or regulatory question | RAG | Document retrieval + grounded answer |
| Question needs numbers and an explanation | Hybrid | Analytics, then RAG; the answer says which half could not be supported |
| Ambiguous intent | LLM classifier | If it fails, the weak keyword guess (RAG, confidence 0.40), reported as such |
| Required data missing | Degrade | Inform user of insufficiency |
| Low confidence score | Fallback | Safe explanatory response |
| Out-of-domain or unsafe input | Reject | Explicit refusal |

Routing is deterministic when the keyword rules or the guardrails decide (in the last live
evaluation of 24 cases: 18 by keyword rules, 4 by guardrails). It is not guaranteed when the LLM
classifier decides (2 of 24): the classifier runs at temperature 0, which makes it stable, not
deterministic.

---

## 8. Determinism Boundaries

### 8.1 Deterministic Components

The following components produce stable outputs:

- Input validation & guardrails
- Language detection heuristics
- Routing heuristics
- SQL queries & analytics computation
- Cost tracking (LLM and embedding responses priced from usage; an unknown price stays unknown)
- CI behavior (LLM disabled)

These components define the **system backbone**.

---

### 8.2 Probabilistic Components

The following components are probabilistic:

- LLM-based intent classification fallback
- Query rewriting
- Natural language answer generation
- Insight interpretation layer
- Optional LLM-based evaluation

---

### 8.3 Randomness Control

Probabilistic behavior is bounded by:

- Low-temperature generation for factual steps
- Explicit fallback behavior on low confidence
- Feature flags to disable LLMs in CI
- A soft cost ceiling that downgrades to a fallback model only when one is configured. By default
  the primary and fallback models are the same, so by default it changes nothing

The system treats LLMs as **unreliable by default**, not authoritative.

---

## 9. Enforcement Ledger

A clause is only as real as its enforcement. Each clause above is enforced by one of four things:
**deterministic** code, a **classifier** (probabilistic), **incidental** behavior, or **none**.
Status comes from reading the code and from one live probe of ten boundary questions on
2026-10-18 (one machine, one day; a probe, not a benchmark). `tests/test_system_contract.py`
pins the entries marked with a test, so a fix forces this table to be updated.

| Clause | Enforced by | Status |
|---|---|---|
| Input has a fraud or finance keyword (domain) | deterministic: guardrail domain allowlist | Enforced in both languages with no LLM call (Indonesian cooking question, English recipe question) |
| Prompt injection refused | deterministic: guardrail patterns, English and Indonesian | Enforced, tested |
| Empty, near-empty or noise-only input refused | deterministic: guardrails | Enforced, tested |
| Out of domain that still contains the word "fraud" (crypto fraud, insurance fraud) | classifier: the prompt lists them as reject | Refused in the probe, but only by the LLM classifier after one LLM call. Not a deterministic guarantee. The guardrails let both through (tested) |
| General programming or math that mentions fraud | classifier | Refused in the probe by the classifier (3 LLM calls). Not deterministic |
| Legal advice; accusing a named merchant | classifier | Refused in the probe by the classifier. Not deterministic |
| AML or KYC "unless covered by the documents" | keyword rules route it to RAG | Not rejected: the documents do not cover it, so RAG answers "not enough information" after 5 LLM calls. Consistent with "unless covered", but costly |
| Predicting future fraud (non-goal) | none | **NOT enforced.** "Will card fraud rise next year?" was routed to analytics and ended in "Insufficient data" by accident |
| Languages: English and Indonesian only | none | **NOT enforced.** A French question was processed (5 LLM calls) and answered in English with "not enough information". Language detection only returns `en` or `id` |
| Single query per request | none | **NOT enforced.** A two-part question answered only one half, silently |
| Output language matches input language | evaluation heuristic (not a runtime check) | Measured in evaluation only |
| No hallucinated facts | none | **Not measured.** See section 4 |
| Deterministic routing for identical inputs | deterministic for rule- and guardrail-decided questions | True for those, not for classifier-decided ones (section 7) |
| Cost ceilings | soft budget guard | Exists, ineffective by default (section 8.3) |

---

## 10. Contract Summary

This system is designed with **explicit guarantees and known limitations**.

- Capabilities are intentional
- Failure modes are anticipated
- Uncertainty is bounded and surfaced

This document represents the system’s **hard truth**.
