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
  - Intent classification (`rag`, `analytics`, or `reject`)
  - A grounded answer or rejection message
- Language preservation:
  - Output matches the user’s input language
- Safety enforcement:
  - No hallucinated facts
  - No out-of-domain content
- Deterministic routing decisions for identical inputs

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
| Ambiguous intent | Conservative | Default to RAG |
| Required data missing | Degrade | Inform user of insufficiency |
| Low confidence score | Fallback | Safe explanatory response |
| Out-of-domain or unsafe input | Reject | Explicit refusal |

Routing decisions are deterministic given the same input.

---

## 8. Determinism Boundaries

### 8.1 Deterministic Components

The following components produce stable outputs:

- Input validation & guardrails
- Language detection heuristics
- Routing heuristics
- SQL queries & analytics computation
- Cost tracking and enforcement
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
- Cost ceilings to prevent uncontrolled execution

The system treats LLMs as **unreliable by default**, not authoritative.

---

## 9. Contract Summary

This system is designed with **explicit guarantees and known limitations**.

- Capabilities are intentional
- Failure modes are anticipated
- Uncertainty is bounded and surfaced

This document represents the system’s **hard truth**.
