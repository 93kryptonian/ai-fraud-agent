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
