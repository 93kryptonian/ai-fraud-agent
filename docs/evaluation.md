# Evaluation & Regression Strategy

This document defines how the **Fraud Agents Enhanced** system is evaluated,
monitored for degradation, and validated over time.

The goal is not perfect answers — but **controlled correctness and safe failure**.

---

## 1. Evaluation Philosophy

This system assumes:
- LLM outputs are probabilistic
- Correct behavior sometimes means **refusing to answer**
- Silent degradation is worse than visible failure

Evaluation focuses on **behavioral correctness**, not benchmark scores.

---

## 2. Offline Evaluation Set

The system is evaluated using a **small, curated offline dataset**.

### 2.1 Query Set

- Size: ~20–50 real-world queries
- Languages:
  - English (EN)
  - Indonesian (ID)
- Mix:
  - RAG questions
  - Analytics questions
  - Ambiguous queries
  - Out-of-domain queries

Queries are manually curated to represent realistic usage,
not synthetic benchmarks.

---

### 2.2 Labeled Expectations

Each query is labeled with expected behavior:

- Expected route:
  - `rag`
  - `analytics`
  - `reject`
- Expected outcome:
  - Answer
  - Refusal
  - Degraded response
- Required facts:
  - Key concepts or numbers that **must appear**
- Language expectation:
  - EN or ID

This allows evaluation of **decision correctness**, not just text similarity.

---

## 3. Metrics Tracked

### 3.1 Routing Accuracy

**Question:**
Did the system choose the correct pipeline?

Measured as:
- Correct route / total queries

Routing errors are treated as **system-level failures**, not model failures.

---

### 3.2 Hallucination Rate

**Question:**
Did the system introduce facts not present in documents or data?

Measured as:
- Hallucinated answers / total answered queries

Refusals are **not counted as hallucinations**.

---

### 3.3 Answer Completeness

Binary measure:
- `1` → Required facts present
- `0` → Required facts missing

This avoids subjective scoring.

---

### 3.4 Language Correctness

Checks that:
- Output language matches input language
- No unintended language mixing

Critical for multilingual production systems.

---

## 4. Abstention Quality (Hidden Gap)

Most AI systems evaluate only answers.
This system evaluates **refusals**.

### 4.1 Correct Refusal

**Question:**
Was refusing the correct behavior?

Examples:
- Out-of-domain queries
- Insufficient data
- Ambiguous intent

Incorrect refusal is treated as **false negative**.

---

### 4.2 Refusal Usefulness

A refusal is considered useful if it:
- Clearly explains why the system cannot answer
- Does not hallucinate partial information
- Maintains a professional, non-authoritative tone

This is manually reviewed in offline evaluation.

---

## 5. Regression Testing

The evaluation set is reused for regression testing under changes such as:

- Model upgrade
- Prompt modification
- New document ingestion
- Routing logic changes

### 5.1 Regression Checks

For the same query:
- Route must remain stable (unless intentionally changed)
- Refusal vs answer behavior must not regress
- Required facts must still appear

Behavioral regressions block deployment.

---

## 6. Evaluation Harness

Evaluation is implemented as a **simple Python harness**:

- Executes queries via orchestrator
- Captures:
  - Route
  - Answer
  - Confidence
  - Language
- Compares output against labeled expectations

This is intentionally lightweight and CI-compatible.

---

## 7. What Is Explicitly Not Evaluated

To maintain honesty, the following are **not fully automated**:

- Insight narrative quality
- Stylistic fluency
- Subjective usefulness

These are reviewed manually and treated as **product quality**, not correctness.

---

## 8. Evaluation Summary

This evaluation strategy ensures that:

- Correctness is measurable
- Failures are visible
- Refusals are intentional and testable
- System behavior remains stable as components evolve

The system is considered successful when it:
**answers when confident, refuses when uncertain, and degrades safely**.

---

The offline evaluation set is maintained under `/eval`,
with explicitly labeled expected behavior for routing and abstention.
