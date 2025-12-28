# Offline Evaluation Set

This folder contains the **offline evaluation artifacts** for the Fraud Agents Enhanced system.

The purpose of this evaluation is **not model benchmarking**, but to validate
**system behavior** under realistic queries and detect regressions as the system evolves.

## What This Evaluation Covers

The evaluation focuses on **behavioral correctness**, not text quality.

Specifically, it validates:

- Correct routing between RAG, analytics, and rejection
- Correct decision to answer vs refuse
- Presence of required factual elements in answers
- Language correctness (English / Indonesian)

This mirrors how real AI systems are evaluated in production settings.

## What This Evaluation Does NOT Cover

To remain honest and focused, the following are **not automatically evaluated**:

- Stylistic fluency or writing quality
- Insight narrative depth
- Subjective usefulness
- Model creativity or phrasing variations

These aspects are reviewed manually and treated as **product quality**, not correctness.

## Folder Structure

The evaluation set is intentionally simple:

eval/
- queries.yaml        # Input queries
- expectations.yaml  # Expected system behavior
- run_eval.py        # Minimal evaluation harness
- README.md          # This document

## Query Set

`queries.yaml` contains a small set of realistic user queries.

Each query includes:
- A unique identifier
- The query text
- Expected input language

The queries cover:
- RAG questions
- Analytics questions
- Ambiguous queries
- Out-of-domain queries

## Expected Behavior

`expectations.yaml` defines the **system truth** for each query.

For every query, it specifies:
- Expected routing decision (rag / analytics / reject)
- Expected behavior (answer or refusal)
- Required facts that must appear in the answer (if applicable)
- Expected output language

This file represents the behavioral contract of the system.

## Evaluation Harness

`run_eval.py` executes the evaluation by:

1. Running each query through the orchestrator
2. Capturing routing decisions and outputs
3. Comparing results against expected behavior
4. Failing fast on behavioral mismatches

The harness is intentionally lightweight and CI-friendly.

## Regression Testing

The same evaluation set is reused to detect regressions caused by:

- Model upgrades
- Prompt changes
- Routing logic modifications
- New document ingestion

Any change that alters expected behavior should be intentional and reviewed.

## Why This Exists

Most AI systems fail silently when behavior degrades.

This evaluation set ensures that the system:
- Answers when confident
- Refuses when uncertain
- Degrades safely
- Remains behaviorally stable over time

## Final Note

This evaluation strategy prioritizes **correct decisions over perfect answers**.

