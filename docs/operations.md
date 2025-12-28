# Operations & Ownership

This document describes how the Fraud Agents Enhanced system is operated,
maintained, and evolved once deployed.

The focus is not DevOps tooling, but **operational ownership**:
knowing what to do when something breaks, degrades, or must change.

## Operational Philosophy

Operations is treated as a continuation of system design.

Principles:
- Every failure mode must have a known response
- AI behavior changes are treated as deployments
- Cost and quality regressions are operational incidents
- Manual intervention paths must exist
- Ownership is explicit, not implied

## Deployment Model

The system is deployed as a stateless FastAPI service.

Characteristics:
- Single entry point (API gateway / FastAPI)
- Stateless request handling
- Externalized state (documents, vectors, analytics data)
- Environment-driven configuration

This allows:
- Safe restarts
- Horizontal scaling
- Predictable failure recovery

## CI / CD Responsibilities

Continuous Integration ensures:
- Code imports correctly
- No accidental LLM calls in CI
- Routing logic remains callable
- System contracts remain intact

Continuous Deployment:
- Triggers on main branch updates
- Deploys the API service automatically
- Treats prompt and routing changes as deployable artifacts

Any change that affects behavior is considered a production change.

## Change Management for AI Behavior

Not all changes are equal.

High-risk changes:
- Prompt updates
- Routing logic modifications
- Model swaps
- Cost thresholds
- Guardrail rules

These changes require:
- Offline evaluation (Layer 3)
- Regression checks
- Observability review post-deploy

AI behavior drift is treated as an operational risk.

## Common Incident Scenarios

Examples of operational incidents:
- Sudden increase in fallback answers
- Unexpected cost spikes
- Routing misclassification
- Increased latency
- Language handling regressions

Expected responses:
- Inspect query-level traces
- Identify failing step
- Roll back recent changes
- Apply stricter routing or guardrails if needed

## Manual Override Paths

The system is designed with manual control points:

- Disable LLM reranking
- Force fallback responses
- Lower cost thresholds
- Restrict supported routes
- Temporarily disable analytics or RAG

These controls ensure the system can degrade safely instead of failing catastrophically.

## Scaling Considerations

Scaling is addressed at the system level, not the model level.

Key considerations:
- Stateless API enables horizontal scaling
- Retrieval and analytics scale independently
- Cost scales with usage, not data size
- Evaluation and observability scale before traffic

This avoids premature optimization while remaining production-aligned.

