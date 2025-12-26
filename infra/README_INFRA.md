# Infrastructure Notes — AI Fraud Agents Systems

This folder documents infrastructure and deployment considerations
for the Fraud Agents Enhanced system.

The goal is architectural clarity, not full IaC coverage.

## Scope & Intent

This project is designed as a portfolio-grade system architecture.
Infrastructure is intentionally simplified to focus on:

- AI orchestration
- Routing logic
- Safety and failure handling
- Production-readiness patterns

## Current Deployment Model

- Stateless FastAPI service
- Container-based deployment
- Environment-variable configuration
- External managed services:
  - LLM providers
  - Supabase (Postgres + vector storage)

This matches a common early-stage or internal tooling setup.


## What Is Intentionally Simplified

The following are NOT implemented in this project:

- Infrastructure-as-Code (Terraform / Pulumi)
- Kubernetes / service mesh
- Secrets management systems (Vault, AWS Secrets Manager)
- Autoscaling policies
- Multi-region deployment

These omissions are deliberate and documented.

## Production Mapping

In a real production environment, this system would typically add:

- IaC for reproducible environments
- CI-triggered deployments with rollback
- Load balancers and autoscaling
- Centralized logging and metrics
- Authentication and authorization layers

## Architecture Diagram Reference

The system architecture diagram is defined in:

docs/architecture.mmd

This diagram focuses on logical system components and decision flow,
not infrastructure provisioning.

It illustrates:
- Guardrails and safety boundaries
- Intent classification and routing
- Orchestration between RAG and analytics paths
- Failure handling and fallback behavior


## Why This README Exists

This document exists to make infrastructure assumptions explicit.

It demonstrates:
- Awareness of production constraints
- Conscious tradeoffs, not omissions
- Separation between system design and infra implementation

