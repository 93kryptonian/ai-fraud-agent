"""
Response schemas for the Fraud AI System.

This module defines all structured outputs returned by:
- RAG pipelines
- Analytics pipelines
- Guardrail / error handling

These schemas act as the public API contract between:
LLM ↔ Orchestrator ↔ API ↔ UI
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# =============================================================================
# RAG RESPONSE SCHEMAS
# =============================================================================

class Citation(BaseModel):
    """
    Single citation item used in RAG responses.

    Fields map directly to what the RAG pipeline produces:
    - source: document name (e.g., 'Bhatla.pdf')
    - page: page number within the document (if available)
    - preview: short snippet from the cited text
    """
    source: str = Field(
        ...,
        description="Source document name."
    )
    page: Optional[int] = Field(
        default=None,
        description="Page number in the source document, if available."
    )
    preview: Optional[str] = Field(
        default=None,
        description="Short text snippet from the cited chunk."
    )


class RAGResponse(BaseModel):
    """
    Structured output for Retrieval-Augmented Generation (RAG).

    This schema is intentionally explicit to:
    - prevent UI ambiguity
    - support explainability
    - enable confidence-aware fallbacks
    """
    answer: str = Field(
        ...,
        description="Final factual answer grounded in retrieved context."
    )
    insight: Optional[str] = Field(
        default=None,
        description="Expert interpretation or implications derived from the answer."
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Structured citations pointing to source documents."
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Overall confidence score for the answer (0–1)."
    )

# =============================================================================
# ANALYTICS RESPONSE SCHEMAS
# =============================================================================

class AnalyticsRow(BaseModel):
    """
    Single data point for chart or metric display.
    """
    label: str = Field(
        ...,
        description="Label for the data point (e.g., date, category, merchant)."
    )
    value: float = Field(
        ...,
        description="Numeric value associated with the label."
    )


class AnalyticsResponse(BaseModel):
    """
    Structured output for analytics queries.

    Supports:
    - KPIs
    - aggregated fraud counts
    - time-series data
    - merchant/category rankings
    """
    answer: str = Field(
        ...,
        description="Natural language summary of the analytics result."
    )
    data_points: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Raw analytics rows for tabular inspection."
    )
    chart_data: Optional[List[AnalyticsRow]] = Field(
        default=None,
        description="Chart-friendly representation of the analytics result."
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence score for the analytics interpretation (0–1)."
    )

# =============================================================================
# ERROR RESPONSE SCHEMA
# =============================================================================

class ErrorResponse(BaseModel):
    """
    Standardized error response.

    Used when:
    - guardrails trigger
    - downstream systems fail safely
    - analytics or RAG cannot proceed
    """
    error: str = Field(
        ...,
        description="Short error message."
    )
    details: Optional[str] = Field(
        default=None,
        description="Optional technical details for debugging."
    )

# =============================================================================
# INTERNAL / DEBUG RESPONSE SCHEMA
# =============================================================================

class InternalModelResponse(BaseModel):
    """
    Internal-only LLM response structure.

    Used for:
    - debugging
    - logging
    - evaluation
    Not exposed to external consumers.
    """
    raw_answer: str = Field(
        ...,
        description="Raw text output returned by the LLM."
    )
    parsed: Optional[dict] = Field(
        default=None,
        description="Parsed structured output, if applicable."
    )
    validated: bool = Field(
        default=False,
        description="Whether the response successfully passed schema validation."
    )
