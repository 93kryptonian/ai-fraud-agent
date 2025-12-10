from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================================
# RAG RESPONSE SCHEMA
# ============================================================

class Citation(BaseModel):
    """
    Single citation item used in RAG responses.
    Matches what rag_chain builds:
      - source: document name (e.g. 'Bhatla.pdf')
      - page: page number (if known)
      - preview: short text snippet
    """
    source: str = Field(..., description="Document/source name.")
    page: Optional[int] = Field(
        default=None,
        description="Page number in the source document, if available."
    )
    preview: Optional[str] = Field(
        default=None,
        description="Short snippet from the cited chunk."
    )


class RAGResponse(BaseModel):
    """
    Structured output for RAG answers.

    - answer: factual, grounded answer (LLM from RAG context).
    - insight: expert interpretation / implications (insight layer).
    - citations: structured citations from retrieval.
    - confidence: overall confidence in this answer (0-1).
    """
    answer: str = Field(
        ...,
        description="Final factual answer in the user's language."
    )
    insight: str = Field(
        ...,
        description="Insightful expert interpretation in the user's language."
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of citations pointing to source documents."
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Model confidence in the answer."
    )


# ============================================================
# ANALYTICS RESPONSE SCHEMA
# ============================================================

class AnalyticsRow(BaseModel):
    label: str
    value: float


class AnalyticsResponse(BaseModel):
    """
    Structured analytics output, e.g.:
        - single-value KPIs
        - aggregated fraud counts
        - time series
        - merchant/category summaries
    """
    answer: str
    data_points: Optional[List[Dict[str, Any]]] = None
    chart_data: Optional[List[AnalyticsRow]] = None
    confidence: float = 0.7


# ============================================================
# ERROR RESPONSE (GUARDRAILS)
# ============================================================

class ErrorResponse(BaseModel):
    """
    Standardized error object for safe and predictable failures.
    """
    error: str
    details: Optional[str] = None


# ============================================================
# INTERNAL LLM RESPONSE SCHEMA
# (Used for debugging/logging)
# ============================================================

class InternalModelResponse(BaseModel):
    """
    Internal response for debugging or logging purposes.
    Not exposed to UI.
    """
    raw_answer: str
    parsed: Optional[dict] = None
    validated: bool = False
