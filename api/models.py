# api/models.py
"""
Pydantic request models for the public API.

These models define the external contract of the service and are
intentionally kept minimal and stable.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request model for the unified AI query endpoint.
    """
    query: str = Field(
        ...,
        description="User input query",
        example="Detect suspicious transaction patterns in Q4",
    )


class RAGRequest(BaseModel):
    """
    Request model for Retrieval-Augmented Generation (RAG).
    """
    query: str = Field(
        ...,
        description="User question for context-aware retrieval",
        example="What are common fraud indicators in fintech?",
    )
    lang: str = Field(
        default="en",
        description="User language code",
        example="en",
    )


class AnalyticsRequest(BaseModel):
    """
    Request model for fraud analytics queries.
    """
    query: str = Field(
        ...,
        description="Analytical query over fraud signals",
        example="Show high-risk users with abnormal login frequency",
    )
    lang: str = Field(
        default="en",
        description="User language code",
        example="en",
    )
