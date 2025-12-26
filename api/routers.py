# api/router.py
"""
API routing layer.

Responsibilities:
- Define public API endpoints
- Validate request schemas
- Delegate execution to application services

This module intentionally contains no business logic.
"""

from fastapi import APIRouter

from api.models import QueryRequest, RAGRequest, AnalyticsRequest
from src.orchestrator import run_query
from src.rag.rag_chain import run_rag
from src.analytics.fraud_analytics import run_analytics

router = APIRouter(prefix="", tags=["api"])


@router.post("/query", summary="Run unified AI query")
async def query_endpoint(req: QueryRequest):
    """
    Execute the unified AI orchestrator.

    This endpoint is typically used for:
    - High-level questions
    - Intelligent routing between RAG and analytics flows
    """
    return run_query(req.query)


@router.post("/rag", summary="Run Retrieval-Augmented Generation (RAG)")
async def rag_endpoint(req: RAGRequest):
    """
    Execute the RAG pipeline.

    Inputs:
    - query: User question
    - lang: User language (default: English)

    Output:
    - Context-aware LLM response
    """
    return run_rag(
        query_en=req.query,
        user_lang=req.lang,
    )


@router.post("/analytics", summary="Run fraud analytics query")
async def analytics_endpoint(req: AnalyticsRequest):
    """
    Execute fraud analytics logic.

    This endpoint is optimized for:
    - Pattern detection
    - Risk insights
    - Analytical reasoning over structured signals
    """
    return run_analytics(
        req.query,
        lang=req.lang,
    )
