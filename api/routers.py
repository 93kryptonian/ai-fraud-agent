# api/router.py
"""
API routing layer.

Responsibilities:
- Define public API endpoints
- Validate request schemas
- Enforce input guardrails (safety/domain) before delegating
- Delegate execution to application services

This module intentionally contains no domain reasoning logic — only
request validation and guardrail enforcement live here.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.models import QueryRequest, RAGRequest, AnalyticsRequest
from src.orchestrator import run_query
from src.rag.rag_chain import run_rag
from src.analytics.fraud_analytics import run_analytics
from src.safety.guardrails import validate_query
from src.llm.response_schema import ErrorResponse

router = APIRouter(prefix="", tags=["api"])


@router.post("/query", summary="Run unified AI query")
async def query_endpoint(req: QueryRequest):
    """
    Execute the unified AI orchestrator.

    This endpoint is typically used for:
    - High-level questions
    - Intelligent routing between RAG and analytics flows
    """
    ok, cleaned_or_msg, lang = validate_query(req.query)
    if not ok:
        return {
            "query": req.query,
            "intent": "reject",
            "error": None,
            "result": {"type": "reject", "message": cleaned_or_msg},
        }

    return run_query(cleaned_or_msg, detected_lang=lang)


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
    ok, cleaned_or_msg, _ = validate_query(req.query)
    if not ok:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error=cleaned_or_msg).model_dump(),
        )

    return run_rag(
        query_en=cleaned_or_msg,
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
    ok, cleaned_or_msg, _ = validate_query(req.query)
    if not ok:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error=cleaned_or_msg).model_dump(),
        )

    return run_analytics(
        cleaned_or_msg,
        lang=req.lang,
    )
