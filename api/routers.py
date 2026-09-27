# api/router.py
"""
API routing layer.

Responsibilities:
- Define public API endpoints
- Assign a request_id to every incoming request (M3 observability context),
  before anything else runs, so even a guardrail rejection is correlatable
- Emit the request/guardrail structured events (M4), timed (M5), at the
  actual decision boundary — based on validate_query()'s return value, not
  by relying on guardrails.py's own (incomplete) internal logging
- Guarantee exactly one request.completed per request, including on an
  unhandled exception (M5) — timing/observability infrastructure never
  swallows an exception, only observes it and re-raises
- Validate request schemas
- Enforce input guardrails (safety/domain) before delegating
- Delegate execution to application services

This module intentionally contains no domain reasoning logic — only
request validation, guardrail enforcement, and event emission live here.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.models import QueryRequest, RAGRequest, AnalyticsRequest
from src.orchestrator import run_query
from src.rag.rag_chain import run_rag
from src.analytics.fraud_analytics import run_analytics
from src.safety.guardrails import validate_query
from src.llm.response_schema import ErrorResponse
from src.observability.context import new_request_id, set_request_id
from src.observability.cost import reset_request_cost
from src.observability.events import emit_event, query_hash
from src.observability.timing import elapsed_timer

router = APIRouter(prefix="", tags=["api"])


def _run_guardrails(route: str, query: str):
    """
    Shared guardrail + event-emission sequence for all three endpoints.

    Emits, in order: request.started, then guardrails.completed or
    guardrails.blocked — timed around just the validate_query() call, not
    the request.started emission itself. Returns the same 4-tuple
    validate_query() does, so each endpoint's existing branching logic is
    unchanged.

    Not built on observe_step(): guardrails has 2 outcomes decided by a
    boolean, not by whether an exception occurred, so it doesn't fit that
    helper's binary success/failure shape (see timing.py).
    """
    emit_event("request.started", step="request", status="success", metadata={"route": route})

    with elapsed_timer() as elapsed:
        ok, cleaned_or_msg, lang, reason = validate_query(query)
        emit_event(
            "guardrails.blocked" if not ok else "guardrails.completed",
            step="guardrails",
            status="blocked" if not ok else "success",
            duration_ms=elapsed(),
            metadata={
                "blocked": not ok,
                "reason": reason,
                "query_length": len(query or ""),
                "query_hash": query_hash(query or ""),
            },
        )

    return ok, cleaned_or_msg, lang, reason


@router.post("/query", summary="Run unified AI query")
async def query_endpoint(req: QueryRequest):
    """
    Execute the unified AI orchestrator.

    This endpoint is typically used for:
    - High-level questions
    - Intelligent routing between RAG and analytics flows
    """
    set_request_id(new_request_id())
    reset_request_cost()

    with elapsed_timer() as elapsed:
        try:
            ok, cleaned_or_msg, lang, _reason = _run_guardrails("/query", req.query)
            if not ok:
                emit_event(
                    "request.completed", step="request", status="blocked",
                    duration_ms=elapsed(), metadata={"route": "/query"},
                )
                return {
                    "query": req.query,
                    "intent": "reject",
                    "error": None,
                    "result": {"type": "reject", "message": cleaned_or_msg},
                }

            result = run_query(cleaned_or_msg, detected_lang=lang)
            emit_event(
                "request.completed", step="request", status="success",
                duration_ms=elapsed(), metadata={"route": "/query"},
            )
            return result
        except Exception as e:
            emit_event(
                "request.completed", step="request", status="error",
                duration_ms=elapsed(),
                metadata={"route": "/query", "error_type": type(e).__name__},
            )
            raise


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
    set_request_id(new_request_id())
    reset_request_cost()

    with elapsed_timer() as elapsed:
        try:
            ok, cleaned_or_msg, _lang, _reason = _run_guardrails("/rag", req.query)
            if not ok:
                emit_event(
                    "request.completed", step="request", status="blocked",
                    duration_ms=elapsed(), metadata={"route": "/rag"},
                )
                return JSONResponse(
                    status_code=400,
                    content=ErrorResponse(error=cleaned_or_msg).model_dump(),
                )

            result = run_rag(
                query_en=cleaned_or_msg,
                user_lang=req.lang,
            )
            emit_event(
                "request.completed", step="request", status="success",
                duration_ms=elapsed(), metadata={"route": "/rag"},
            )
            return result
        except Exception as e:
            emit_event(
                "request.completed", step="request", status="error",
                duration_ms=elapsed(),
                metadata={"route": "/rag", "error_type": type(e).__name__},
            )
            raise


@router.post("/analytics", summary="Run fraud analytics query")
async def analytics_endpoint(req: AnalyticsRequest):
    """
    Execute fraud analytics logic.

    This endpoint is optimized for:
    - Pattern detection
    - Risk insights
    - Analytical reasoning over structured signals
    """
    set_request_id(new_request_id())
    reset_request_cost()

    with elapsed_timer() as elapsed:
        try:
            ok, cleaned_or_msg, _lang, _reason = _run_guardrails("/analytics", req.query)
            if not ok:
                emit_event(
                    "request.completed", step="request", status="blocked",
                    duration_ms=elapsed(), metadata={"route": "/analytics"},
                )
                return JSONResponse(
                    status_code=400,
                    content=ErrorResponse(error=cleaned_or_msg).model_dump(),
                )

            result = run_analytics(
                cleaned_or_msg,
                lang=req.lang,
            )
            emit_event(
                "request.completed", step="request", status="success",
                duration_ms=elapsed(), metadata={"route": "/analytics"},
            )
            return result
        except Exception as e:
            emit_event(
                "request.completed", step="request", status="error",
                duration_ms=elapsed(),
                metadata={"route": "/analytics", "error_type": type(e).__name__},
            )
            raise
