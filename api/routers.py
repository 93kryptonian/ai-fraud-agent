    # api/router.py
from fastapi import APIRouter
from api.models import QueryRequest, RAGRequest, AnalyticsRequest
from src.orchestrator import run_query
from src.rag.rag_chain import run_rag
from src.analytics.fraud_analytics import run_analytics

router = APIRouter()

@router.post("/query")
async def query(req: QueryRequest):
    return run_query(req.query)

@router.post("/rag")
async def rag(req: RAGRequest):
    return run_rag(query_en=req.query, user_lang=req.lang)

@router.post("/analytics")
async def analytics(req: AnalyticsRequest):
    return run_analytics(req.query, lang=req.lang)
