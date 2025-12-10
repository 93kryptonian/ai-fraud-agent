# api/models.py
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

class RAGRequest(BaseModel):
    query: str
    lang: str = "en"

class AnalyticsRequest(BaseModel):
    query: str
    lang: str = "en"
