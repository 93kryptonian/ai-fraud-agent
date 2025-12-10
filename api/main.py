    # api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router
import os

app = FastAPI(
    title="AI Fraud Agents Public API",
    description="RAG + Analytics AI Engine for Fraud Intelligence",
    version="1.0.0"
)

# Allow all origins because it's public portfolio demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
