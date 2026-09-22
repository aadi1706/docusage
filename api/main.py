"""
DocuSage — FastAPI backend
Endpoints:
  POST /query          → run a query through the agent graph
  POST /ingest         → trigger document ingestion
  GET  /health         → health check
  GET  /docs           → auto-generated Swagger UI (FastAPI built-in)
"""
import time
import uuid
import os
import psutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from agents.graph import run_query

logger.info(f"Startup complete. PID={os.getpid()}")


app = FastAPI(
    title="DocuSage API",
    description="Agentic Multi-Modal RAG for Indian Financial Documents",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None

class QueryResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[str]
    verified: bool
    hallucination_flags: list[str]
    confidence_score: float | None
    latency_ms: float
    query_type: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "docusage-api"}


@app.get("/memory")
def memory():
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / 1024 / 1024
    return {
        "rss_mb": round(rss_mb, 1),
        "lightweight_mode": os.getenv("LIGHTWEIGHT_MODE", "false"),
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    logger.info(f"[API] /query session={session_id} query={req.query[:60]}")

    try:
        result = run_query(query=req.query, session_id=session_id)
    except Exception as e:
        logger.error(f"[API] Graph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        session_id=session_id,
        answer=result.final_answer or "No answer generated",
        citations=result.citations,
        verified=result.verified or False,
        hallucination_flags=result.hallucination_flags,
        confidence_score=result.confidence_score,
        latency_ms=result.latency_ms or 0.0,
        query_type=result.query_type or "unknown",
    )


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a PDF for ingestion into the vector store.
    TODO Week 3: wire to data/ingestion/pdf_ingestion.py
    """
    logger.info(f"[API] /ingest filename={file.filename}")
    return {
        "status": "queued",
        "filename": file.filename,
        "message": "Ingestion pipeline will be wired in Week 3",
    }
