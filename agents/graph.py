"""
DocuSage — LangGraph orchestration graph.

Flow:
  user_query
      → router        (classify query type)
      → retrieval     (ColPali visual + BM25 hybrid search)
      → extraction    (VLM table/chart extraction, if image pages found)
      → synthesis     (LLM answer generation with citations)
      → verification  (numeric hallucination check)
      → END

Langfuse tracing wraps every node for full observability.
"""
import time
import uuid
import os
from loguru import logger
from langgraph.graph import StateGraph, END

from .state import DocuSageState
from .router_agent import RouterAgent
from .retrieval_agent import RetrievalAgent
from .extraction_agent import ExtractionAgent
from .synthesis_agent import SynthesisAgent
from .verification_agent import VerificationAgent


# ── Instantiate agents (singletons for the graph lifetime) ──────────────────
router      = RouterAgent()
retrieval   = RetrievalAgent()
extraction  = ExtractionAgent()
synthesis   = SynthesisAgent()
verification = VerificationAgent()


# ── Node functions (LangGraph calls these) ───────────────────────────────────
def node_router(state: dict) -> dict:
    s = DocuSageState(**state)
    s = router.run(s)
    return s.model_dump()

def node_retrieval(state: dict) -> dict:
    s = DocuSageState(**state)
    s = retrieval.run(s)
    return s.model_dump()

def node_extraction(state: dict) -> dict:
    s = DocuSageState(**state)
    s = extraction.run(s)
    return s.model_dump()

def node_synthesis(state: dict) -> dict:
    s = DocuSageState(**state)
    s = synthesis.run(s)
    return s.model_dump()

def node_verification(state: dict) -> dict:
    s = DocuSageState(**state)
    s = verification.run(s)
    return s.model_dump()


# ── Conditional edge: should we run extraction? ──────────────────────────────
def should_extract(state: dict) -> str:
    """Skip VLM extraction for pure text queries — saves latency + cost."""
    query_type = state.get("query_type", "hybrid")
    if query_type == "text":
        return "synthesis"
    return "extraction"


# ── Build the graph ──────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    g = StateGraph(dict)

    g.add_node("router",       node_router)
    g.add_node("retrieval",    node_retrieval)
    g.add_node("extraction",   node_extraction)
    g.add_node("synthesis",    node_synthesis)
    g.add_node("verification", node_verification)

    g.set_entry_point("router")

    g.add_edge("router", "retrieval")

    # Conditional: text-only queries skip VLM extraction
    g.add_conditional_edges(
        "retrieval",
        should_extract,
        {
            "extraction": "extraction",
            "synthesis":  "synthesis",
        },
    )

    g.add_edge("extraction",   "synthesis")
    g.add_edge("synthesis",    "verification")
    g.add_edge("verification", END)

    return g.compile()


# ── Public entry point ───────────────────────────────────────────────────────
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(query: str, session_id: str = None) -> DocuSageState:
    """Run a query through the full DocuSage agent graph."""
    if session_id is None:
        session_id = str(uuid.uuid4())

    start = time.time()
    initial_state = DocuSageState(
        user_query=query,
        session_id=session_id,
    ).model_dump()

    try:
        result = get_graph().invoke(initial_state)
        result["latency_ms"] = round((time.time() - start) * 1000, 2)
        logger.info(f"[Graph] Query completed in {result['latency_ms']}ms")
        return DocuSageState(**result)
    except Exception as e:
        logger.error(f"[Graph] Unhandled error: {e}")
        raise

# Langfuse tracing — called after every query
def trace_result(query: str, result: DocuSageState):
    try:
        from agents.langfuse_tracer import trace_query
        trace_id = trace_query(query, result, result.latency_ms or 0)
        if trace_id:
            logger.info(f"[Graph] Langfuse trace ID: {trace_id}")
    except Exception as e:
        logger.warning(f"[Graph] Tracing failed (non-fatal): {e}")
