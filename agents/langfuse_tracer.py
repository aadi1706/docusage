"""
Langfuse tracing wrapper for DocuSage agents.
Compatible with Langfuse v4+
"""
import os
from langfuse import Langfuse

_client = None

def get_langfuse():
    global _client
    if _client is None:
        _client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    return _client


def trace_query(query: str, result, latency_ms: float):
    """Send a completed query trace to Langfuse."""
    try:
        lf = get_langfuse()
        with lf.start_as_current_observation(
            name="docusage-query",
            as_type="agent",
            input={"query": query},
            output={
                "answer": result.final_answer,
                "citations": result.citations,
                "verified": result.verified,
                "hallucination_flags": result.hallucination_flags,
                "query_type": result.query_type,
            },
            metadata={"latency_ms": latency_ms},
        ):
            trace_id = lf.get_current_trace_id()
        lf.flush()
        return trace_id
    except Exception as e:
        print(f"[Langfuse] Tracing error (non-fatal): {e}")
        return None
