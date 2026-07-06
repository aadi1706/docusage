"""
Shared LangGraph state schema — passed between all agents in the graph.
Every field is Optional so agents can incrementally populate it.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class DocumentChunk(BaseModel):
    page_number: int
    content: str                  # extracted text (may be empty for image-only pages)
    image_path: Optional[str]     # path to page image for visual retrieval
    source_doc: str               # e.g. "RBI_Circular_2024_03.pdf"
    chunk_id: str
    score: Optional[float] = None # retrieval similarity score


class DocuSageState(BaseModel):
    # Input
    user_query: str
    session_id: str

    # Router output
    query_type: Optional[str] = None      # "visual", "text", "hybrid"
    page_type_hint: Optional[str] = None  # "table", "chart", "text"

    # Retrieval output
    retrieved_chunks: List[DocumentChunk] = []

    # Extraction output (VLM)
    extracted_data: Optional[Dict[str, Any]] = None  # structured data from tables/charts

    # Verification output
    verified: Optional[bool] = None
    verification_notes: Optional[str] = None
    hallucination_flags: List[str] = []

    # Synthesis output
    final_answer: Optional[str] = None
    citations: List[str] = []
    confidence_score: Optional[float] = None

    # Metadata
    latency_ms: Optional[float] = None
    error: Optional[str] = None
