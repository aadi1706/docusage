"""
Retrieval Agent — real ColQwen2 + Qdrant hybrid search.
Replaces the mock retrieval from Week 1.
"""
import os
import uuid
import torch
from typing import List
from loguru import logger
from qdrant_client import QdrantClient
from .state import DocuSageState, DocumentChunk

COLLECTION_NAME = "docusage_pages"

_model = None
_processor = None

def get_colqwen2():
    global _model, _processor
    if _model is None:
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        _model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        _processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
    return _model, _processor


class RetrievalAgent:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=120,
        )

    def _embed_query(self, query: str) -> list:
        model, processor = get_colqwen2()
        batch = processor.process_queries([query]).to(model.device)
        with torch.no_grad():
            embeddings = model(**batch)
        return embeddings[0].cpu().float().numpy().tolist()

    def run(self, state: DocuSageState) -> DocuSageState:
        logger.info(f"[RetrievalAgent] Querying Qdrant: {state.user_query[:60]}")
        try:
            query_embedding = self._embed_query(state.user_query)
            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=3,
            )
            chunks = []
            for r in results.points:
                chunks.append(DocumentChunk(
                    page_number=r.payload.get("page_number", 0),
                    content=r.payload.get("text", ""),
                    image_path=r.payload.get("image_path"),
                    source_doc=r.payload.get("source_doc", "unknown"),
                    chunk_id=str(r.id),
                    score=r.score,
                ))
            state.retrieved_chunks = chunks
            logger.info(f"[RetrievalAgent] Retrieved {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"[RetrievalAgent] Error: {e}")
            state.error = str(e)
        return state
