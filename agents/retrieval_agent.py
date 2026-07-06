"""
Retrieval Agent — hybrid dense (ColPali page-image embeddings) + sparse (BM25).

ColPali embeds entire document PAGE IMAGES → no OCR needed.
This is the core technical differentiator of DocuSage.

References:
  - ColPali paper: https://arxiv.org/abs/2407.01449
  - colpali-engine: pip install colpali-engine
"""
import os
import uuid
from typing import List
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from .state import DocuSageState, DocumentChunk


COLLECTION_NAME = "docusage_pages"
EMBEDDING_DIM   = 128  # ColQwen2 produces 128-dim patch embeddings (multi-vector)


class RetrievalAgent:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        # ColPali/ColQwen model — loaded lazily to save memory
        self._colpali_model = None

    def _get_colpali(self):
        """Lazy load ColQwen2 model (downloads ~5GB on first run)."""
        if self._colpali_model is None:
            from colpali_engine.models import ColQwen2, ColQwen2Processor
            self._colpali_model = ColQwen2.from_pretrained(
                "vidore/colqwen2-v1.0",
                torch_dtype="auto",
                device_map="auto",
            )
        return self._colpali_model

    def ensure_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info(f"[RetrievalAgent] Created Qdrant collection: {COLLECTION_NAME}")

    def run(self, state: DocuSageState) -> DocuSageState:
        logger.info(f"[RetrievalAgent] Retrieving for query type: {state.query_type}")

        try:
            # TODO Week 4: implement full ColPali query embedding + Qdrant search
            # For now, returns placeholder so the graph runs end-to-end
            state.retrieved_chunks = self._mock_retrieve(state.user_query)
        except Exception as e:
            logger.error(f"[RetrievalAgent] Error: {e}")
            state.error = str(e)

        return state

    def _mock_retrieve(self, query: str) -> List[DocumentChunk]:
        """Placeholder — replace with real ColPali + Qdrant in Week 4."""
        return [
            DocumentChunk(
                page_number=1,
                content=f"[MOCK] Retrieved chunk for: {query[:50]}",
                image_path=None,
                source_doc="RBI_Circular_2024_03.pdf",
                chunk_id=str(uuid.uuid4()),
                score=0.91,
            )
        ]
