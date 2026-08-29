"""
Retrieval Agent — Hybrid ColQwen2 (dense) + BM25 (sparse) + RRF fusion.
Week 3 implementation.

Why hybrid?
- Dense (ColQwen2): great for semantic queries ("explain liquidity ratio")
- Sparse (BM25): great for exact matches ("repo rate 6.50%", "circular RBI/2024/03")
- RRF fusion: combines both rankings without needing to tune weights
"""
import os
import uuid
import torch
from typing import List
from loguru import logger
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from .state import DocuSageState, DocumentChunk

COLLECTION_NAME = "docusage_pages"
RRF_K = 60  # RRF constant — 60 is the standard default

_model = None
_processor = None


def get_colqwen2():
    global _model, _processor
    if _model is None:
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        logger.info("[RetrievalAgent] Loading ColQwen2...")
        _model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        _processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
        logger.info("[RetrievalAgent] ColQwen2 loaded ✓")
    return _model, _processor


def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    k: int = RRF_K
) -> list:
    """
    Merge dense and sparse rankings using Reciprocal Rank Fusion.
    RRF score = 1/(k + rank) summed across all result lists.
    Higher score = better combined rank.
    """
    scores = {}

    for rank, chunk in enumerate(dense_results):
        cid = chunk.chunk_id
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)

    for rank, chunk in enumerate(sparse_results):
        cid = chunk.chunk_id
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)

    # Build a lookup of chunk_id → chunk object
    all_chunks = {c.chunk_id: c for c in dense_results + sparse_results}

    # Sort by RRF score descending
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [all_chunks[cid] for cid in sorted_ids]


class RetrievalAgent:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=120,
        )
        self._bm25 = None
        self._bm25_chunks = []

    def _embed_query(self, query: str) -> list:
        model, processor = get_colqwen2()
        batch = processor.process_queries([query]).to(model.device)
        with torch.no_grad():
            embeddings = model(**batch)
        return embeddings[0].cpu().float().numpy().tolist()

    def _dense_search(self, query: str, limit: int = 5) -> List[DocumentChunk]:
        """ColQwen2 visual embedding search."""
        query_embedding = self._embed_query(query)
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
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
        return chunks

    def _build_bm25_index(self):
        """Build BM25 index from all text in Qdrant."""
        logger.info("[RetrievalAgent] Building BM25 index...")
        all_points = self.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=True,
            with_vectors=False,
        )[0]

        self._bm25_chunks = []
        corpus = []
        for point in all_points:
            text = point.payload.get("text", "")
            chunk = DocumentChunk(
                page_number=point.payload.get("page_number", 0),
                content=text,
                image_path=point.payload.get("image_path"),
                source_doc=point.payload.get("source_doc", "unknown"),
                chunk_id=str(point.id),
                score=0.0,
            )
            self._bm25_chunks.append(chunk)
            corpus.append(text.lower().split() if text.strip() else ["empty"])

        self._bm25 = BM25Okapi(corpus)
        logger.info(f"[RetrievalAgent] BM25 index built — {len(corpus)} docs")

    def _sparse_search(self, query: str, limit: int = 5) -> List[DocumentChunk]:
        """BM25 keyword search."""
        if self._bm25 is None:
            self._build_bm25_index()

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-limit indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:limit]

        results = []
        for idx in top_indices:
            chunk = self._bm25_chunks[idx]
            chunk.score = float(scores[idx])
            results.append(chunk)
        return results

    def run(self, state: DocuSageState) -> DocuSageState:
        logger.info(f"[RetrievalAgent] Hybrid search: {state.user_query[:60]}")
        try:
            # Dense search (ColQwen2)
            dense = self._dense_search(state.user_query, limit=5)
            logger.info(f"[RetrievalAgent] Dense: {len(dense)} results")

            # Sparse search (BM25)
            sparse = self._sparse_search(state.user_query, limit=5)
            logger.info(f"[RetrievalAgent] Sparse: {len(sparse)} results")

            # RRF fusion
            fused = reciprocal_rank_fusion(dense, sparse)[:3]
            logger.info(f"[RetrievalAgent] After RRF fusion: {len(fused)} chunks")

            state.retrieved_chunks = fused

        except Exception as e:
            logger.error(f"[RetrievalAgent] Error: {e}")
            state.error = str(e)

        return state
