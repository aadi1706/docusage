# Week 3 Devlog — Hybrid Retrieval + Real Documents

## What I did
- Added BM25Okapi sparse retrieval alongside ColQwen2 dense retrieval
- Implemented Reciprocal Rank Fusion (RRF) to merge both rankings
- Fixed division by zero bug for empty page text
- Ingested 92-page BIS monetary policy paper — 135 total docs in Qdrant
- Real answers with citations working on actual financial documents
- Added Langfuse v4 tracing — traces visible at cloud.langfuse.com

## Key decisions
- RRF_K=60 (standard default, no tuning needed)
- batch_size=1 for Qdrant upsert (multi-vectors too large otherwise)
- Langfuse non-fatal — tracing errors never break main pipeline

## Results
- Query: "What is the impact of monetary policy on inflation?"
- Got real answer citing BIS_Monetary_Policy_2024 Pages 28 and 60
- Hallucination verifier correctly flagged page numbers as unverified

## Next (Week 7)
- RAGAS eval harness with real golden Q&A pairs
- GitHub Actions eval gate
