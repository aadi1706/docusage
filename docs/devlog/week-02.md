# Week 2 Devlog — Real ColQwen2 Retrieval Pipeline

## What I did
- Built real PDF ingestion pipeline using pdf2image + ColQwen2 + Qdrant
- Replaced mock retrieval with real ColQwen2 query embedding + Qdrant ANN search
- Ingested 33 pages from BIS Working Paper into Qdrant cloud
- Full pipeline working end to end: Router → Retrieval → Synthesis → Verification
- Added daily cron job to keep Qdrant free cluster alive

## Key decisions made
- batch_size=1 for Qdrant upsert (multi-vector embeddings too large for bigger batches)
- timeout=300 to handle slow uploads to Qdrant cloud
- device_map="cpu" for ColQwen2 on M1 (MPS has compatibility issues with ColQwen2)
- Permanent ~/.docusage_env backup so keys survive terminal sessions

## Results
- 33 pages indexed in Qdrant ✓
- Real query retrieves 3 relevant chunks ✓
- End to end latency: ~32s (cold ColQwen2 load dominates)

## What's next (Week 3)
- Add BM25 text search alongside ColQwen2 dense search
- Implement Reciprocal Rank Fusion (RRF) to merge both rankings
- Download real RBI circulars and ingest them

## Blockers encountered
- Qdrant free tier pauses after 7 days inactivity — solved with daily cron ping
- numpy version conflict during pip install — solved by unpinning versions
- openai-whisper pkg_resources error — solved by installing from GitHub directly
