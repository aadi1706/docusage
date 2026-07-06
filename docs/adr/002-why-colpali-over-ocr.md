# ADR 002 — Why ColPali page-image embeddings instead of OCR chunking

**Date:** Week 1  
**Status:** Accepted

## Context
DocuSage ingests RBI circulars, SEBI regulations, and annual reports. These documents contain:
- Complex multi-column tables (balance sheets, P&L statements)
- Figures and charts (often embedded as images in PDFs)
- Headers/footers that confuse chunking boundaries
- Scanned pages (especially older RBI circulars)

## Decision
Use **ColPali/ColQwen2** to embed entire page *images* directly — no OCR step.

## Rationale
Standard RAG pipeline:
```
PDF → OCR → text chunks → text embeddings → Qdrant
```
Failure modes:
- OCR mangles table structure (merged cells, numbers split across lines)
- Charts/figures are completely lost
- Column layouts produce garbled text

ColPali pipeline:
```
PDF → page images → ColQwen2 multi-vector image embeddings → Qdrant
```
Why it works:
- ColQwen2 embeds visual layout — a table looks different from a paragraph, and that difference is captured
- No OCR = no OCR errors
- Scanned pages work natively
- Published benchmark: ColPali outperforms OCR-RAG on the ViDoRe benchmark by a large margin

## Trade-offs
- Index build is slower (image rendering + VLM inference per page)
- Model is ~5GB (manageable on any GPU machine; use HF Inference Endpoint for dev)
- Query time is slightly higher than pure text retrieval

## Consequences
- `data/ingestion/pdf_ingestion.py` renders pages with pdf2image (poppler)
- ColQwen2 runs via HF Inference Endpoint in dev, local GPU in prod
- BM25 text search is kept as a fallback for pure text queries
