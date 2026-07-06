# ADR 003 — Why Hybrid Retrieval (Dense + Sparse) instead of Pure Vector Search

**Date:** Week 2  
**Status:** Accepted

## Context
Financial documents contain both dense semantic content ("explain the Basel III norms") AND exact keyword queries ("repo rate 6.50%", "Circular No. RBI/2024/03"). Pure vector search fails on exact number lookups.

## Decision
Use **hybrid retrieval**: ColPali dense embeddings + BM25 sparse retrieval, fused with Reciprocal Rank Fusion (RRF).

## Rationale
| Query type | Pure vector | Pure BM25 | Hybrid |
|---|---|---|---|
| "Explain liquidity coverage ratio" | ✓ | ✗ | ✓ |
| "What is the exact figure on page 12, table 3?" | ✗ | ✓ | ✓ |
| "RBI circular dated 15 March 2024" | ✗ | ✓ | ✓ |
| "Risk-weighted assets similar to Basel" | ✓ | ✗ | ✓ |

## Implementation
- Dense: ColQwen2 embeddings → Qdrant ANN search
- Sparse: BM25 via `rank_bm25` over PostgreSQL stored text
- Fusion: Reciprocal Rank Fusion (RRF) — simple, parameter-free, effective
