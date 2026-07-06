# Week 1 Devlog — Project Setup & Architecture

## What I did
- Designed the 4-agent LangGraph graph (Router → Retrieval → Extraction → Synthesis → Verification)
- Scaffolded the repo: agents, API, eval harness, Docker, GitHub Actions
- Wrote ADRs 001-003 documenting the three biggest architectural decisions
- Set up local dev environment: Qdrant + PostgreSQL via docker-compose

## Key decisions made
1. **ColPali over OCR** — financial docs have too many tables/charts for OCR to handle reliably (ADR 002)
2. **LangGraph over plain chains** — need conditional branching to skip VLM for text-only queries (ADR 001)
3. **India-specific data sources** — RBI.org.in and SEBI.gov.in are free, public, and produce more relevant interview talking points than SEC EDGAR

## What surprised me
- ColQwen2 produces *multi-vector* embeddings (one vector per image patch, not one per page) — Qdrant handles this natively with its "multi-vector" collection type. This will affect how I set up the collection in Week 3.

## What's next (Week 2)
- [ ] Implement RouterAgent keyword classification
- [ ] Wire VerificationAgent numeric regex logic
- [ ] Set up Langfuse locally and confirm traces appear
- [ ] Write first 10 golden eval samples from a real RBI circular

## Blockers
- None yet

## Useful links found this week
- ColPali paper: https://arxiv.org/abs/2407.01449
- Qdrant multi-vector docs: https://qdrant.tech/documentation/concepts/vectors/#multivectors
- LangGraph tutorial: https://langchain-ai.github.io/langgraph/tutorials/introduction/
