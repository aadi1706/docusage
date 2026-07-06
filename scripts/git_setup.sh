#!/bin/bash
# Run this ONCE to initialize your GitHub repo and push the first commit.
# Replace YOUR_USERNAME and YOUR_REPO_NAME below.

set -e

YOUR_USERNAME="YOUR_GITHUB_USERNAME"
YOUR_REPO_NAME="docusage"

echo "=== DocuSage GitHub Setup ==="

# 1. Init git
git init
git add .
git commit -m "feat: initial scaffold — LangGraph agent graph, FastAPI, eval harness, Docker

- 4-agent LangGraph graph: Router → Retrieval → Extraction → Synthesis → Verification
- ColPali/ColQwen2 retrieval agent stub (Week 3: wire real embeddings)
- Qwen2-VL extraction agent stub (Week 5: wire real VLM calls)
- Numeric hallucination verification agent
- FastAPI backend with /query and /ingest endpoints
- RAGAS eval harness with golden set template
- GitHub Actions eval gate (blocks merge if RAGAS thresholds fail)
- docker-compose: Qdrant + PostgreSQL + Langfuse + API
- Architecture Decision Records: ADR 001-003
- Week 1 devlog

Closes #1"

# 2. Create GitHub repo (requires GitHub CLI: brew install gh)
# gh repo create $YOUR_REPO_NAME --public --description "Agentic Multi-Modal RAG for Indian Financial Documents"

# 3. Push
# git remote add origin https://github.com/$YOUR_USERNAME/$YOUR_REPO_NAME.git
# git branch -M main
# git push -u origin main

echo ""
echo "✓ Local repo initialized with first commit."
echo ""
echo "Next steps:"
echo "  1. Create repo on github.com/new (name: $YOUR_REPO_NAME)"
echo "  2. Run: git remote add origin https://github.com/$YOUR_USERNAME/$YOUR_REPO_NAME.git"
echo "  3. Run: git branch -M main && git push -u origin main"
echo "  4. Add secrets in GitHub repo settings:"
echo "       OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY,"
echo "       LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY"
echo "  5. Create a GitHub Project board and add the 12-week issues"
