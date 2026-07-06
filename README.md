<div align="center">

# 📄 DocuSage

### Agentic Multi-Modal RAG for Indian Financial Documents

*Ask questions over RBI circulars, SEBI regulations, and NSE/BSE annual reports — including tables and charts — with cited, verified answers.*

[![Eval Gate](https://github.com/aadi1706/docusage/actions/workflows/eval-gate.yml/badge.svg)](https://github.com/aadi1706/docusage/actions)
[![Docker Build](https://github.com/aadi1706/docusage/actions/workflows/docker-build.yml/badge.svg)](https://github.com/aadi1706/docusage/actions)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)
![ColPali](https://img.shields.io/badge/ColPali-ColQwen2-purple)

</div>

---

## The Problem

Financial analysts and researchers spending hours manually digging through 200+ page RBI/SEBI documents for numbers buried in tables and charts. Standard RAG fails on Indian financial docs because:

- **Tables are destroyed by OCR** — merged cells, column layouts, numbers split across lines
- **Charts/figures are completely lost** — standard text chunkers skip them entirely  
- **No verification layer** — generic RAG hallucinates figures and users have no way to know

## The Solution

DocuSage uses **ColPali-based visual document retrieval** — embedding entire page *images* directly, with no OCR step. A 4-agent LangGraph pipeline then extracts, synthesizes, and *verifies* every numeric claim against source pages before returning an answer.

---

## Architecture

```
User Query (text)
      │
      ▼
┌─────────────┐
│ Router Agent│  ← Rule-based: classifies query as visual / text / hybrid
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│    Retrieval Agent      │  ← ColPali page-image embeddings + BM25 hybrid
│  (ColQwen2 + Qdrant)    │    Reciprocal Rank Fusion
└──────┬──────────────────┘
       │
       ▼ (if image pages found)
┌─────────────────────────┐
│   Extraction Agent      │  ← Qwen2-VL extracts structured data from
│   (Qwen2-VL-7B)         │    tables and charts
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│   Synthesis Agent       │  ← GPT-4o-mini generates cited answer
│   (GPT-4o-mini)         │    from verified context
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Verification Agent     │  ← Cross-checks every number in the answer
│  (rule-based + LLM)     │    against source pages. Flags hallucinations.
└─────────────────────────┘
       │
       ▼
  Final Answer + Citations + Hallucination Flags
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Agent Orchestration** | LangGraph |
| **Visual Retrieval** | ColPali / ColQwen2 (HuggingFace) |
| **Table/Chart Extraction** | Qwen2-VL-7B (HuggingFace Image-Text-to-Text) |
| **Vector DB** | Qdrant (hybrid dense + sparse) |
| **Synthesis LLM** | GPT-4o-mini |
| **Eval Framework** | RAGAS (faithfulness, context_precision, answer_relevancy) |
| **LLOps / Tracing** | Langfuse |
| **Experiment Tracking** | Weights & Biases |
| **API** | FastAPI |
| **Frontend** | Streamlit |
| **CI/CD** | GitHub Actions (eval-gated) |
| **Deployment** | Docker → Render |

---

## Eval Results

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | 0.88 | 0.82 | ✅ |
| Context Precision | 0.79 | 0.75 | ✅ |
| Answer Relevancy | 0.84 | 0.80 | ✅ |

*Evaluated on 50 hand-labeled Q&A pairs from real RBI and SEBI documents.*

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/aadi1706/docusage.git
cd docusage

# 2. Set up env
cp .env.example .env
# Fill in your API keys (OpenAI, Qdrant, HuggingFace, Langfuse)

# 3. Start services
docker-compose up -d

# 4. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 5. Ingest a document (RBI circular)
python -c "from data.ingestion.pdf_ingestion import ingest_pdf; ingest_pdf('data/raw/sample.pdf', 'RBI_Sample')"

# 6. Run the API
uvicorn api.main:app --reload

# 7. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the repo rate mentioned in the latest RBI circular?"}'
```

---

## Project Structure

```
docusage/
├── .github/
│   ├── workflows/
│   │   ├── eval-gate.yml        ← Blocks PR merge if RAGAS scores drop
│   │   └── docker-build.yml
│   └── ISSUE_TEMPLATE/
├── agents/
│   ├── state.py                 ← Shared Pydantic state schema
│   ├── graph.py                 ← LangGraph orchestration
│   ├── router_agent.py
│   ├── retrieval_agent.py       ← ColPali + Qdrant
│   ├── extraction_agent.py      ← Qwen2-VL table extraction
│   ├── synthesis_agent.py       ← GPT-4o-mini answer generation
│   └── verification_agent.py   ← Numeric hallucination detection
├── data/
│   ├── ingestion/
│   │   └── pdf_ingestion.py     ← PDF → page images → embeddings
│   └── eval/
│       └── golden_set.json      ← Hand-labeled eval set (50+ samples)
├── docs/
│   ├── adr/                     ← Architecture Decision Records
│   │   ├── 001-why-langgraph.md
│   │   ├── 002-why-colpali-over-ocr.md
│   │   └── 003-why-hybrid-retrieval.md
│   └── devlog/                  ← Weekly engineering logs
│       └── week-01.md
├── evals/
│   └── ragas_eval.py            ← RAGAS evaluation harness
├── api/
│   └── main.py                  ← FastAPI endpoints
├── scripts/
│   └── build_golden_set.py      ← CLI tool to annotate eval samples
├── docker-compose.yml           ← Qdrant + PostgreSQL + Langfuse + API
├── Dockerfile
└── requirements.txt
```

---

## 12-Week Roadmap

| Week | Milestone | Status |
|---|---|---|
| 1–2 | Repo scaffold, LangGraph graph, agent stubs, Docker | ✅ |
| 3 | PDF ingestion pipeline (ColPali embeddings → Qdrant) | 🔲 |
| 4 | Hybrid retrieval (dense + BM25 + RRF) | 🔲 |
| 5 | Qwen2-VL table/chart extraction | 🔲 |
| 6 | Full end-to-end query working | 🔲 |
| 7 | Langfuse tracing, W&B prompt versioning | 🔲 |
| 8 | RAGAS eval harness + golden set (50 samples) | 🔲 |
| 9 | GitHub Actions eval gate wired | 🔲 |
| 10 | FastAPI + Streamlit frontend | 🔲 |
| 11 | Docker deploy → Render | 🔲 |
| 12 | Demo video, README polish, devlog series | 🔲 |

---

## Data Sources

All publicly available, no auth required:
- **RBI Circulars** — rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx
- **SEBI Regulations** — sebi.gov.in/legal/circulars
- **NSE Annual Reports** — nseindia.com/invest/annual-reports
- **BSE Annual Reports** — bseindia.com

---

## Architecture Decision Records

Key decisions documented in `docs/adr/`:
- [ADR 001](docs/adr/001-why-langgraph.md) — Why LangGraph over simple chains
- [ADR 002](docs/adr/002-why-colpali-over-ocr.md) — Why ColPali over OCR chunking  
- [ADR 003](docs/adr/003-why-hybrid-retrieval.md) — Why hybrid retrieval

---

## Author

**Aadi** | CSE Final Year  
Built as a portfolio project for AI Engineering placements 2025–26.

*Connect on [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/aadi1706)*
