# Deploying DocuSage to Render

## Prerequisites

| Service | What you need |
|---|---|
| [Qdrant Cloud](https://cloud.qdrant.io) | Free cluster URL + API key (already has your 135 ingested docs) |
| [OpenAI](https://platform.openai.com) | API key with access to `gpt-4o-mini` |
| [Langfuse](https://cloud.langfuse.com) | Project public key + secret key (for tracing) |

---

## Step 1 — Fork or clone the repo

```bash
git clone https://github.com/aadi1706/docusage.git
```

Or fork it on GitHub so Render can connect to your own copy.

---

## Step 2 — Create a new Blueprint on Render

1. Go to [render.com](https://render.com) → **New** → **Blueprint**
2. Connect your GitHub account and select the `docusage` repository
3. Render will detect `render.yaml` automatically and propose a `docusage-api` web service

---

## Step 3 — Set environment variables in the Render dashboard

Before deploying, set these 5 variables under **Environment**:

| Variable | Where to find it |
|---|---|
| `OPENAI_API_KEY` | platform.openai.com → API keys |
| `QDRANT_URL` | Qdrant Cloud dashboard → Cluster URL (e.g. `https://xyz.qdrant.io:6333`) |
| `QDRANT_API_KEY` | Qdrant Cloud dashboard → API keys |
| `LANGFUSE_PUBLIC_KEY` | Langfuse project → Settings → API keys |
| `LANGFUSE_SECRET_KEY` | Langfuse project → Settings → API keys |

---

## Step 4 — Deploy

Click **Apply** / **Deploy**. The first build takes **~8–12 minutes** because:
- `torch==2.5.1` is ~800 MB
- `colpali-engine` and `transformers` add another ~2 GB of model weights on first cold start

Subsequent deploys are faster (Docker layer cache).

---

## Step 5 — Verify the deployment

```bash
curl https://<your-app>.onrender.com/health
# Expected: {"status":"ok","service":"docusage-api"}
```

```bash
curl -X POST https://<your-app>.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the repo rate in RBI Monetary Policy 2024?"}'
```

---

## Step 6 — Explore the API

Open `https://<your-app>.onrender.com/docs` for the auto-generated Swagger UI.

---

## Performance notes

- **Free tier (CPU):** ColQwen2 dense retrieval runs on CPU → ~10–15 seconds per query
- **Render Starter GPU (or equivalent):** Reduces dense retrieval to ~1–2 seconds per query
- BM25 sparse retrieval is instant on either tier — only the ColQwen2 embedding step is slow on CPU

To upgrade, change `plan: free` → `plan: starter` (or add a GPU instance) in `render.yaml`.
