"""
Lightweight PDF ingestion — text-embedding-3-small → docusage_pages_lightweight

Skips ColQwen2 (no GPU required). Extracts text per page with pdfplumber,
embeds with OpenAI text-embedding-3-small, and upserts into Qdrant.

Usage:
    env $(cat .env | grep -v '^#' | grep -v '^$' | xargs) \
        .venv/bin/python scripts/ingest_lightweight.py \
        --pdf data/raw/RBI_MPC_Aug2024.pdf --name RBI_MPC_Aug2024 \
        --pdf data/raw/SEBI_AnnualReport_2023_24.pdf --name SEBI_AnnualReport_2023_24
"""
import argparse
import os
import time
import uuid
from pathlib import Path

import pdfplumber
from loguru import logger
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION   = "docusage_pages_lightweight"
EMBED_MODEL  = "text-embedding-3-small"
VECTOR_SIZE  = 1536
UPSERT_BATCH = 50
RATE_DELAY   = 0.05


def get_clients():
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
        timeout=120,
    )
    openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return qdrant, openai


def ensure_collection(qdrant: QdrantClient):
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created collection '{COLLECTION}'")
    else:
        logger.info(f"Collection '{COLLECTION}' already exists — will upsert")


def embed_texts(openai: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = openai.embeddings.create(input=texts, model=EMBED_MODEL)
    return [item.embedding for item in resp.data]


def ingest_pdf(pdf_path: str, source_name: str, qdrant: QdrantClient, openai: OpenAI) -> int:
    pdf_path = Path(pdf_path)
    logger.info(f"Ingesting '{source_name}' from {pdf_path}")

    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(pdf.pages)]

    logger.info(f"  Extracted {len(pages)} pages")

    points = []
    for page_num, text in pages:
        points.append({
            "page_number": page_num,
            "text": text[:4000],
            "source_doc": source_name,
            "image_path": None,
        })

    # Embed and upsert in batches
    total = 0
    for start in range(0, len(points), UPSERT_BATCH):
        batch = points[start : start + UPSERT_BATCH]
        texts = [p["text"] if p["text"].strip() else " " for p in batch]

        embeddings = embed_texts(openai, texts)

        qdrant_points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload=pt,
            )
            for pt, emb in zip(batch, embeddings)
        ]

        qdrant.upsert(collection_name=COLLECTION, points=qdrant_points)
        total += len(batch)
        logger.info(f"  Upserted {total}/{len(points)} pages for '{source_name}'")
        time.sleep(RATE_DELAY)

    logger.success(f"✓ Done — {total} pages indexed for '{source_name}'")
    return total


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into docusage_pages_lightweight")
    parser.add_argument("--pdf", action="append", required=True, help="Path to PDF")
    parser.add_argument("--name", action="append", required=True,
                        help="Source document name (parallel to --pdf)")
    args = parser.parse_args()

    if len(args.pdf) != len(args.name):
        parser.error("Each --pdf must have a matching --name")

    qdrant, openai_client = get_clients()
    ensure_collection(qdrant)

    grand_total = 0
    for pdf_path, name in zip(args.pdf, args.name):
        n = ingest_pdf(pdf_path, name, qdrant, openai_client)
        grand_total += n

    # Final count
    info = qdrant.get_collection(COLLECTION)
    logger.success(
        f"Ingestion complete — added {grand_total} pages. "
        f"Total points in '{COLLECTION}': {info.points_count}"
    )
    return grand_total


if __name__ == "__main__":
    main()
