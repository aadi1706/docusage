"""
Re-embed all chunks from docusage_pages (ColQwen2 vectors) into
docusage_pages_lightweight (text-embedding-3-small, dim=1536).

Usage:
    env $(cat .env | grep -v '^#' | grep -v '^$' | xargs) \
        .venv/bin/python scripts/reindex_lightweight.py
"""
import os
import time
import uuid

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from loguru import logger

SOURCE_COLLECTION = "docusage_pages"
TARGET_COLLECTION = "docusage_pages_lightweight"
VECTOR_SIZE = 1536          # text-embedding-3-small
EMBED_MODEL = "text-embedding-3-small"
SCROLL_BATCH = 100          # points per scroll page
UPSERT_BATCH = 50           # points per upsert call
RATE_LIMIT_DELAY = 0.05     # seconds between OpenAI calls


def get_clients():
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
        timeout=120,
    )
    openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return qdrant, openai


def ensure_target_collection(qdrant: QdrantClient):
    existing = {c.name for c in qdrant.get_collections().collections}
    if TARGET_COLLECTION in existing:
        logger.info(f"Collection '{TARGET_COLLECTION}' already exists — will upsert into it")
    else:
        qdrant.create_collection(
            collection_name=TARGET_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created collection '{TARGET_COLLECTION}' (dim={VECTOR_SIZE}, cosine)")


def embed_texts(openai: OpenAI, texts: list[str]) -> list[list[float]]:
    # OpenAI batches up to 2048 inputs; we keep batches small for rate-limit safety
    resp = openai.embeddings.create(input=texts, model=EMBED_MODEL)
    return [item.embedding for item in resp.data]


def reindex(qdrant: QdrantClient, openai: OpenAI):
    ensure_target_collection(qdrant)

    offset = None
    total_processed = 0

    while True:
        scroll_result = qdrant.scroll(
            collection_name=SOURCE_COLLECTION,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = scroll_result

        if not points:
            break

        # Split into upsert-sized batches
        for batch_start in range(0, len(points), UPSERT_BATCH):
            batch = points[batch_start : batch_start + UPSERT_BATCH]

            texts = [p.payload.get("text", "") or "" for p in batch]
            # Replace empty text with a space so the API doesn't error
            texts_for_embed = [t if t.strip() else " " for t in texts]

            try:
                embeddings = embed_texts(openai, texts_for_embed)
            except Exception as e:
                logger.error(f"OpenAI embed error on batch starting {total_processed}: {e}")
                raise

            new_points = []
            for point, embedding in zip(batch, embeddings):
                new_points.append(
                    PointStruct(
                        id=str(point.id) if not isinstance(point.id, (int, str)) else point.id,
                        vector=embedding,
                        payload={
                            "text": point.payload.get("text", ""),
                            "source_doc": point.payload.get("source_doc", "unknown"),
                            "page_number": point.payload.get("page_number", 0),
                            "image_path": point.payload.get("image_path"),
                        },
                    )
                )

            qdrant.upsert(collection_name=TARGET_COLLECTION, points=new_points)
            total_processed += len(batch)
            logger.info(f"Upserted {total_processed} points so far...")

            time.sleep(RATE_LIMIT_DELAY)

        if next_offset is None:
            break
        offset = next_offset

    logger.success(f"Reindex complete — {total_processed} points in '{TARGET_COLLECTION}'")
    return total_processed


if __name__ == "__main__":
    qdrant, openai_client = get_clients()

    # Verify source collection exists
    existing = {c.name for c in qdrant.get_collections().collections}
    if SOURCE_COLLECTION not in existing:
        logger.error(f"Source collection '{SOURCE_COLLECTION}' not found. Available: {existing}")
        raise SystemExit(1)

    info = qdrant.get_collection(SOURCE_COLLECTION)
    logger.info(f"Source collection: {SOURCE_COLLECTION} — {info.points_count} points")

    reindex(qdrant, openai_client)
