"""
PDF Ingestion Pipeline — Week 2 implementation.

Flow:
  1. PDF → page images (pdf2image + poppler)
  2. Page images → ColQwen2 multi-vector embeddings
  3. Embeddings → Qdrant collection
  4. Raw text → extracted via pdfplumber (BM25 fallback)
"""
import os
import uuid
from pathlib import Path
from loguru import logger
from pdf2image import convert_from_path
import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    MultiVectorConfig, MultiVectorComparator
)
from PIL import Image


# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "docusage_pages"
DPI             = 150       # page render resolution
IMG_DIR         = Path("data/raw/images")
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")


# ── Qdrant client ─────────────────────────────────────────────────────────────
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)


# ── ColQwen2 model (lazy load) ────────────────────────────────────────────────
_model = None
_processor = None

def get_colqwen2():
    global _model, _processor
    if _model is None:
        logger.info("Loading ColQwen2 model (first run: ~3GB download)...")
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        import torch
        _model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.float32,
            device_map="cpu",  # M1: use CPU, MPS has issues with ColQwen2
        )
        _processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
        logger.info("ColQwen2 loaded ✓")
    return _model, _processor


# ── Ensure Qdrant collection exists ───────────────────────────────────────────
def ensure_collection(client: QdrantClient, embedding_dim: int = 128):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                ),
            ),
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
    else:
        logger.info(f"Collection {COLLECTION_NAME} already exists")


# ── Embed a single page image ─────────────────────────────────────────────────
def embed_page(image: Image.Image) -> list:
    import torch
    model, processor = get_colqwen2()
    batch = processor.process_images([image]).to(model.device)
    with torch.no_grad():
        embeddings = model(**batch)
    # Returns list of patch vectors (multi-vector)
    return embeddings[0].cpu().float().numpy().tolist()


# ── Main ingestion function ───────────────────────────────────────────────────
def ingest_pdf(pdf_path: str, source_name: str) -> dict:
    """
    Ingest a PDF into Qdrant.
    
    Args:
        pdf_path:    Local path to PDF
        source_name: Human-readable name e.g. 'RBI_Circular_2024_03'
    
    Returns:
        dict with ingestion stats
    """
    logger.info(f"Starting ingestion: {source_name}")
    pdf_path = Path(pdf_path)

    # 1. Render pages as images
    logger.info("Rendering PDF pages...")
    pages = convert_from_path(str(pdf_path), dpi=DPI)
    logger.info(f"Rendered {len(pages)} pages")

    # 2. Save page images
    img_dir = IMG_DIR / source_name
    img_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for i, page in enumerate(pages):
        img_path = img_dir / f"page_{i+1}.png"
        page.save(str(img_path))
        image_paths.append(img_path)

    # 3. Extract text with pdfplumber (BM25 fallback)
    logger.info("Extracting text...")
    page_texts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            page_texts.append(text)

    # 4. Connect to Qdrant
    client = get_qdrant_client()

    # 5. Embed each page and upsert to Qdrant
    logger.info("Embedding pages with ColQwen2...")
    points = []
    for i, (image, text) in enumerate(zip(pages, page_texts)):
        logger.info(f"  Embedding page {i+1}/{len(pages)}...")
        embedding = embed_page(image)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "source_doc":   source_name,
                "page_number":  i + 1,
                "text":         text[:2000],  # store first 2000 chars
                "image_path":   str(image_paths[i]),
                "total_pages":  len(pages),
            },
        )
        points.append(point)

    # Ensure collection exists with correct dim
    if points:
        embedding_dim = len(points[0].vector[0])  # dim of first patch vector
        ensure_collection(client, embedding_dim)

        # Upsert in batches of 10
        batch_size = 10
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            logger.info(f"  Upserted pages {i+1}–{min(i+batch_size, len(points))}")

    logger.info(f"✓ Ingestion complete: {len(points)} pages indexed")
    return {
        "status":      "success",
        "source":      source_name,
        "pages":       len(points),
        "collection":  COLLECTION_NAME,
    }


if __name__ == "__main__":
    # Quick test — ingest the sample PDF
    result = ingest_pdf("data/raw/sample.pdf", "BIS_Working_Paper_1")
    print(result)
