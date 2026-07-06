"""
PDF Ingestion Pipeline — Week 3 implementation target.

Strategy:
  1. Download PDF (RBI/SEBI website or user upload)
  2. Render each page as an image (pdf2image)
  3. Embed each page image with ColPali/ColQwen2 → multi-vector embedding
  4. Store embeddings in Qdrant
  5. Also extract text (pdfplumber) → store in PostgreSQL for BM25 fallback

Why page-as-image instead of OCR chunking?
  → See docs/adr/002-why-colpali-over-ocr.md
"""
import os
import uuid
from pathlib import Path
from loguru import logger


# Data sources — free, public, no auth required
RBI_BASE_URL  = "https://www.rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx"
SEBI_BASE_URL = "https://www.sebi.gov.in/legal/circulars"


def ingest_pdf(pdf_path: str, source_name: str) -> dict:
    """
    Main ingestion entry point.
    
    Args:
        pdf_path: Local path to PDF file
        source_name: Human-readable name (e.g. "RBI_Circular_2024_03")
    
    Returns:
        dict with ingestion stats
    """
    logger.info(f"[Ingestion] Starting: {source_name}")
    
    # TODO Week 3:
    # 1. from pdf2image import convert_from_path
    #    pages = convert_from_path(pdf_path, dpi=150)
    #
    # 2. For each page image:
    #    a. Save to data/raw/images/{source_name}/page_{n}.png
    #    b. Run ColQwen2 to get multi-vector embedding
    #    c. Upsert to Qdrant with metadata
    #
    # 3. Also extract text with pdfplumber:
    #    import pdfplumber
    #    with pdfplumber.open(pdf_path) as pdf:
    #        for i, page in enumerate(pdf.pages):
    #            text = page.extract_text()
    #            # store in PostgreSQL for BM25
    #
    # 4. Return stats

    return {
        "status": "not_implemented",
        "source": source_name,
        "note": "Implement in Week 3 sprint"
    }


def download_rbi_circular(circular_url: str, save_dir: str = "data/raw") -> str:
    """Download an RBI circular PDF. Returns local file path."""
    import httpx
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = circular_url.split("/")[-1] or f"rbi_{uuid.uuid4().hex[:8]}.pdf"
    save_path = os.path.join(save_dir, filename)
    
    with httpx.Client(timeout=30) as client:
        r = client.get(circular_url)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
    
    logger.info(f"[Ingestion] Downloaded: {save_path}")
    return save_path
