"""
Extraction Agent — uses a Vision Language Model (VLM) to extract
structured data from table/chart pages retrieved by the RetrievalAgent.

Model: Qwen2-VL-7B (via HuggingFace Inference Endpoint or local)
HuggingFace task: Image-Text-to-Text
"""
import os
from loguru import logger
from .state import DocuSageState


class ExtractionAgent:
    """
    Extracts structured data (numbers, labels, relationships) from page images.
    Falls back gracefully when no image is available (text-only chunk).
    """

    def __init__(self):
        self.hf_token = os.getenv("HF_API_TOKEN")
        self.model_id = "Qwen/Qwen2-VL-7B-Instruct"  # HF model ID
        # Client initialized lazily
        self._client = None

    def run(self, state: DocuSageState) -> DocuSageState:
        image_chunks = [c for c in state.retrieved_chunks if c.image_path]

        if not image_chunks:
            logger.info("[ExtractionAgent] No image chunks — skipping VLM extraction")
            return state

        logger.info(f"[ExtractionAgent] Extracting from {len(image_chunks)} image pages")

        extracted = {}
        for chunk in image_chunks:
            result = self._extract_from_image(chunk.image_path, state.user_query)
            extracted[chunk.chunk_id] = result

        state.extracted_data = extracted
        return state

    def _extract_from_image(self, image_path: str, query: str) -> dict:
        """
        Call Qwen2-VL via HF Inference API.
        TODO Week 5: wire up real API call below.
        """
        # import requests, base64
        # with open(image_path, "rb") as f:
        #     img_b64 = base64.b64encode(f.read()).decode()
        # payload = {
        #     "model": self.model_id,
        #     "messages": [{"role": "user", "content": [
        #         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        #         {"type": "text", "text": f"Extract all data relevant to: {query}. Return as JSON."}
        #     ]}]
        # }
        # ... call HF endpoint ...
        return {"mock": True, "note": "Replace in Week 5"}
