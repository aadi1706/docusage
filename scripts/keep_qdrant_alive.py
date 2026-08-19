"""Run daily to keep Qdrant free cluster alive."""
import os
from qdrant_client import QdrantClient

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
result = client.get_collections()
print(f"Qdrant alive — {len(result.collections)} collections")
