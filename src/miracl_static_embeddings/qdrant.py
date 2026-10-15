import os
from functools import lru_cache

from qdrant_client import QdrantClient


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    api_key = os.getenv("QDRANT_API_KEY")
    url = os.getenv("QDRANT_URL")
    if url is None:
        raise RuntimeError("Could not find `QDRANT_URL` in the current environment")
    return QdrantClient(url=url, api_key=api_key)
