import time
from typing import NamedTuple

from fastembed.text.static_embedding import StaticTextEmbedding
from qdrant_client import QdrantClient

from .embed import ModelType, embed, get_model
from .qdrant import DataPoint, get_qdrant_client, upload_points

DEFAULT_BUFFER_SIZE = 256


class Durations(NamedTuple):
    embedding_time: float
    upload_time: float


class UploadBuffer:
    def __init__(
        self, model: ModelType, max_buffer_size: int = DEFAULT_BUFFER_SIZE
    ) -> None:
        self.max_buffer_size = max_buffer_size
        self.model_name = model

        self._model: StaticTextEmbedding | None = None
        self._client: QdrantClient = get_qdrant_client()
        self._buffer: list[DataPoint] = []

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = get_model(self.model_name)

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.max_buffer_size

    def push(self, text: str, title: str, docid: str) -> None:
        self._buffer.append(DataPoint(vector=[], text=text, title=title, docid=docid))

    def flush(self, collection_name: str, hybrid_search: bool = False) -> Durations:
        self._ensure_model()
        assert self._model is not None
        docs = [d["text"] for d in self._buffer]
        start = time.monotonic()
        embeddings = embed(self._model, docs)
        duration = time.monotonic() - start
        self._buffer = [
            DataPoint(
                text=d["text"], title=d["title"], docid=d["docid"], vector=embeddings[i]
            )
            for i, d in enumerate(self._buffer)
        ]
        start_upload = time.monotonic()
        upload_points(
            self._client,
            collection_name,
            data=self._buffer,
            hybrid_search=hybrid_search,
            batch_size=len(self._buffer),
        )
        duration_upload = time.monotonic() - start_upload
        self._buffer = []
        return Durations(embedding_time=duration, upload_time=duration_upload)
