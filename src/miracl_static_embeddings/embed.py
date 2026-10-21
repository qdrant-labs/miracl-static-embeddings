from collections.abc import Iterable
from functools import lru_cache
from typing import Literal

from fastembed import TextEmbedding
from fastembed.text.static_embedding import StaticTextEmbedding
from fastembed.text.text_embedding_base import TextEmbeddingBase

ModelType = Literal[
    "minishlab/potion-base-8M",
    "minishlab/potion-retrieval-32M",
    "minishlab/potion-multilingual-128M",
    "Qdrant/constella-zero",
    "Qdrant/stella-en-400M-v5-doc-onnx",
]
Embedding = list[float]
MODELS: dict[ModelType, str | None] = {
    "minishlab/potion-base-8M": "models/8M",
    "minishlab/potion-retrieval-32M": "models/32M",
    "minishlab/potion-multilingual-128M": "models/128M",
    "Qdrant/constella-zero": "models/constella",
    "Qdrant/stella-en-400M-v5-doc-onnx": None,
}
DIMENSIONS: dict[ModelType, int] = {
    "minishlab/potion-base-8M": 256,
    "minishlab/potion-retrieval-32M": 512,
    "minishlab/potion-multilingual-128M": 256,
    "Qdrant/constella-zero": 1024,
    "Qdrant/stella-en-400M-v5-doc-onnx": 1024,
}


@lru_cache(maxsize=1, typed=True)
def get_model(model_name: ModelType) -> TextEmbeddingBase:
    if model_name != "Qdrant/stella-en-400M-v5-doc-onnx":
        return StaticTextEmbedding(
            model_name=model_name,
            specific_model_path=MODELS[model_name],
        )
    return TextEmbedding(model_name=model_name, cache_dir="models")


def embed(model: TextEmbeddingBase, documents: Iterable[str]) -> list[Embedding]:
    embd = model.embed(documents)

    embeddings = []

    for i in embd:
        embeddings.append(i.tolist())

    return embeddings
