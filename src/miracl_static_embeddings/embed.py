from collections.abc import Iterable
from functools import lru_cache
from typing import Literal

from fastembed.text.static_embedding import StaticTextEmbedding

ModelType = Literal[
    "minishlab/potion-base-8M",
    "minishlab/potion-retrieval-32M",
    "minishlab/potion-multilingual-128M",
    "Qdrant/constella-zero",
]
Embedding = list[float]
MODELS: dict[ModelType, str] = {
    "minishlab/potion-base-8M": "models/8M",
    "minishlab/potion-retrieval-32M": "models/32M",
    "minishlab/potion-multilingual-128M": "models/128M",
    "Qdrant/constella-zero": "models/constella",
}
DIMENSIONS: dict[ModelType, int] = {
    "minishlab/potion-base-8M": 256,
    "minishlab/potion-retrieval-32M": 512,
    "minishlab/potion-multilingual-128M": 256,
    "Qdrant/constella-zero": 1024,
}


# Fastembed only supports a few model names
def _alias_model_for_fastembed(model: ModelType) -> ModelType:
    if model == "Qdrant/constella-zero":
        return "minishlab/potion-retrieval-32M"
    return model


@lru_cache(maxsize=1, typed=True)
def get_model(model_name: ModelType) -> StaticTextEmbedding:
    return StaticTextEmbedding(
        model_name=_alias_model_for_fastembed(model_name),
        specific_model_path=MODELS[model_name],
    )


def embed(model: StaticTextEmbedding, documents: Iterable[str]) -> list[Embedding]:
    embd = model.embed(documents)

    embeddings = []

    for i in embd:
        embeddings.append(i.tolist())

    return embeddings
