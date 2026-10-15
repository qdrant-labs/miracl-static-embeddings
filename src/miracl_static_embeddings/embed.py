from functools import lru_cache
from typing import Literal

from fastembed.text.static_embedding import StaticTextEmbedding

ModelType = Literal[
    "minishlab/potion-base-8M",
    "minishlab/potion-base-32M",
    "minishlab/potion-multilingual-128M",
]
Embedding = list[float]
MODELS: dict[ModelType, str] = {
    "minishlab/potion-base-8M": "models/8M",
    "minishlab/potion-base-32M": "models/32M",
    "minishlab/potion-multilingual-128M": "models/128M",
}


@lru_cache(maxsize=1, typed=True)
def get_model(model_name: ModelType) -> StaticTextEmbedding:
    return StaticTextEmbedding(
        model_name=model_name, specific_model_path=MODELS[model_name]
    )


def embed_documents(
    model: StaticTextEmbedding, documents: list[str]
) -> list[Embedding]:
    embd = model.embed(documents)

    embeddings = []

    for i in embd:
        embeddings.append(i.tolist())

    return embeddings


def embed_query(model: StaticTextEmbedding, query: str) -> Embedding:
    embs = embed_documents(model, [query])

    return embs[0]
