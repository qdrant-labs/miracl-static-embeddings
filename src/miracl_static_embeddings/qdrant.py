import os
import uuid
from collections import defaultdict
from functools import lru_cache
from typing import TypedDict

from fastembed.text.static_embedding import StaticTextEmbedding
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from .embed import DIMENSIONS, Embedding, ModelType, embed
from .retry import with_retry


class DataPoint(TypedDict):
    vector: Embedding
    docid: str
    text: str
    title: str


class CollectionConfig(BaseModel):
    name: str
    model_name: ModelType
    datatype: models.Datatype = models.Datatype.FLOAT32
    quantization: models.QuantizationConfig | None = None
    hybrid_search: bool = False


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    api_key = os.getenv("QDRANT_API_KEY")
    url = os.getenv("QDRANT_URL")
    if url is None:
        raise RuntimeError("Could not find `QDRANT_URL` in the current environment")
    return QdrantClient(
        url=url, api_key=api_key, cloud_inference=True, prefer_grpc=True, timeout=60
    )


@with_retry
def create_collection(client: QdrantClient, config: CollectionConfig) -> None:
    if config.hybrid_search:
        client.create_collection(
            collection_name=config.name,
            vectors_config={
                "dense": models.VectorParams(
                    size=DIMENSIONS[config.model_name],
                    distance=models.Distance.COSINE,
                    memory=models.Memory.COLD,
                    datatype=config.datatype,
                    quantization_config=config.quantization,
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        return
    client.create_collection(
        collection_name=config.name,
        vectors_config={
            "dense": models.VectorParams(
                size=DIMENSIONS[config.model_name],
                distance=models.Distance.COSINE,
                memory=models.Memory.COLD,
                datatype=config.datatype,
                quantization_config=config.quantization,
            )
        },
    )


def _determinstic_id(docid: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, docid))


@with_retry
def upload_points(
    client: QdrantClient,
    collection_name: str,
    data: list[DataPoint],
    batch_size: int,
    hybrid_search: bool = False,
) -> None:
    if hybrid_search:
        client.upload_points(
            collection_name=collection_name,
            points=(
                models.PointStruct(
                    id=_determinstic_id(d["docid"]),
                    vector={
                        "dense": d["vector"],
                        "sparse": models.Document(text=d["text"], model="qdrant/bm25"),
                    },
                    payload={k: v for k, v in d.items() if k != "vector"},
                )
                for d in data
            ),
            batch_size=batch_size,
        )
        return
    client.upload_points(
        collection_name=collection_name,
        points=(
            models.PointStruct(
                id=_determinstic_id(d["docid"]),
                vector={
                    "dense": d["vector"],
                },
                payload={k: v for k, v in d.items() if k != "vector"},
            )
            for d in data
        ),
        batch_size=batch_size,
    )


@with_retry
def search(
    client: QdrantClient,
    embedding_model: StaticTextEmbedding,
    collection_name: str,
    queries: dict[str, str],
    hybrid_search: bool = False,
) -> dict[str, list[tuple[str, float]]]:
    query_vecs = embed(embedding_model, queries.values())
    query_docs = zip(queries.values(), query_vecs)
    if hybrid_search:
        results = client.query_batch_points(
            collection_name=collection_name,
            requests=[
                models.QueryRequest(
                    prefetch=[
                        models.Prefetch(query=query_vec, using="dense", limit=50),
                        models.Prefetch(
                            query=models.Document(text=query, model="qdrant/bm25"),
                            using="sparse",
                            limit=50,
                        ),
                    ],
                    query=models.RrfQuery(rrf=models.Rrf()),
                    limit=10,
                    with_payload=True,
                )
                for (query, query_vec) in query_docs
            ],
        )
    else:
        results = client.query_batch_points(
            collection_name=collection_name,
            requests=[
                models.QueryRequest(
                    query=query_vec, using="dense", limit=10, with_payload=True
                )
                for query_vec in query_vecs
            ],
        )

    query_keys = list(queries.keys())
    query_results: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
    for i, r in enumerate(results):
        points = sorted(r.points, key=lambda x: x.score, reverse=True)
        for p in points:
            if p.payload is not None:
                query_results[query_keys[i]].append((p.payload["docid"], p.score))

    return query_results
