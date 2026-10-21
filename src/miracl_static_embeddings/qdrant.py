import os
import time
import uuid
from collections import defaultdict
from functools import lru_cache
from typing import TypedDict, cast

from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from .embed import DIMENSIONS, Embedding, ModelType, embed
from .patch_fastembed import StaticTextEmbedding
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
        url=url, api_key=api_key, cloud_inference=True, prefer_grpc=True, timeout=600
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


@with_retry
def update_quantization(
    client: QdrantClient, collection_name: str, config: CollectionConfig
) -> None:
    """Swap the quantization of an existing collection's dense vectors in place."""
    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    if not isinstance(vectors, dict) or "dense" not in vectors:
        raise ValueError(f"Collection {collection_name} has no `dense` vector")
    current = vectors["dense"].datatype
    if current is not None and current != config.datatype:
        raise ValueError(
            f"Config datatype {config.datatype.value} differs from the collection's "
            f"{current.value}: datatype can't be changed on an existing collection"
        )
    client.update_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParamsDiff(
                quantization_config=config.quantization or models.Disabled.DISABLED
            )
        },
    )


_MISSING_SPARSE = models.Filter(
    must_not=[models.HasVectorCondition(has_vector="sparse")]
)


@with_retry
def _scroll_texts(
    client: QdrantClient,
    collection_name: str,
    batch_size: int,
    offset: models.ExtendedPointId | None,
) -> tuple[list[tuple[models.ExtendedPointId, str]], models.ExtendedPointId | None]:
    records, next_offset = client.scroll(
        collection_name=collection_name,
        limit=batch_size,
        offset=offset,
        with_payload=["text"],
        with_vectors=False,
        scroll_filter=_MISSING_SPARSE,
    )
    batch = [
        (r.id, r.payload["text"]) for r in records if r.payload and "text" in r.payload
    ]
    return batch, cast(models.ExtendedPointId | None, next_offset)


@with_retry
def _update_sparse_vectors(
    client: QdrantClient,
    collection_name: str,
    batch: list[tuple[models.ExtendedPointId, str]],
) -> None:
    client.update_vectors(
        collection_name=collection_name,
        points=[
            models.PointVectors(
                id=point_id,
                vector={"sparse": models.Document(text=text, model="qdrant/bm25")},
            )
            for point_id, text in batch
        ],
    )


def add_sparse_vector(
    client: QdrantClient, collection_name: str, batch_size: int = 256
) -> None:
    """Add a BM25 `sparse` vector to an existing dense collection and fill it.

    The text is read back from the points' payload, so the dense vectors are
    neither recomputed nor re-uploaded. Safe to re-run after a failure: only
    points that don't have a sparse vector yet are processed.
    """
    info = client.get_collection(collection_name)
    if "sparse" not in (info.config.params.sparse_vectors or {}):
        client.create_vector_name(
            collection_name=collection_name,
            vector_name="sparse",
            vector_name_config=models.SparseVectorNameConfig(
                sparse=models.SparseVectorConfig()
            ),
        )

    total = client.count(
        collection_name, count_filter=_MISSING_SPARSE, exact=True
    ).count
    offset: models.ExtendedPointId | None = None
    with tqdm(total=total, desc="Adding sparse vectors", unit="point") as bar:
        while True:
            batch, offset = _scroll_texts(client, collection_name, batch_size, offset)
            if batch:
                _update_sparse_vectors(client, collection_name, batch)
                bar.update(len(batch))
            if offset is None:
                break


def wait_for_green(
    client: QdrantClient,
    collection_name: str,
    timeout: float = 3600,
    poll_interval: float = 5,
) -> None:
    """Block until the optimizers are done and the collection is green."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (
            client.get_collection(collection_name).status
            == models.CollectionStatus.GREEN
        ):
            return
        time.sleep(poll_interval)
    raise TimeoutError(f"Collection {collection_name} not green after {timeout}s")


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
    timeout: int | None = None,
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
            timeout=timeout,
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
            timeout=timeout,
        )

    query_keys = list(queries.keys())
    query_results: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
    for i, r in enumerate(results):
        points = sorted(r.points, key=lambda x: x.score, reverse=True)
        for p in points:
            if p.payload is not None:
                query_results[query_keys[i]].append((p.payload["docid"], p.score))

    return query_results


@with_retry
def delete_collection(client: QdrantClient, collection_name: str) -> None:
    client.delete_collection(collection_name=collection_name)
