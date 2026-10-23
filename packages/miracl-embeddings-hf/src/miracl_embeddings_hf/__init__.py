import asyncio
import functools
import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from qdrant_client import AsyncQdrantClient, models

app = typer.Typer()

PAGE = 2_000
ROWS_PER_GROUP = 250_000


@functools.lru_cache(maxsize=1)
def get_qdrant_client() -> AsyncQdrantClient:
    key = os.getenv("QDRANT_API_KEY")
    url = os.getenv("QDRANT_URL")
    if url is None:
        raise RuntimeError("Could not find QDRANT_URL in the current environment")
    return AsyncQdrantClient(api_key=key, url=url, timeout=600, prefer_grpc=True)


@functools.lru_cache(maxsize=1)
def get_schema(vectors_dim: int) -> pa.Schema:
    return pa.schema(
        [
            ("docid", pa.string()),
            ("title", pa.string()),
            ("text", pa.string()),
            ("vector", pa.list_(pa.float32(), vectors_dim)),
        ]
    )


class SliceWritingStatus(StrEnum):
    DONE = "done"
    SKIPPED = "skipped"
    ERROR = "error"


def to_table(points: list[models.Record], vectors_dim: int) -> pa.Table:
    vecs = np.asarray([cast(dict[str, list[float]], p.vector)["dense"] for p in points], dtype=np.float32)
    vec_col = pa.FixedSizeListArray.from_arrays(pa.array(vecs.ravel()), vectors_dim)
    doc_ids = [cast(models.Payload, p.payload).get("docid") for p in points]
    texts = [cast(models.Payload, p.payload).get("text") for p in points]
    titles = [cast(models.Payload, p.payload).get("title") for p in points]
    return pa.table(
        {"docid": doc_ids, "title": titles, "text": texts, "vector": vec_col},
        schema=get_schema(vectors_dim),
    )


def write_group(writer: pq.ParquetWriter, tables: list[pa.Table]):
    writer.write_table(pa.concat_tables(tables))


async def export_slice(
    i: int,
    total: int,
    sem: asyncio.Semaphore,
    collection: str,
    output_dir: Path,
    vector_dims: int,
) -> tuple[int, int, SliceWritingStatus]:
    done = output_dir / f"slice-{i:03d}.parquet"
    if done.exists():
        return i, 0, SliceWritingStatus.SKIPPED

    tmp = output_dir / f"slice-{i:03d}.parquet.tmp"
    flt = models.Filter(
        must=[models.SliceCondition(slice=models.Slice(index=i, total=total))]
    )
    client = get_qdrant_client()

    async with sem:
        writer = pq.ParquetWriter(tmp, get_schema(vector_dims), compression="zstd")
        try:
            offset, total, buf, buffered = None, 0, [], 0
            while True:
                points, offset = await client.scroll(
                    collection_name=collection,
                    scroll_filter=flt,
                    limit=PAGE,
                    offset=offset,
                    with_vectors=True,
                    with_payload=True,
                    timeout=600,
                )
                if points:
                    # conversion is CPU work: keep it off the event loop
                    buf.append(await asyncio.to_thread(to_table, points, vector_dims))
                    buffered += len(points)
                    total += len(points)
                    if buffered >= ROWS_PER_GROUP:
                        await asyncio.to_thread(write_group, writer, buf)
                        buf, buffered = [], 0
                if offset is None:
                    break
            if buf:
                await asyncio.to_thread(write_group, writer, buf)
        finally:
            await asyncio.to_thread(writer.close)

    tmp.rename(done)
    return i, total, SliceWritingStatus.DONE


async def main(
    num_slices: int,
    collection_name: str,
    output_dir: str,
    vector_dims: int,
    concurrency: int,
) -> None:
    out = Path(output_dir)
    if not out.is_dir():
        out.mkdir(exist_ok=True, parents=True)
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            export_slice(i, num_slices, sem, collection_name, out, vector_dims)
        )
        for i in range(num_slices)
    ]
    for coro in asyncio.as_completed(tasks):
        print(await coro)


@app.command(
    name="to-parquet",
    help="Download the Qdrant collection to a directory of parquet files",
)
def to_parquet(
    collection_name: Annotated[
        str, typer.Argument(help="Name of the collection to export")
    ],
    vector_dims: Annotated[
        int,
        typer.Option("--vector-dims", help="Vector dimensionality for the collection"),
    ],
    num_slices: Annotated[
        int,
        typer.Option(
            "--num-slices",
            help="Number of slices to slice the collection into for parallel scrolling",
        ),
    ] = 64,
    output_dir: Annotated[str, typer.Option("--out", help="Output directory")] = "out",
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency", help="Number of workers allowed to run concurrently"
        ),
    ] = 8,
) -> None:
    asyncio.run(main(num_slices, collection_name, output_dir, vector_dims, concurrency))
