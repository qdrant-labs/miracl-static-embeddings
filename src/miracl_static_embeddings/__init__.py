import argparse
import itertools
import json
import logging
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from datasets import load_dataset
from download_queries import load_qrels, load_queries

from miracl_static_embeddings.evaluation import evaluate_ndcg

from .buffer import DEFAULT_BUFFER_SIZE, UploadBuffer
from .embed import ModelType, get_model
from .qdrant import (
    CollectionConfig,
    create_collection,
    get_qdrant_client,
    search,
)

logger = logging.getLogger(__name__)


class LoadTimings(NamedTuple):
    embedding_times: list[float]
    upload_times: list[float]


def summarize_times(times: Sequence[float]) -> dict[str, float | int]:
    """Min, p50, p95, p99, max (in seconds) and count of per-batch durations."""
    if len(times) == 0:
        return {"count": 0}
    arr = np.asarray(times, dtype=float)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {
        "count": len(arr),
        "min": float(arr.min()),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(arr.max()),
    }


def save_timings(timings: LoadTimings, output_file: str) -> None:
    d = {
        name: {"stats": summarize_times(times), "per_batch": times}
        for name, times in (
            ("embedding_time", timings.embedding_times),
            ("upload_time", timings.upload_times),
        )
    }
    with open(output_file, "w") as f:
        json.dump(d, f, indent=2)


def load_data(
    model: ModelType,
    collection_name: str,
    hybrid_search: bool = False,
    max_buffer_size: int = DEFAULT_BUFFER_SIZE,
) -> LoadTimings:
    buf = UploadBuffer(model=model, max_buffer_size=max_buffer_size)
    embed_times: list[float] = []
    upload_times: list[float] = []
    dataset = load_dataset(
        "json",
        data_files="hf://datasets/miracl/miracl-corpus/miracl-corpus-v1.0-en/*.jsonl.gz",
        split="train",
        streaming=True,
    )

    for d in dataset:
        try:
            buf.push(d["text"], d["title"], d["docid"])
        except KeyError:
            # malformed entry
            logger.warning("Unexpected schema in entry: %s", d)
            continue
        if buf.is_full:
            durations = buf.flush(collection_name, hybrid_search)
            embed_times.append(durations.embedding_time)
            upload_times.append(durations.upload_time)
            logger.info(f"Uploaded {buf.max_buffer_size} points to Qdrant")

    if len(buf) > 0:
        durations = buf.flush(collection_name, hybrid_search)
        embed_times.append(durations.embedding_time)
        upload_times.append(durations.upload_time)
        logger.info("Uploaded remaining points to Qdrant")

    return LoadTimings(embedding_times=embed_times, upload_times=upload_times)


def run_searches(
    queries_file: str,
    model_name: ModelType,
    collection_name: str,
    output_file: str,
    hybrid_search: bool = False,
    batch_size: int = 256,
) -> dict[str, list[tuple[str, float]]]:
    queries = load_queries(queries_file)
    model = get_model(model_name)
    client = get_qdrant_client()
    full_results: dict[str, list[tuple[str, float]]] = {}
    for b in itertools.batched(queries.items(), n=batch_size):
        d = {el[0]: el[1] for el in b}
        results = search(
            client,
            model,
            collection_name=collection_name,
            queries=d,
            hybrid_search=hybrid_search,
        )
        full_results.update(results)
    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)
    return full_results


def evaluate_searches(
    qrels_file: str,
    search_results: dict[str, list[tuple[str, float]]],
    output_file: str,
) -> None:
    qrels = load_qrels(qrels_file)
    mean_ndcg, per_quer_ndcg = evaluate_ndcg(qrels, search_results)
    d = {"ndcg@10": {"mean": mean_ndcg, "per_query": per_quer_ndcg}}
    with open(output_file, "w") as f:
        json.dump(d, f, indent=2)


def load_config(path: str) -> CollectionConfig:
    with open(path) as f:
        return CollectionConfig.model_validate_json(f.read())


def _cmd_create(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    create_collection(get_qdrant_client(), config)
    logger.info("Created collection %s", config.name)


def _cmd_load(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    timings = load_data(
        model=config.model_name,
        collection_name=config.name,
        hybrid_search=config.hybrid_search,
        max_buffer_size=args.buffer_size,
    )
    save_timings(timings, args.timings_output)


def _cmd_search(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    results = run_searches(
        queries_file=args.queries,
        model_name=config.model_name,
        collection_name=config.name,
        output_file=args.output,
        hybrid_search=config.hybrid_search,
        batch_size=args.batch_size,
    )
    if args.qrels is not None:
        evaluate_searches(args.qrels, results, args.eval_output)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    with open(args.results) as f:
        results = json.load(f)
    evaluate_searches(args.qrels, results, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="miracl-static-embeddings")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a collection from a JSON config")
    create.add_argument("--config", required=True, help="Path to collection config")
    create.set_defaults(func=_cmd_create)

    load = sub.add_parser("load", help="Embed and upload the MIRACL corpus")
    load.add_argument("--config", required=True, help="Path to collection config")
    load.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE)
    load.add_argument("--timings-output", default="load_timings.json")
    load.set_defaults(func=_cmd_load)

    srch = sub.add_parser("search", help="Run queries (and evaluate if --qrels given)")
    srch.add_argument("--config", required=True, help="Path to collection config")
    srch.add_argument("--queries", required=True, help="Path to queries file")
    srch.add_argument("--output", default="search_results.json")
    srch.add_argument("--batch-size", type=int, default=256)
    srch.add_argument(
        "--qrels", help="Path to qrels TSV file (TREC format); enables evaluation"
    )
    srch.add_argument("--eval-output", default="evaluation.json")
    srch.set_defaults(func=_cmd_search)

    ev = sub.add_parser("evaluate", help="Compute nDCG@10 from saved search results")
    ev.add_argument("--results", required=True, help="Search results JSON")
    ev.add_argument(
        "--qrels", required=True, help="Path to qrels TSV file (TREC format)"
    )
    ev.add_argument("--output", default="evaluation.json")
    ev.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    args.func(args)
