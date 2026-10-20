import argparse
import itertools
import json
import logging
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from datasets import load_dataset
from download_queries import load_qrels, load_queries
from tqdm import tqdm

from miracl_static_embeddings.evaluation import (
    evaluate_mrr,
    evaluate_ndcg,
    evaluate_recall,
)

from .buffer import DEFAULT_BUFFER_SIZE, UploadBuffer
from .embed import ModelType, get_model
from .qdrant import (
    CollectionConfig,
    add_sparse_vector,
    create_collection,
    delete_collection,
    get_qdrant_client,
    search,
    update_quantization,
    wait_for_green,
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
    n_batches = -(-len(queries) // batch_size)
    for b in tqdm(
        itertools.batched(queries.items(), n=batch_size),
        total=n_batches,
        desc="Searching",
        unit="batch",
    ):
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
    d = {
        name: {"mean": mean, "per_query": per_query}
        for name, (mean, per_query) in (
            ("ndcg@10", evaluate_ndcg(qrels, search_results)),
            ("mrr@10", evaluate_mrr(qrels, search_results)),
            ("recall@10", evaluate_recall(qrels, search_results)),
        )
    }
    with open(output_file, "w") as f:
        json.dump(d, f, indent=2)


def print_timings(timings_file: str) -> None:
    with open(timings_file) as f:
        data = json.load(f)
    stat_names = ("count", "min", "p50", "p95", "p99", "max")
    print(f"{'':<16}" + "".join(f"{n:>12}" for n in stat_names))
    for name, section in data.items():
        stats = summarize_times(section["per_batch"])
        row = "".join(
            f"{stats[n]:>12d}" if n == "count" else f"{stats[n]:>12.4f}"
            for n in stat_names
            if n in stats
        )
        print(f"{name:<16}{row}")
    print("(times in seconds, per batch)")


def load_config(path: str) -> CollectionConfig:
    with open(path) as f:
        return CollectionConfig.model_validate_json(f.read())


def _cmd_create(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    create_collection(get_qdrant_client(), config)
    logger.info("Created collection %s", config.name)


def _cmd_quantize(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    collection = args.collection or config.name
    client = get_qdrant_client()
    update_quantization(client, collection, config)
    logger.info("Updated quantization of %s, waiting for optimizers", collection)
    wait_for_green(client, collection)
    logger.info("Collection %s is green", collection)


def _cmd_add_sparse(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if not config.hybrid_search:
        raise SystemExit(f"{args.config} has `hybrid_search: false`, nothing to add")
    collection = args.collection or config.name
    client = get_qdrant_client()
    add_sparse_vector(client, collection, batch_size=args.batch_size)
    wait_for_green(client, collection)
    logger.info("Collection %s is green", collection)


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
        collection_name=args.collection or config.name,
        output_file=args.output,
        hybrid_search=config.hybrid_search,
        batch_size=args.batch_size,
    )
    if args.qrels is not None:
        evaluate_searches(args.qrels, results, args.eval_output)


def _cmd_timings(args: argparse.Namespace) -> None:
    print_timings(args.timings)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    with open(args.results) as f:
        results = json.load(f)
    evaluate_searches(args.qrels, results, args.output)


def _cmd_delete(args: argparse.Namespace) -> None:
    delete_collection(get_qdrant_client(), args.collection)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="miracl-static-embeddings")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a collection from a JSON config")
    create.add_argument("--config", required=True, help="Path to collection config")
    create.set_defaults(func=_cmd_create)

    quant = sub.add_parser(
        "quantize",
        help="Apply a config's quantization to an existing collection (no re-upload)",
    )
    quant.add_argument("--config", required=True, help="Path to collection config")
    quant.add_argument(
        "--collection", help="Collection to update (default: the config's name)"
    )
    quant.set_defaults(func=_cmd_quantize)

    sparse = sub.add_parser(
        "add-sparse",
        help="Add BM25 sparse vectors to an existing collection (no dense re-upload)",
    )
    sparse.add_argument("--config", required=True, help="Path to a hybrid config")
    sparse.add_argument(
        "--collection", help="Collection to update (default: the config's name)"
    )
    sparse.add_argument("--batch-size", type=int, default=256)
    sparse.set_defaults(func=_cmd_add_sparse)

    load = sub.add_parser("load", help="Embed and upload the MIRACL corpus")
    load.add_argument("--config", required=True, help="Path to collection config")
    load.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE)
    load.add_argument("--timings-output", default="load_timings.json")
    load.set_defaults(func=_cmd_load)

    srch = sub.add_parser("search", help="Run queries (and evaluate if --qrels given)")
    srch.add_argument("--config", required=True, help="Path to collection config")
    srch.add_argument(
        "--collection", help="Collection to search (default: the config's name)"
    )
    srch.add_argument("--queries", required=True, help="Path to queries file")
    srch.add_argument("--output", default="search_results.json")
    srch.add_argument("--batch-size", type=int, default=256)
    srch.add_argument(
        "--qrels", help="Path to qrels TSV file (TREC format); enables evaluation"
    )
    srch.add_argument("--eval-output", default="evaluation.json")
    srch.set_defaults(func=_cmd_search)

    ev = sub.add_parser(
        "evaluate",
        help="Compute nDCG@10, MRR@10 and recall@10 from saved search results",
    )
    ev.add_argument("--results", required=True, help="Search results JSON")
    ev.add_argument(
        "--qrels", required=True, help="Path to qrels TSV file (TREC format)"
    )
    ev.add_argument("--output", default="evaluation.json")
    ev.set_defaults(func=_cmd_evaluate)

    tm = sub.add_parser("timings", help="Print load timing stats from a timings file")
    tm.add_argument("timings", help="Timings JSON written by `load`")
    tm.set_defaults(func=_cmd_timings)

    delt = sub.add_parser(
        "delete", help="Delete a collection after you're done with evals"
    )
    delt.add_argument("collection", help="Name of the collection to delete")
    delt.set_defaults(func=_cmd_delete)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    args.func(args)
