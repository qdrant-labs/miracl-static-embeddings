from collections import defaultdict
from typing import Literal, NamedTuple

from huggingface_hub import snapshot_download


class Qrel(NamedTuple):
    doc_id: str
    relevance: Literal[0, 1]


def load_queries(path: str) -> dict[str, str]:
    """
    Load queries for miracl. Meant for external usage.

    Args:
        path (str): Path to the TSV file where the queries are stored as `query_id\tquery`

    Returns:
        dict[str, str]: dictionary associating each query_id to its query
    """
    queries = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            qid, q = line.split("\t", maxsplit=1)
            queries[qid] = q

    return queries


def _validate_relevance(rel: str) -> Literal[0, 1]:
    rel_int = int(rel)
    if rel_int in (0, 1):
        return rel_int
    raise TypeError("Relevance score should be 0 or 1")


def load_qrels(path: str) -> dict[str, list[Qrel]]:
    """
    Load qrels for miracl. Meant for external usage.

    Args:
        path (str): Path to the TSV file where the qrels are stored in TREC format

    Returns:
        dict[str, list[Qrel]]: dictionary associating each query ID to its (doc_id, relevance) tuple
    """

    qrels: defaultdict[str, list[Qrel]] = defaultdict(list)

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            qid, _, docid, relevance = line.split("\t", maxsplit=3)
            qrels[qid].append(
                Qrel(doc_id=docid, relevance=_validate_relevance(relevance))
            )

    return qrels


def main() -> None:
    snapshot_download(
        repo_id="miracl/miracl",
        repo_type="dataset",
        token=True,
        allow_patterns=[
            "miracl-v1.0-en/topics/topics.miracl-v1.0-en-train.tsv",
            "miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-train.tsv",
        ],
        local_dir="queries/",
    )
