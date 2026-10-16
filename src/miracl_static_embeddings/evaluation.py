import numpy as np
import numpy.typing as npt
from download_queries import Qrel

Qrels = dict[str, list[Qrel]]  # query_id -> judged docs
Run = dict[str, list[tuple[str, float]]]  # query_id -> [(doc_id, score), ...]


def dcg(gains: npt.NDArray[np.float64]) -> float:
    """Discounted cumulative gain: sum of gain / log2(rank + 1)."""
    discounts: npt.NDArray[np.float64] = np.log2(np.arange(2, gains.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(
    judgments: list[Qrel],
    retrieved: list[tuple[str, float]],
    k: int = 10,
) -> float:
    """nDCG@k for a single query."""
    relevance_by_doc: dict[str, int] = {q.doc_id: q.relevance for q in judgments}

    # Best score first, keep top k
    ranked: list[tuple[str, float]] = sorted(
        retrieved, key=lambda x: x[1], reverse=True
    )[:k]

    # Gain of each retrieved doc (unjudged docs count as 0)
    gains: npt.NDArray[np.float64] = np.array(
        [relevance_by_doc.get(doc_id, 0) for doc_id, _ in ranked],
        dtype=np.float64,
    )

    # Ideal ranking: all judged relevances, best first, cut at k
    ideal: npt.NDArray[np.float64] = np.array(
        sorted(relevance_by_doc.values(), reverse=True)[:k],
        dtype=np.float64,
    )

    idcg: float = dcg(ideal)
    if idcg == 0.0:  # no relevant docs for this query
        return 0.0
    return dcg(gains) / idcg


def evaluate_ndcg(
    qrels: Qrels,
    run: Run,
    k: int = 10,
) -> tuple[float, dict[str, float]]:
    """Returns (mean nDCG@k, per-query nDCG@k)."""
    per_query: dict[str, float] = {}
    for query_id, judgments in qrels.items():
        retrieved: list[tuple[str, float]] = run.get(
            query_id, []
        )  # missing query -> 0.0
        per_query[query_id] = ndcg_at_k(judgments, retrieved, k)

    mean: float = float(np.mean(list(per_query.values()))) if per_query else 0.0
    return mean, per_query
