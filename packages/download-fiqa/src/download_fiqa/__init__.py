import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

CORPUS_REPO = "BeIR/fiqa"
QRELS_REPO = "BeIR/fiqa-qrels"
SPLITS = ("train", "dev", "test")


def _one_line(text: str) -> str:
    return " ".join(text.split())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="download-fiqa")
    parser.add_argument("--output-dir", default="data/fiqa")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    corpus_dir = Path(
        snapshot_download(repo_id=CORPUS_REPO, repo_type="dataset", token=True)
    )
    qrels_dir = Path(
        snapshot_download(repo_id=QRELS_REPO, repo_type="dataset", token=True)
    )

    # corpus.jsonl: one {"docid", "title", "text"} object per line
    corpus = pq.read_table(corpus_dir / "corpus").to_pylist()
    with open(out / "corpus.jsonl", "w") as f:
        f.writelines(
            json.dumps(
                {
                    "docid": row["_id"],
                    "title": row["title"],
                    "text": row["text"],
                }
            )
            + "\n"
            for row in corpus
        )
    print(f"Wrote {len(corpus)} documents to {out / 'corpus.jsonl'}")

    queries = {
        row["_id"]: _one_line(row["text"])
        for row in pq.read_table(corpus_dir / "queries").to_pylist()
    }

    for split in SPLITS:
        # qrels.<split>.tsv: `query_id\tQ0\tdoc_id\trelevance` (TREC format)
        with open(qrels_dir / f"{split}.tsv", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        query_ids = list(dict.fromkeys(r["query-id"] for r in rows))
        with open(out / f"qrels.{split}.tsv", "w") as f:
            f.writelines(
                f"{r['query-id']}\tQ0\t{r['corpus-id']}\t{r['score']}\n" for r in rows
            )
        # queries.<split>.tsv: `query_id\tquery`, only the queries judged in this split
        with open(out / f"queries.{split}.tsv", "w") as f:
            f.writelines(f"{qid}\t{queries[qid]}\n" for qid in query_ids)
        print(f"{split}: {len(query_ids)} queries, {len(rows)} judgments")
