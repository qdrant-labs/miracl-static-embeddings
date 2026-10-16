# download-queries

Downloads the English train queries and qrels of [MIRACL](https://huggingface.co/datasets/miracl/miracl) into `queries/`, and provides helpers to read them.

Paths are relative to the current directory, so run it from the repository root. A Hugging Face token must be available (for example `HF_TOKEN` or a prior `hf auth login`).

```bash
uv run download-queries
```

This creates:

- `queries/miracl-v1.0-en/topics/topics.miracl-v1.0-en-train.tsv`
- `queries/miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-train.tsv`

## Library usage

```python
from download_queries import load_qrels, load_queries

queries = load_queries(
    "queries/miracl-v1.0-en/topics/topics.miracl-v1.0-en-train.tsv"
)  # {query_id: query}
qrels = load_qrels(
    "queries/miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-train.tsv"
)  # {query_id: [Qrel(doc_id, relevance)]}
```
