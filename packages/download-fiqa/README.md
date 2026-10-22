# download-fiqa

Downloads [FiQA](https://huggingface.co/datasets/BeIR/fiqa) (BEIR) and converts it to the layout `miracl-static-embeddings` reads, so the rest of the pipeline works unchanged. Needs a Hugging Face token (for example `HF_TOKEN`).

```bash
uv run --package download-fiqa download-fiqa [--output-dir data/fiqa]
```

Files written to the output directory:

| File | Format |
|---|---|
| `corpus.jsonl` | One `{"docid", "title", "text"}` object per line. Pass to `stat-bench load --corpus-file`. |
| `queries.{train,dev,test}.tsv` | `query_id<TAB>query`, only the queries judged in that split. |
| `qrels.{train,dev,test}.tsv` | `query_id<TAB>Q0<TAB>doc_id<TAB>relevance` (TREC format). |

Sizes: 57,638 documents; 5,500 train, 500 dev and 648 test queries.
