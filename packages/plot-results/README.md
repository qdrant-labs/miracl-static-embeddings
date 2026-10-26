# plot-results

Builds static charts that compare the embedding models and dense/hybrid search on FiQA and on the full MIRACL English corpus, from the files in `results/`. Standard library only.

```bash
uv run --package plot-results plot-results [--results-dir results] [--output-dir charts]
```

Each chart is a self-contained HTML file written to `charts/`, plus an `index.html` that links to all of them. There is no interactivity: every value is printed on the chart, so you can open a file in a browser and screenshot it to a PNG.

| File | Reads | Content |
|---|---|---|
| `fiqa_{ndcg,mrr,recall}at10.html` | `{dense,hybrid}_{8M,32M,128M,constella}_fiqa_evaluation.json` | Dense and hybrid bars per model, one chart per metric. |
| `fiqa_hybrid_gain_ndcgat10.html` | same | Hybrid minus dense nDCG@10 per model. |
| `fiqa_embedding_throughput.html` | `dense_*_fiqa_load_timings.json` | Documents embedded per second while loading FiQA (log axis). |
| `miracl_{ndcg,mrr,recall}at10.html` | `{dense,hybrid}_{8M,32M,128M}_eval.json` | Same as FiQA. Runs without a results file show "not run". |
| `miracl_load_time.html` | `dense_{8M,32M,128M}_load_timings.json` | Embedding time and upload time for the full corpus, in minutes. |

On FiQA, the `constella` results are the stella pipeline: documents embedded with `stella_en_400M_v5` and queries embedded with the static `constella-zero` model.
