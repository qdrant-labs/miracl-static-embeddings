# plot-fiqa

Builds static charts that compare the embedding models and dense/hybrid search on FiQA, from the `*_fiqa_evaluation.json` files in `results/`. Standard library only.

```bash
uv run --package plot-fiqa plot-fiqa [--results-dir results] [--output-dir charts]
```

It expects `{dense,hybrid}_{8M,32M,128M,constella}_fiqa_evaluation.json` in the results directory and writes one self-contained HTML file per chart to `charts/`:

| File | Content |
|---|---|
| `fiqa_ndcgat10.html`, `fiqa_mrrat10.html`, `fiqa_recallat10.html` | Dense and hybrid bars per model, one chart per metric (shared 0 to 0.5 axis). |
| `fiqa_hybrid_gain_ndcgat10.html` | Hybrid minus dense nDCG@10 per model. |
| `index.html` | Landing page with links to every chart. |

The charts have no interactivity: every value is printed on the chart, so you can open a file in a browser and screenshot it to a PNG.
