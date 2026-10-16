# miracl-static-embeddings

Benchmarks on [MIRACL](https://huggingface.co/datasets/miracl/miracl) to test speed and retrieval quality of static embeddings in [FastEmbed](https://github.com/qdrant/fastembed)

The pipeline has four steps: download the models and queries, create a Qdrant collection, load the MIRACL corpus into it, then search and evaluate (nDCG@10).

## Setup

Requirements: Python 3.13+, [uv](https://docs.astral.sh/uv/), a Hugging Face account with access to the `miracl/miracl-corpus` dataset, and a Qdrant instance with cloud inference enabled.

1. Install the dependencies:

   ```bash
   uv sync
   ```

2. Create a `.env` file in the repository root:

   ```bash
   HF_TOKEN=<your Hugging Face token>
   QDRANT_URL=<your Qdrant URL>
   QDRANT_API_KEY=<your Qdrant API key>
   ```

   The tools do not read `.env` themselves. Either pass it to uv (`uv run --env-file .env ...`) or export it once in your shell:

   ```bash
   set -a; source .env; set +a
   ```

3. Download the models and the English train queries/qrels. Run both from the repository root, because they write to `models/` and `queries/`:

   ```bash
   uv run --package download-models download-models
   uv run download-queries
   ```

   See [`packages/download-models`](packages/download-models/README.md) and [`packages/download-queries`](packages/download-queries/README.md) for details.

## Usage

All commands are available through the `stat-bench` CLI.

### 1. Collection configs

Each collection is described by a JSON file. Ready-made configs are in [`configs/`](configs/). The fields are:

| Field | Description |
|---|---|
| `name` | Collection name. |
| `model_name` | One of `minishlab/potion-base-8M`, `minishlab/potion-retrieval-32M`, `minishlab/potion-multilingual-128M`. |
| `datatype` | Vector datatype: `float32`, `float16`, `uint8` or `turbo4`. Defaults to `float32`. |
| `quantization` | A Qdrant quantization config (`scalar`, `binary`, `turbo`, ...) or `null`. |
| `hybrid_search` | If `true`, adds a BM25 sparse vector and fuses results with RRF. |

Example:

```json
{
  "name": "miracl-potion-32M-scalar",
  "model_name": "minishlab/potion-retrieval-32M",
  "datatype": "float32",
  "quantization": {"scalar": {"type": "int8", "quantile": 0.99, "memory": "pinned"}},
  "hybrid_search": false
}
```

### 2. Create the collection

```bash
stat-bench create --config configs/potion-32M-scalar.json
```

### 3. Load the corpus

Streams `miracl/miracl-corpus` (English), embeds it with the configured model and uploads it in batches.

```bash
stat-bench load --config configs/potion-32M-scalar.json [--buffer-size 256] [--timings-output load_timings.json]
```

Per-batch embedding and upload times are saved to `--timings-output`, together with min/p50/p95/p99/max (in seconds) for each.

### 4. Search and evaluate

Run the queries and write the results to a JSON file. Add `--qrels` to compute nDCG@10 in the same run:

```bash
stat-bench search \
  --config configs/potion-32M-scalar.json \
  --queries queries/miracl-v1.0-en/topics/topics.miracl-v1.0-en-train.tsv \
  --qrels queries/miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-train.tsv \
  --output search_results.json \
  --eval-output evaluation.json
```

To evaluate saved results later:

```bash
stat-bench evaluate \
  --results search_results.json \
  --qrels queries/miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-train.tsv \
  --output evaluation.json
```

Queries and qrels are TSV files (qrels in TREC format). The evaluation output contains the mean nDCG@10 and the per-query scores.

## Development

```bash
make lint format typecheck
```
