# download-models

Downloads the static embedding models used by `miracl-static-embeddings` from Hugging Face. For each model it fetches `model.safetensors` and `tokenizer.json`.

| Model | Local path |
|---|---|
| `minishlab/potion-base-8M` | `models/8M` |
| `minishlab/potion-retrieval-32M` | `models/32M` |
| `minishlab/potion-multilingual-128M` | `models/128M` |

Paths are relative to the current directory, so run it from the repository root. A Hugging Face token must be available (for example `HF_TOKEN` or a prior `hf auth login`).

```bash
uv run --package download-models download-models
```
