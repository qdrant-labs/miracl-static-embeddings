from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS: dict[str, str] = {
    "minishlab/potion-base-8M": "models/8M",
    "minishlab/potion-retrieval-32M": "models/32M",
    "minishlab/potion-multilingual-128M": "models/128M",
}


def main() -> None:
    for model, path in MODELS.items():
        print(f"Downloading safetensors model and tokenizer for {model}")
        print()
        p = Path(path)
        if not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=model,
            filename="model.safetensors",
            local_dir=path,
            token=True,
        )
        hf_hub_download(
            repo_id=model,
            filename="tokenizer.json",
            local_dir=path,
            token=True,
        )
