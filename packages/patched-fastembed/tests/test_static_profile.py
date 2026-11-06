"""Benchmark of the three stages of static embedding computation.

The stages are:

- tokenization: `Tokenizer.encode_batch` plus building the flat id array;
- lookup: mapping token ids to table rows, deduplicating them and gathering their vectors;
- pooling: building the (documents x distinct tokens) count matrix and multiplying it with the
  gathered vectors (or weighting and summing them, for a single document).

The profiling is done by monkeypatching `StaticTextEmbedding._token_ids` and `._sum_rows` for the
duration of a test, so the package itself carries no timing code. `_sum_rows` is re-implemented
below to split lookup from pooling, which the original fuses in its last matrix product: keep it
in sync with `fastembed/text/static_embedding.py`. Outputs are checked against the unpatched
model, so a drift between the two shows up as a failure.

Compute is estimated, not measured: wall time and process CPU time are real, FLOPs and bytes moved
are derived from the shapes involved. CPU time above wall time means the stage ran multi-threaded
(tokenizers uses a thread pool).

The point of the benchmark is how the share of tokenization changes between `tokenizers` v0 and
v1, so it runs under both: each run writes `bench_results/profile-<tokenizers version>-<model>.json`
and `test_compare_tokenizers_versions` prints the two side by side. The package targets the v1
API (`Truncation`, `ids_array`); under v0 the test installs small shims for those two spots
(see `install_v0_shims`), which leave the computation itself unchanged.

    # v1: the project environment
    uv run --group dev --package patched-fastembed pytest packages/patched-fastembed/tests -s
    # v0: a throwaway environment (outside the repo, so uv ignores its `tokenizers` build config),
    # with the package on PYTHONPATH because its own metadata pins tokenizers v1
    cd /tmp && PYTHONPATH=<repo>/packages/patched-fastembed/src uv run --no-config --isolated \\
        --no-project --with 'tokenizers<1' --with numpy --with pytest --with onnxruntime \\
        --with tqdm --with requests --with huggingface-hub --with loguru --with pillow \\
        --with mmh3 --with py-rust-stemmers \\
        pytest <repo>/packages/patched-fastembed/tests -s -p no:cacheprovider
    # then, in either environment, the comparison
    uv run --group dev --package patched-fastembed pytest packages/patched-fastembed/tests -s -k compare

`-s` shows the reports.
"""

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tokenizers
from numpy.typing import NDArray

TOKENIZERS_VERSION = version("tokenizers")
IS_TOKENIZERS_V0 = not hasattr(tokenizers, "Truncation")

if IS_TOKENIZERS_V0:
    # fastembed imports `Truncation` and `Padding`, which only exist in v1. The stubs let the
    # imports succeed; the static path never uses them under v0 (see `install_v0_shims`).
    tokenizers.Truncation = lambda max_length: max_length  # type: ignore[attr-defined]
    tokenizers.Padding = lambda **kwargs: kwargs  # type: ignore[attr-defined]

from fastembed.common.types import NumpyArray
from fastembed.text import static_embedding
from fastembed.text.static_embedding import MAX_LENGTH, StaticTextEmbedding


def install_v0_shims() -> None:
    """Adapt the two v1-only spots of `static_embedding` to the tokenizers v0 API.

    - `load_static_tokenizer` assigns `Tokenizer.truncation`, which v0 has no setter for.
    - `_token_ids` reads `Encoding.ids_array` (numpy views), which v0 has no equivalent of: it
      converts the Python id lists, and that conversion is part of v0's real cost.
    """

    def load_static_tokenizer(model_dir: Path):
        tokenizer = tokenizers.Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(MAX_LENGTH)
        tokenizer.no_padding()
        with open(model_dir / "tokenizer.json", encoding="utf-8") as f:
            tokenizer_model = json.load(f)["model"]
        if tokenizer_model.get("type") == "Unigram":
            return tokenizer, tokenizer_model.get("unk_id")
        unk_token = tokenizer_model.get("unk_token")
        vocab = tokenizer_model.get("vocab")
        unk = vocab.get(unk_token) if unk_token and isinstance(vocab, dict) else None
        return tokenizer, unk

    def token_ids(self: StaticTextEmbedding, documents: list[str]):
        assert self.tokenizer is not None
        encodings = self.tokenizer.encode_batch(
            documents, add_special_tokens=self.model_description.add_special_tokens
        )
        ids_per_doc = [np.asarray(e.ids, dtype=np.uint32) for e in encodings]
        lengths = np.fromiter(map(len, ids_per_doc), dtype=np.int64, count=len(documents))
        ids = np.concatenate(ids_per_doc) if ids_per_doc else np.empty(0, dtype=np.uint32)
        return ids, lengths

    static_embedding.load_static_tokenizer = load_static_tokenizer
    StaticTextEmbedding._token_ids = token_ids  # type: ignore[method-assign]


if IS_TOKENIZERS_V0:
    install_v0_shims()

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
MODELS = {
    "minishlab/potion-base-8M": MODELS_DIR / "8M",
    "minishlab/potion-retrieval-32M": MODELS_DIR / "32M",
}
RESULTS_DIR = Path(__file__).parent / "bench_results"
STAGES = ("tokenization", "lookup", "pooling")
N_DOCUMENTS = 2048
WORDS_PER_DOCUMENT = 150
BATCH_SIZE = 256
REPEATS = 10


@dataclass
class StageStats:
    wall: float = 0.0
    cpu: float = 0.0
    flops: float = 0.0
    bytes_moved: float = 0.0
    calls: int = 0


@dataclass
class Profile:
    stages: dict[str, StageStats] = field(
        default_factory=lambda: defaultdict(StageStats)
    )
    tokens: int = 0
    documents: int = 0

    def measure(self, stage: str, flops: float = 0.0, bytes_moved: float = 0.0):
        return _Timer(self.stages[stage], flops, bytes_moved)


class _Timer:
    def __init__(self, stats: StageStats, flops: float, bytes_moved: float):
        self.stats, self.flops, self.bytes_moved = stats, flops, bytes_moved

    def __enter__(self) -> "_Timer":
        self.wall0, self.cpu0 = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stats.wall += time.perf_counter() - self.wall0
        self.stats.cpu += time.process_time() - self.cpu0
        self.stats.flops += self.flops
        self.stats.bytes_moved += self.bytes_moved
        self.stats.calls += 1


def install_profiler(
    monkeypatch: pytest.MonkeyPatch, profile: Profile
) -> None:
    original_token_ids = StaticTextEmbedding._token_ids

    def token_ids(self: StaticTextEmbedding, documents: list[str]):
        chars = sum(map(len, documents))
        with profile.measure("tokenization", bytes_moved=chars):
            ids, lengths = original_token_ids(self, documents)
        profile.tokens += int(ids.size)
        profile.documents += len(documents)
        return ids, lengths

    def sum_rows(
        self: StaticTextEmbedding,
        ids: NDArray[np.integer[Any]],
        lengths: NDArray[np.int64],
    ) -> NumpyArray:
        # Mirrors StaticTextEmbedding._sum_rows, with lookup and pooling timed apart.
        assert self.embeddings is not None
        dim = self.embeddings.shape[1]
        row_bytes = dim * self.embeddings.itemsize
        if not ids.size:
            return np.zeros((len(lengths), dim), dtype=np.float32)
        sqrt_counts = self.model_description.sqrt_token_counts

        if len(lengths) == 1 and not sqrt_counts:
            with profile.measure("lookup", bytes_moved=ids.size * row_bytes):
                rows = ids if self.token_mapping is None else self.token_mapping[ids]
                vectors = self.embeddings[rows]
            with profile.measure("pooling", flops=2 * ids.size * dim):
                if self.weights is not None:
                    vectors = vectors * self.weights[ids, None]
                return vectors.sum(axis=0, keepdims=True, dtype=np.float32)

        with profile.measure("lookup"):
            rows = ids if self.token_mapping is None else self.token_mapping[ids]
            distinct_rows, row_index = np.unique(rows, return_inverse=True)
        n_docs, n_distinct = len(lengths), len(distinct_rows)
        if n_docs > 1 and n_docs * n_distinct > self.POOLING_CELLS:
            half = n_docs // 2
            split = int(lengths[:half].sum())
            return np.concatenate(
                [
                    self._sum_rows(ids[:split], lengths[:half]),
                    self._sum_rows(ids[split:], lengths[half:]),
                ]
            )
        with profile.measure("lookup", bytes_moved=n_distinct * row_bytes):
            table = self.embeddings[distinct_rows]
        with profile.measure("pooling", flops=2 * n_docs * n_distinct * dim + ids.size):
            doc_index = np.repeat(np.arange(n_docs), lengths)
            counts = (
                np.bincount(
                    doc_index * n_distinct + row_index,
                    weights=None
                    if self.weights is None or sqrt_counts
                    else self.weights[ids],
                    minlength=n_docs * n_distinct,
                )
                .astype(np.float32)
                .reshape(n_docs, n_distinct)
            )
            if sqrt_counts:
                np.sqrt(counts, out=counts)
                if self.weights is not None:
                    counts *= self.weights[distinct_rows]
            return counts @ table

    monkeypatch.setattr(StaticTextEmbedding, "_token_ids", token_ids)
    monkeypatch.setattr(StaticTextEmbedding, "_sum_rows", sum_rows)


def make_documents(model: StaticTextEmbedding) -> list[str]:
    """Deterministic synthetic corpus: random words drawn from the model's own vocabulary."""
    with open(model._model_dir / "tokenizer.json", encoding="utf-8") as f:
        tokenizer_model = json.load(f)["model"]
    tokens = tokenizer_model["vocab"]
    if not isinstance(tokens, dict):  # Unigram: a list of [token, score]
        tokens = [t for t, _ in tokens]
    vocab = [t for t in tokens if t.isalpha() and t.isascii() and len(t) > 2]
    vocab.sort()
    rng = random.Random(0)
    return [
        " ".join(rng.choices(vocab, k=WORDS_PER_DOCUMENT)) for _ in range(N_DOCUMENTS)
    ]


def run_embed(model: StaticTextEmbedding, documents: list[str]) -> list[NumpyArray]:
    return [
        model.raw_embed(documents[i : i + BATCH_SIZE])
        for i in range(0, len(documents), BATCH_SIZE)
    ]


def format_report(name: str, profile: Profile, total_wall: float) -> str:
    lines = [
        f"\n== {name}: {profile.documents} docs, {profile.tokens} tokens "
        f"(batch {BATCH_SIZE}, best of {REPEATS}) ==",
        f"{'stage':<14}{'wall ms':>10}{'%wall':>8}{'cpu ms':>10}{'cpu/wall':>10}"
        f"{'GFLOP':>9}{'MB moved':>10}{'us/doc':>9}",
    ]
    for stage in STAGES:
        s = profile.stages[stage]
        lines.append(
            f"{stage:<14}{s.wall * 1e3:>10.1f}{100 * s.wall / total_wall:>7.1f}%"
            f"{s.cpu * 1e3:>10.1f}{s.cpu / max(s.wall, 1e-9):>10.2f}"
            f"{s.flops / 1e9:>9.3f}{s.bytes_moved / 1e6:>10.1f}"
            f"{s.wall * 1e6 / profile.documents:>9.1f}"
        )
    return "\n".join(lines)


@pytest.mark.parametrize("model_name", list(MODELS))
def test_profile_static_stages(
    model_name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model_dir = MODELS[model_name]
    if not (model_dir / "model.safetensors").exists():
        pytest.skip(f"{model_dir} not downloaded")
    model = StaticTextEmbedding(model_name, specific_model_path=str(model_dir))
    documents = make_documents(model)

    reference = run_embed(model, documents)  # unpatched, also warms caches

    best: Profile | None = None
    best_wall = float("inf")
    for _ in range(REPEATS):
        profile = Profile()
        with monkeypatch.context() as m:
            install_profiler(m, profile)
            start = time.perf_counter()
            result = run_embed(model, documents)
            wall = time.perf_counter() - start
        for got, want in zip(result, reference, strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)
        if wall < best_wall:
            best, best_wall = profile, wall
    assert best is not None

    staged = sum(best.stages[s].wall for s in STAGES)
    assert all(best.stages[s].calls > 0 for s in STAGES)
    assert staged <= best_wall * 1.01
    with capsys.disabled():
        print(f"\ntokenizers {TOKENIZERS_VERSION}", end="")
        print(format_report(model_name, best, best_wall))
        print(f"{'other/untimed':<14}{(best_wall - staged) * 1e3:>10.1f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    short = model_name.rsplit("/", 1)[-1]
    (RESULTS_DIR / f"profile-{TOKENIZERS_VERSION}-{short}.json").write_text(
        json.dumps(
            {
                "tokenizers": TOKENIZERS_VERSION,
                "model": short,
                "documents": best.documents,
                "tokens": best.tokens,
                "total_wall_ms": best_wall * 1e3,
                "stages": {
                    s: {"wall_ms": best.stages[s].wall * 1e3, "cpu_ms": best.stages[s].cpu * 1e3}
                    for s in STAGES
                },
            },
            indent=2,
        )
    )


def test_compare_tokenizers_versions(capsys: pytest.CaptureFixture[str]) -> None:
    results: dict[tuple[str, int], dict[str, Any]] = {}
    for path in RESULTS_DIR.glob("profile-*.json"):
        r = json.loads(path.read_text())
        results[(r["model"], int(r["tokenizers"].split(".")[0]))] = r
    models = sorted({m for m, major in results if (m, 0) in results and (m, 1) in results})
    if not models:
        pytest.skip("run the profile under tokenizers v0 and v1 first")
    with capsys.disabled():
        print("\nshare of wall time per stage, tokenizers v0 -> v1 (total ms in brackets)")
        print(f"{'model':<22}{'stage':<14}{'v0':>14}{'v1':>14}{'v0 ms':>9}{'v1 ms':>9}")
        for m in models:
            v0, v1 = results[(m, 0)], results[(m, 1)]
            for st in STAGES:
                a, b = v0["stages"][st]["wall_ms"], v1["stages"][st]["wall_ms"]
                print(
                    f"{m:<22}{st:<14}{100 * a / v0['total_wall_ms']:>13.1f}%"
                    f"{100 * b / v1['total_wall_ms']:>13.1f}%{a:>9.1f}{b:>9.1f}"
                )
            print(
                f"{m:<22}{'total':<14}{'':>14}{'':>14}"
                f"{v0['total_wall_ms']:>9.1f}{v1['total_wall_ms']:>9.1f}"
            )
