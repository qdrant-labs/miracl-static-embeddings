import os
from collections.abc import Iterable, Sequence
from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tokenizers import Encoding, Tokenizer

from ..common.onnx_model import EmbeddingWorker, OnnxModel, OnnxOutputContext, T
from ..common.preprocessor_utils import load_tokenizer
from ..common.types import Device, NumpyArray, OnnxProvider
from ..common.utils import iter_batch
from ..parallel_processor import ParallelWorkerPool

# Texts of a batch are sorted by length and run in chunks of this size, so that short
# texts are not padded up to the longest one. Pass `micro_batch_size=None` (or 0) to
# `embed` to disable it.
DEFAULT_MICRO_BATCH_SIZE = 32


class OnnxTextModel(OnnxModel[T]):
    ONNX_OUTPUT_NAMES: list[str] | None = None

    @classmethod
    def _get_worker_class(cls) -> type["TextEmbeddingWorker[T]"]:
        raise NotImplementedError("Subclasses must implement this method")

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[T]:
        """Post-process the ONNX model output to convert it into a usable format.

        Args:
            output (OnnxOutputContext): The raw output from the ONNX model.
            **kwargs: Additional keyword arguments that may be needed by specific implementations.

        Returns:
            Iterable[T]: Post-processed output as an iterable of type T.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer: Tokenizer | None = None

    def _load_tokenizer(self, model_dir: Path) -> None:
        self.tokenizer = load_tokenizer(model_dir=model_dir)

    def _ensure_tokenizer(self) -> None:
        """Load the tokenizer if it has not been loaded yet, leaving the onnx model alone.

        Tokenizer files are lightweight, so counting tokens does not have to resolve
        `lazy_load` into a full onnx session.
        """
        if getattr(self, "tokenizer", None) is None:
            self._load_tokenizer(model_dir=self._model_dir)

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray | NDArray[np.int64]]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    def _load_onnx_model(
        self,
        model_dir: Path,
        model_file: str,
        threads: int | None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_id: int | None = None,
        extra_session_options: dict[str, Any] | None = None,
        additional_files: list[str] | None = None,
    ) -> None:
        super()._load_onnx_model(
            model_dir=model_dir,
            model_file=model_file,
            threads=threads,
            providers=providers,
            cuda=cuda,
            device_id=device_id,
            extra_session_options=extra_session_options,
            additional_files=additional_files,
        )
        self._ensure_tokenizer()

    def load_onnx_model(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def tokenize(self, documents: list[str], **kwargs: Any) -> list[Encoding]:
        return self.tokenizer.encode_batch(documents)  # type: ignore[union-attr]

    def onnx_embed(
        self,
        documents: list[str],
        **kwargs: Any,
    ) -> OnnxOutputContext:
        encoded = self.tokenize(documents, **kwargs)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        input_names = {node.name for node in self.model.get_inputs()}  # type: ignore[union-attr]
        onnx_input: dict[str, NumpyArray] = {"input_ids": input_ids}
        if "attention_mask" in input_names:
            onnx_input["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            onnx_input["token_type_ids"] = np.zeros_like(input_ids)
        onnx_input = self._preprocess_onnx_input(onnx_input, **kwargs)
        model_output = self._run_model(
            onnx_input=onnx_input, onnx_output_names=self.ONNX_OUTPUT_NAMES
        )

        return OnnxOutputContext(
            model_output=model_output,
            attention_mask=onnx_input.get("attention_mask", attention_mask),
            input_ids=onnx_input.get("input_ids", input_ids),
        )

    def _run_model(
        self, onnx_input: dict[str, Any], onnx_output_names: list[str] | None = None
    ) -> NumpyArray:
        return self.model.run(onnx_output_names, onnx_input)[0]  # type: ignore[union-attr]

    def _embed_batch_sorted(
        self, batch: list[str], micro_batch_size: int | None, **kwargs: Any
    ) -> Iterable[T]:
        """Embed `batch` in length-sorted chunks, yielding results in the input order.

        Texts are sorted by character length (a cheap proxy for the token count that
        avoids tokenizing twice), so each chunk is padded to a similar length.
        """
        if not micro_batch_size or len(batch) <= micro_batch_size:
            yield from self._post_process_onnx_output(
                self.onnx_embed(batch, **kwargs), **kwargs
            )
            return

        order = sorted(range(len(batch)), key=lambda i: len(batch[i]))
        results: list[Any] = [None] * len(batch)
        for start in range(0, len(order), micro_batch_size):
            indices = order[start : start + micro_batch_size]
            outputs = list(
                self._post_process_onnx_output(
                    self.onnx_embed([batch[i] for i in indices], **kwargs), **kwargs
                )
            )
            if len(outputs) != len(indices):
                raise RuntimeError(
                    f"Expected one output per document, got {len(outputs)} for "
                    f"{len(indices)} documents"
                )
            for i, output in zip(indices, outputs):
                results[i] = output
        yield from results

    def _embed_documents(
        self,
        model_name: str,
        cache_dir: str,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        local_files_only: bool = False,
        specific_model_path: str | None = None,
        extra_session_options: dict[str, Any] | None = None,
        micro_batch_size: int | None = DEFAULT_MICRO_BATCH_SIZE,
        **kwargs: Any,
    ) -> Iterable[T]:
        is_small = False

        if isinstance(documents, str):
            documents = [documents]
            is_small = True

        if isinstance(documents, list) and len(documents) < batch_size:
            is_small = True

        if parallel is None or is_small:
            if not hasattr(self, "model") or self.model is None:
                self.load_onnx_model()
            for batch in iter_batch(documents, batch_size):
                yield from self._embed_batch_sorted(batch, micro_batch_size, **kwargs)
        else:
            if parallel == 0:
                parallel = os.cpu_count()

            start_method = (
                "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
            )
            params = {
                "model_name": model_name,
                "cache_dir": cache_dir,
                "providers": providers,
                "local_files_only": local_files_only,
                "specific_model_path": specific_model_path,
                **kwargs,
                **self._get_worker_init_kwargs(),
            }

            if extra_session_options is not None:
                params.update(extra_session_options)

            pool = ParallelWorkerPool(
                num_workers=parallel or 1,
                worker=self._get_worker_class(),
                cuda=cuda,
                device_ids=device_ids,
                start_method=start_method,
            )
            for batch in pool.ordered_map(iter_batch(documents, batch_size), **params):
                yield from self._post_process_onnx_output(batch, **kwargs)  # type: ignore

    def _token_count(
        self, texts: str | Iterable[str], batch_size: int = 1024, **_: Any
    ) -> int:
        self._ensure_tokenizer()
        token_num = 0
        assert self.tokenizer is not None
        texts = [texts] if isinstance(texts, str) else texts
        for batch in iter_batch(texts, batch_size):
            for tokens in self.tokenizer.encode_batch(batch):
                token_num += sum(tokens.attention_mask)

        return token_num


class TextEmbeddingWorker(EmbeddingWorker[T]):
    def process(
        self, items: Iterable[tuple[int, Any]]
    ) -> Iterable[tuple[int, OnnxOutputContext]]:
        for idx, batch in items:
            onnx_output = self.model.onnx_embed(batch)
            yield idx, onnx_output
