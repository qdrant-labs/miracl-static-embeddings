import json
import sys
from pathlib import Path
from typing import Any

from tokenizers import Padding, Tokenizer, Truncation

from fastembed.image.transform.operators import Compose


def _valid_context(value: Any) -> int | None:
    """Return `value` if it can be used as a truncation limit, `None` otherwise.

    Config files do not always carry a real limit: transformers writes `model_max_length` as
    1e30 when the value is unknown, and some repos ship a 0 or a null. `enable_truncation`
    raises an `OverflowError` on the former and silently produces empty encodings on the
    latter, so both are rejected here rather than passed through.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if not 0 < value <= sys.maxsize:
        return None
    return value


def _resolve_max_context(tokenizer_config: dict[str, Any], model_dir: Path) -> int:
    """Pick the truncation limit, preferring the stricter of the two tokenizer config keys.

    `config.json:max_position_embeddings` deliberately is not used as a fallback: it is the size
    of the position table, not the usable context, and the two differ per architecture, e.g.
    roberta reports 514 for a usable 512.
    """
    candidates = [
        context
        for context in (
            _valid_context(tokenizer_config.get("model_max_length")),
            _valid_context(tokenizer_config.get("max_length")),
        )
        if context is not None
    ]
    if not candidates:
        raise ValueError(
            f"Could not determine the maximum context length for {model_dir}. Set a positive "
            "`model_max_length` or `max_length` in tokenizer_config.json."
        )

    return min(candidates)


def load_tokenizer(model_dir: Path) -> Tokenizer:
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise ValueError(f"Could not find tokenizer.json in {model_dir}")

    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        raise ValueError(f"Could not find tokenizer_config.json in {model_dir}")

    # config.json is optional: transformers v5 no longer writes it for every model.
    config_path = model_dir / "config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        with open(str(config_path)) as config_file:
            config = json.load(config_file)

    with open(str(tokenizer_config_path)) as tokenizer_config_file:
        tokenizer_config = json.load(tokenizer_config_file)

    max_context = _resolve_max_context(tokenizer_config, model_dir)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # enable_truncation resets the direction to right unless it is passed. The direction is
    # resolved as in transformers: tokenizer_config.json, then tokenizer.json, then right.
    tokenizer.truncation = Truncation(
        max_length=max_context,
        direction=tokenizer_config.get("truncation_side")
        or tokenizer.truncation.direction
        if tokenizer.truncation is not None
        else "right",
    )

    # Padding is always normalized to batch-longest. A serialized fixed length shorter than the
    # truncation limit leaves longer encodings untouched, which produces ragged batches, and a
    # fixed length equal to it pads every batch to the maximum. Pad token metadata is taken from
    # the serialized settings. The direction follows transformers: `padding_side` from
    # tokenizer_config.json wins over the serialized direction, since some models pad on the left
    # and set it only in the config.
    padding = tokenizer.padding
    pad_token = (
        padding.pad_token if padding is not None else tokenizer_config.get("pad_token")
    )
    if pad_token is None:
        raise ValueError(f"Could not find a pad token for {model_dir}")

    # The vocabulary is the last resort, not a hardcoded 0: that silently disagrees with
    # `pad_token` for every model whose pad token is not the first entry.
    pad_id = padding.pad_id if padding is not None else config.get("pad_token_id")
    if pad_id is None:
        pad_id = tokenizer.encode(pad_token, padding=None).ids[0]
    if pad_id is None:
        raise ValueError(
            f"Could not resolve an id for the pad token {pad_token!r} in {model_dir}"
        )

    tokenizer.padding = Padding(
        direction=tokenizer_config.get("padding_side") or padding.direction
        if padding is not None
        else "right",
        pad_id=pad_id,
        pad_type_id=padding.pad_type_id if padding is not None else 0,
        pad_token=pad_token,
        pad_to_multiple_of=padding.pad_to_multiple_of if padding is not None else None,
        length=None,
    )

    return tokenizer


def load_preprocessor(model_dir: Path) -> Compose:
    preprocessor_config_path = model_dir / "preprocessor_config.json"
    if not preprocessor_config_path.exists():
        raise ValueError(f"Could not find preprocessor_config.json in {model_dir}")

    with open(str(preprocessor_config_path)) as preprocessor_config_file:
        preprocessor_config = json.load(preprocessor_config_file)
        transforms = Compose.from_config(preprocessor_config)
    return transforms
