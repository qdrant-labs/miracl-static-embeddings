import json
from collections.abc import Iterator
from typing import Any


def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    """Read a local corpus file with one {"docid", "title", "text"} object per line."""
    with open(path) as f:
        for line in f:
            yield json.loads(line)
