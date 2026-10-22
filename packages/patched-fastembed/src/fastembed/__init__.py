import importlib.metadata

from fastembed.image import ImageEmbedding
from fastembed.late_interaction import LateInteractionTextEmbedding
from fastembed.late_interaction_multimodal import LateInteractionMultimodalEmbedding
from fastembed.sparse import SparseEmbedding, SparseTextEmbedding
from fastembed.text import TextEmbedding

for _dist in ("fastembed", "fastembed-gpu", "patched-fastembed"):
    try:
        version = importlib.metadata.version(_dist)
        break
    except importlib.metadata.PackageNotFoundError:
        continue
else:
    version = "0.0.0"

__version__ = version
__all__ = [
    "ImageEmbedding",
    "LateInteractionMultimodalEmbedding",
    "LateInteractionTextEmbedding",
    "SparseEmbedding",
    "SparseTextEmbedding",
    "TextEmbedding",
]
