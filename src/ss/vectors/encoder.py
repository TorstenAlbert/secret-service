"""Embedding encoder using SentenceTransformer with lazy loading."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_DIMENSION = 384


class EmbeddingEncoder:
    """Encodes text into normalized embeddings using SentenceTransformer.

    The underlying model is lazy-loaded on the first encode call to avoid
    importing heavy ML dependencies at import time.
    """

    def __init__(self) -> None:
        self._model: "SentenceTransformer | None" = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return _DIMENSION

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Return a normalized float32 embedding of shape (384,)."""
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return np.array(embedding, dtype=np.float32)

    def encode_bytes(self, text: str) -> bytes:
        """Return float32 bytes suitable for sqlite-vss insertion."""
        arr = self.encode(text)
        return arr.astype(np.float32).tobytes()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._model = SentenceTransformer(_MODEL_NAME)
        return self._model
