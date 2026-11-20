"""Embedding backends behind one small interface.

The default is a local open model run in-process (no account, no API cost).
A Bedrock backend can be added later without touching ingestion or retrieval.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (chunks)."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (some models use a different prompt for queries)."""


class FastEmbedEmbedder:
    """Local ONNX model via fastembed. The model downloads once and is cached."""

    def __init__(self, model_name: str = DEFAULT_MODEL, dim: int = DEFAULT_DIM, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        self.name = model_name
        self.dim = dim
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts, batch_size=64)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class HashEmbedder:
    """Deterministic fake for tests: identical text gives an identical unit vector."""

    def __init__(self, dim: int = 16, name: str = "hash-test"):
        self.name = name
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        raw = [b - 127.5 for b in hashlib.sha256(text.encode("utf-8")).digest()]
        values = [raw[i % len(raw)] * (1 + (i // len(raw))) for i in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
