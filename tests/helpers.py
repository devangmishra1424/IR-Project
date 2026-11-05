"""Test doubles shared by several test modules."""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np


class HashingEncoder:
    """Deterministic bag-of-words stand-in for a neural encoder (tests only).

    Texts that share words get a high cosine similarity, which is all the dense-retrieval
    plumbing needs to be exercised without downloading a model.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                out[i, zlib.crc32(token.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
