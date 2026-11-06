"""Dense retrieval: sentence-transformer embeddings, cosine similarity, MaxP over chunks.

A sentence-transformer reads at most ~256 word pieces, so a long e-mail is split into
overlapping word windows ("chunks"), every chunk is embedded, and an e-mail is scored by its
best-matching chunk (the MaxP strategy of Dai & Callan, 2019). Search is an exact matrix-vector
product - at this scale an approximate-nearest-neighbour library would only add moving parts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from maillens.config import CHUNK_STRIDE, CHUNK_WORDS, EMBED_MODEL, MAX_CHUNKS_PER_EMAIL
from maillens.rankers.base import Hit, top_k_indices

log = logging.getLogger(__name__)


class Encoder(Protocol):
    """Turns texts into L2-normalised float32 vectors of width ``dim``."""

    dim: int

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """CPU encoder backed by the ``sentence-transformers`` library (loaded lazily)."""

    def __init__(self, model_name: str = EMBED_MODEL, *, max_seq_length: int = 256) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, keep it lazy

        self._model = SentenceTransformer(model_name, device="cpu")
        self._model.max_seq_length = max_seq_length
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"{model_name} does not report an embedding size")
        self.dim = int(dim)
        self.model_name = model_name

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


# ----------------------------------------------------------------------------- chunking
def split_into_chunks(
    text: str,
    *,
    words: int = CHUNK_WORDS,
    stride: int = CHUNK_STRIDE,
    max_chunks: int = MAX_CHUNKS_PER_EMAIL,
) -> list[str]:
    """Overlapping word windows over ``text``; always at least one (possibly empty) chunk."""
    if words < 1 or stride < 1 or max_chunks < 1:
        raise ValueError("words, stride and max_chunks must be positive")
    if stride > words:
        raise ValueError("stride larger than the window would skip text")
    tokens = text.split()
    if not tokens:
        return [""]
    chunks: list[str] = []
    for start in range(0, len(tokens), stride):
        chunks.append(" ".join(tokens[start : start + words]))
        if start + words >= len(tokens) or len(chunks) >= max_chunks:
            break
    return chunks


def chunk_inputs(subject: str, text: str) -> list[str]:
    """Texts to embed for one e-mail: each chunk prefixed with the subject for context."""
    prefix = f"{subject.strip()}. " if subject.strip() else ""
    return [f"{prefix}{chunk}".strip() for chunk in split_into_chunks(text)]


# ----------------------------------------------------------------------------- embedding
def _fingerprint(texts: Sequence[str], model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for text in texts:
        digest.update(len(text).to_bytes(4, "little"))
        digest.update(text.encode("utf-8", errors="replace"))
    return digest.hexdigest()


def embed_chunks(
    texts: Sequence[str],
    encoder: Encoder,
    out_dir: Path,
    *,
    model_name: str = EMBED_MODEL,
    shard_size: int = 8192,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed ``texts`` in shards saved to ``out_dir``; an interrupted run resumes where it stopped.

    Shards are stored as float16 (half the disk) and returned as float32. A manifest ties the
    shards to the exact input and model; reusing a folder for different input is refused rather
    than silently mixing vectors.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "model": model_name,
        "n_texts": len(texts),
        "shard_size": shard_size,
        "fingerprint": _fingerprint(texts, model_name),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                f"{out_dir} holds embeddings for different input; delete the folder to recompute"
            )
    else:
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    parts: list[np.ndarray] = []
    n_shards = -(-len(texts) // shard_size)
    for i, start in enumerate(range(0, len(texts), shard_size)):
        shard = out_dir / f"shard_{i:05d}.npy"
        expected = min(shard_size, len(texts) - start)
        if shard.exists():
            part = np.load(shard)
            if part.shape != (expected, encoder.dim):
                raise ValueError(
                    f"{shard} has shape {part.shape}, expected {(expected, encoder.dim)}"
                )
        else:
            part = encoder.encode(texts[start : start + shard_size], batch_size=batch_size)
            part = part.astype(np.float16)
            tmp = shard.with_name(shard.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.save(fh, part)
            os.replace(tmp, shard)
            log.info("embedded shard %d/%d (%d chunks)", i + 1, n_shards, expected)
        parts.append(part)
    return np.concatenate(parts).astype(np.float32)


# ----------------------------------------------------------------------------- ranker
class DenseRanker:
    name = "dense"

    def __init__(
        self,
        vectors: np.ndarray,
        chunk_doc: np.ndarray,
        n_docs: int,
        encoder: Encoder,
    ) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != chunk_doc.shape[0]:
            raise ValueError("need exactly one vector per chunk")
        if vectors.shape[1] != encoder.dim:
            raise ValueError("vector width does not match the encoder")
        if chunk_doc.shape[0] and np.any(np.diff(chunk_doc) < 0):
            raise ValueError("chunk_doc must be sorted so each document's chunks are contiguous")
        counts = np.bincount(chunk_doc, minlength=n_docs)
        if counts.shape[0] != n_docs or (counts == 0).any():
            raise ValueError("every document needs at least one chunk")
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.chunk_doc = chunk_doc
        self.n_docs = n_docs
        self.encoder = encoder
        self.doc_starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
        self._doc_ends = np.cumsum(counts).astype(np.int64)

    def doc_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """MaxP: each document's best chunk similarity."""
        sims = self.vectors @ query_vector
        return np.maximum.reduceat(sims, self.doc_starts)

    def search(self, query: str, k: int, *, allowed: np.ndarray | None = None) -> list[Hit]:
        scores = self.doc_scores(self.encoder.encode([query])[0])
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        picked = top_k_indices(scores, k)
        return [Hit(int(d), float(scores[d]), r) for r, d in enumerate(picked, start=1)]

    def rank_vectors(self, query_vectors: np.ndarray, k: int) -> list[np.ndarray]:
        """Top-``k`` document ids (best first) for each already-encoded query.

        Encoding every query in one batch is much faster than one call per query, which is what
        the evaluation over thousands of queries needs.
        """
        return [
            top_k_indices(self.doc_scores(vector), k).astype(np.int32) for vector in query_vectors
        ]

    def best_chunk(self, query: str, doc_id: int) -> tuple[int, float]:
        """(index of the best chunk within the document, its cosine similarity)."""
        q = self.encoder.encode([query])[0]
        start, end = int(self.doc_starts[doc_id]), int(self._doc_ends[doc_id])
        sims = self.vectors[start:end] @ q
        best = int(np.argmax(sims))
        return best, float(sims[best])
