"""Turn the prepared corpus into searchable artefacts: the sparse index and dense embeddings."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from maillens import config
from maillens.index.inverted import InvertedIndex
from maillens.rankers.dense import DenseRanker, Encoder, chunk_inputs, embed_chunks
from maillens.text.analyzer import Analyzer

log = logging.getLogger(__name__)

SPARSE_DIR = "sparse"
DENSE_DIR = "dense"


def _require_positional_ids(emails: pd.DataFrame) -> None:
    """Every index is addressed by document id, which must equal the row position."""
    if not np.array_equal(emails["doc_id"].to_numpy(), np.arange(len(emails))):
        raise ValueError("doc_id must equal the row position (0..N-1)")


def index_text(subject: str, body: str) -> str:
    """What the sparse rankers see for one e-mail: subject line, then body."""
    return f"{subject}\n{body}"


def build_sparse_index(
    emails: pd.DataFrame,
    analyzer: Analyzer,
    out_dir: Path,
    *,
    body_column: str = "text",
) -> InvertedIndex:
    """Analyse every e-mail and build + save its inverted index.

    ``body_column`` selects the text that is indexed (the cleaned body by default); the raw body
    can be passed for the cleaning ablation.
    """
    _require_positional_ids(emails)
    docs = (
        analyzer(index_text(s, b))
        for s, b in zip(emails["subject"], emails[body_column], strict=True)
    )
    index = InvertedIndex.build(docs)
    index.save(out_dir)
    log.info(
        "sparse index: %d docs, %d terms, %d postings -> %s",
        index.n_docs,
        index.n_terms,
        index.n_postings,
        out_dir,
    )
    return index


def build_dense_chunks(emails: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """(texts to embed, owning document id of each text); chunks of one e-mail are adjacent."""
    _require_positional_ids(emails)
    texts: list[str] = []
    owners: list[int] = []
    for doc_id, (subject, body) in enumerate(zip(emails["subject"], emails["text"], strict=True)):
        parts = chunk_inputs(subject, body)
        texts.extend(parts)
        owners.extend([doc_id] * len(parts))
    return texts, np.asarray(owners, dtype=np.int32)


def _save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    os.replace(tmp, path)


def build_dense_index(
    emails: pd.DataFrame,
    encoder: Encoder,
    out_dir: Path,
    *,
    model_name: str = config.EMBED_MODEL,
) -> None:
    """Embed every chunk (resumable, see ``embed_chunks``) and save the finished index."""
    texts, chunk_doc = build_dense_chunks(emails)
    log.info("dense index: %d e-mails -> %d chunks", len(emails), len(texts))
    vectors = embed_chunks(texts, encoder, out_dir / "shards", model_name=model_name)
    _save_npy(out_dir / "chunk_doc.npy", chunk_doc)
    _save_npy(out_dir / "vectors.npy", vectors.astype(np.float16))
    meta = {
        "model": model_name,
        "n_docs": len(emails),
        "n_chunks": len(texts),
        "dim": int(vectors.shape[1]),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def load_dense_ranker(n_docs: int, encoder: Encoder, directory: Path) -> DenseRanker:
    """Load a finished dense index; refuses an index built for a different corpus size."""
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if meta["n_docs"] != n_docs:
        raise ValueError(
            f"dense index holds {meta['n_docs']} documents but the corpus has {n_docs}; rebuild it"
        )
    vectors = np.load(directory / "vectors.npy").astype(np.float32)
    chunk_doc = np.load(directory / "chunk_doc.npy")
    return DenseRanker(vectors, chunk_doc, n_docs, encoder)
