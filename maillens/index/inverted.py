"""A compact in-memory inverted index.

Layout ("CSR by term"): the postings of every term sit in two flat arrays, ``doc_ids`` and
``tfs``. ``indptr[t]:indptr[t + 1]`` is the slice that belongs to term id ``t`` and is sorted by
document id. Looking a term up is two array reads, and scoring a query is a loop over a handful
of slices - there are no per-document Python objects at query time.
"""

from __future__ import annotations

import json
import os
from array import array
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np


def _atomic_write(path: Path, write: object) -> None:
    """Write via a temp file and rename, so an interrupted run never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        write(fh)  # type: ignore[operator]
    os.replace(tmp, path)


class InvertedIndex:
    """term -> postings (document id, term frequency), plus the document lengths."""

    def __init__(
        self,
        terms: list[str],
        indptr: np.ndarray,
        doc_ids: np.ndarray,
        tfs: np.ndarray,
        doc_len: np.ndarray,
    ) -> None:
        if indptr.shape != (len(terms) + 1,):
            raise ValueError("indptr must have one more entry than there are terms")
        if doc_ids.shape != tfs.shape:
            raise ValueError("doc_ids and tfs must have the same length")
        if int(indptr[-1]) != doc_ids.shape[0]:
            raise ValueError("indptr[-1] must equal the number of postings")
        if doc_len.shape[0] == 0:
            raise ValueError("an index needs at least one document")
        self.terms = terms
        self.vocab: dict[str, int] = {t: i for i, t in enumerate(terms)}
        self.indptr = indptr
        self.doc_ids = doc_ids
        self.tfs = tfs
        self.doc_len = doc_len
        self.df: np.ndarray = np.diff(indptr).astype(np.int32)

    # ------------------------------------------------------------------ properties
    @property
    def n_docs(self) -> int:
        return int(self.doc_len.shape[0])

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    @property
    def n_postings(self) -> int:
        return int(self.doc_ids.shape[0])

    @property
    def avgdl(self) -> float:
        return float(self.doc_len.mean())

    # ------------------------------------------------------------------ lookups
    def term_id(self, term: str) -> int | None:
        return self.vocab.get(term)

    def posting_slice(self, term_id: int) -> slice:
        return slice(int(self.indptr[term_id]), int(self.indptr[term_id + 1]))

    def postings(self, term_id: int) -> tuple[np.ndarray, np.ndarray]:
        """(document ids, term frequencies) of one term, ascending by document id."""
        sl = self.posting_slice(term_id)
        return self.doc_ids[sl], self.tfs[sl]

    def posting_position(self, term_id: int, doc_id: int) -> int | None:
        """Absolute position of (term, doc) in the flat arrays, or None if the term is absent."""
        sl = self.posting_slice(term_id)
        docs = self.doc_ids[sl]
        j = int(np.searchsorted(docs, doc_id))
        if j < docs.shape[0] and int(docs[j]) == doc_id:
            return sl.start + j
        return None

    def term_of_posting(self) -> np.ndarray:
        """Term id of every posting (used to broadcast per-term values such as idf)."""
        return np.repeat(np.arange(self.n_terms, dtype=np.int32), self.df)

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(cls, docs: Iterable[Sequence[str]]) -> InvertedIndex:
        """Index an iterable of already-analysed documents (lists of terms)."""
        vocab: dict[str, int] = {}
        term_col = array("i")
        doc_col = array("i")
        tf_col = array("i")
        lengths = array("i")
        for doc_id, tokens in enumerate(docs):
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                tid = vocab.get(term)
                if tid is None:
                    tid = vocab[term] = len(vocab)
                term_col.append(tid)
                doc_col.append(doc_id)
                tf_col.append(tf)
        if not lengths:
            raise ValueError("cannot index an empty corpus")

        term_arr = np.frombuffer(term_col, dtype=np.int32)
        # Documents were appended in order, so a *stable* sort by term id keeps each term's
        # postings ascending by document id.
        order = np.argsort(term_arr, kind="stable")
        counts = np.bincount(term_arr, minlength=len(vocab))
        indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        return cls(
            terms=list(vocab),  # dicts preserve insertion order, i.e. term id order
            indptr=indptr,
            doc_ids=np.frombuffer(doc_col, dtype=np.int32)[order],
            tfs=np.frombuffer(tf_col, dtype=np.int32)[order],
            doc_len=np.frombuffer(lengths, dtype=np.int32).copy(),
        )

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("indptr", "doc_ids", "tfs", "doc_len"):
            arr = getattr(self, name)
            _atomic_write(directory / f"{name}.npy", lambda fh, a=arr: np.save(fh, a))
        payload = json.dumps(self.terms).encode("utf-8")
        _atomic_write(directory / "terms.json", lambda fh: fh.write(payload))

    @classmethod
    def load(cls, directory: Path, *, mmap: bool = True) -> InvertedIndex:

        def read(name: str) -> np.ndarray:
            path = directory / f"{name}.npy"
            return np.load(path, mmap_mode="r") if mmap else np.load(path)

        arrays = {name: read(name) for name in ("indptr", "doc_ids", "tfs", "doc_len")}
        terms = json.loads((directory / "terms.json").read_text(encoding="utf-8"))
        return cls(terms=terms, **arrays)
