"""Okapi BM25 (the probabilistic baseline).

    score(q, d) = sum over query terms t of
                  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))      (never negative, as in Lucene)

``k1`` controls term-frequency saturation, ``b`` the strength of document-length normalisation.
Each distinct query term counts once; repeating a word in the query does not change the score.
Per-posting weights are computed once at construction, so a query is just a few array slices.
"""

from __future__ import annotations

import numpy as np

from maillens.config import BM25_B, BM25_K1
from maillens.index.inverted import InvertedIndex
from maillens.rankers.sparse import SparseRanker
from maillens.text.analyzer import Analyzer


class Bm25Ranker(SparseRanker):
    name = "bm25"

    def __init__(
        self,
        index: InvertedIndex,
        analyzer: Analyzer,
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be within [0, 1], got {b}")
        self.k1, self.b = k1, b
        n = index.n_docs
        df = index.df.astype(np.float64)
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        doc_len = index.doc_len.astype(np.float32)
        tf = index.tfs.astype(np.float32)
        length_norm = k1 * (1.0 - b + b * doc_len[index.doc_ids] / np.float32(index.avgdl))
        weights = self.idf[index.term_of_posting()] * tf * (k1 + 1.0) / (tf + length_norm)
        super().__init__(index, analyzer, weights.astype(np.float32))

    def query_vector(self, query: str) -> dict[int, float]:
        return dict.fromkeys(self.known_term_counts(query), 1.0)
