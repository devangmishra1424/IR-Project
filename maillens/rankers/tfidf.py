"""TF-IDF with cosine similarity (the classical vector-space baseline).

    w(t, d) = (1 + ln tf(t, d)) * idf(t)          idf(t) = ln((1 + N) / (1 + df(t))) + 1

Document vectors and the query vector are L2-normalised, so the score of a document is the
cosine between the two. The idf smoothing and sub-linear tf are the same as scikit-learn's
``TfidfVectorizer`` defaults, which is what the tests compare against.
"""

from __future__ import annotations

import math

import numpy as np

from maillens.index.inverted import InvertedIndex
from maillens.rankers.sparse import SparseRanker
from maillens.text.analyzer import Analyzer


class TfidfRanker(SparseRanker):
    name = "tfidf"

    def __init__(self, index: InvertedIndex, analyzer: Analyzer) -> None:
        n = index.n_docs
        self.idf = (np.log((1.0 + n) / (1.0 + index.df)) + 1.0).astype(np.float32)
        raw = (1.0 + np.log(index.tfs.astype(np.float32))) * self.idf[index.term_of_posting()]
        squared = np.bincount(index.doc_ids, weights=raw.astype(np.float64) ** 2, minlength=n)
        norms = np.sqrt(squared)
        norms[norms == 0] = 1.0  # empty documents have no postings; avoid 0/0
        weights = (raw / norms[index.doc_ids].astype(np.float32)).astype(np.float32)
        super().__init__(index, analyzer, weights)

    def query_vector(self, query: str) -> dict[int, float]:
        raw = {
            tid: (1.0 + math.log(count)) * float(self.idf[tid])
            for tid, count in self.known_term_counts(query).items()
        }
        norm = math.sqrt(sum(v * v for v in raw.values()))
        if norm == 0.0:
            return {}
        return {tid: v / norm for tid, v in raw.items()}
