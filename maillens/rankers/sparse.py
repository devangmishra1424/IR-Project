"""Term-at-a-time scoring over an inverted index with precomputed per-posting weights.

TF-IDF and BM25 differ only in (a) the weight stored for each posting and (b) how the query
terms are weighted. Everything else - walking the postings, accumulating scores, taking the
top-k, explaining a score - is shared here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from maillens.index.inverted import InvertedIndex
from maillens.rankers.base import Hit, top_k_indices
from maillens.text.analyzer import Analyzer


@dataclass(frozen=True, slots=True)
class TermContribution:
    """How much one query term added to one document's score."""

    term: str
    tf: int
    df: int
    contribution: float


class SparseRanker:
    name = "sparse"

    def __init__(self, index: InvertedIndex, analyzer: Analyzer, weights: np.ndarray) -> None:
        if weights.shape != index.doc_ids.shape:
            raise ValueError("need exactly one weight per posting")
        self.index = index
        self.analyzer = analyzer
        self.weights = weights

    # ---- subclasses decide how a query term is weighted --------------------------------
    def query_vector(self, query: str) -> dict[int, float]:
        raise NotImplementedError

    # ---- shared machinery --------------------------------------------------------------
    def known_term_counts(self, query: str) -> dict[int, int]:
        """Query term id -> occurrences; terms unknown to the index cannot match, so are dropped."""
        counts: Counter[int] = Counter()
        for term in self.analyzer(query):
            tid = self.index.term_id(term)
            if tid is not None:
                counts[tid] += 1
        return dict(counts)

    def score_all(self, query: str, allowed: np.ndarray | None = None) -> np.ndarray:
        """Score of every document (0.0 where nothing matched)."""
        scores = np.zeros(self.index.n_docs, dtype=np.float32)
        for tid, q_weight in self.query_vector(query).items():
            sl = self.index.posting_slice(tid)
            scores[self.index.doc_ids[sl]] += q_weight * self.weights[sl]
        if allowed is not None:
            scores[~allowed] = 0.0
        return scores

    def search(self, query: str, k: int, *, allowed: np.ndarray | None = None) -> list[Hit]:
        scores = self.score_all(query, allowed)
        picked = top_k_indices(scores, k, positive_only=True)
        return [Hit(int(d), float(scores[d]), r) for r, d in enumerate(picked, start=1)]

    def explain(self, query: str, doc_id: int) -> list[TermContribution]:
        """Per-term breakdown of ``doc_id``'s score, largest contribution first."""
        out: list[TermContribution] = []
        for tid, q_weight in self.query_vector(query).items():
            pos = self.index.posting_position(tid, doc_id)
            if pos is None:
                continue
            out.append(
                TermContribution(
                    term=self.index.terms[tid],
                    tf=int(self.index.tfs[pos]),
                    df=int(self.index.df[tid]),
                    contribution=float(q_weight * self.weights[pos]),
                )
            )
        return sorted(out, key=lambda c: -c.contribution)
