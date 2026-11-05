"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    RRF(d) = sum over rankers m of   w_m / (k + rank_m(d))

Only ranks are used, so the (incomparable) scores of TF-IDF, BM25 and cosine similarity never
have to be normalised. A document missing from a ranker's list simply earns nothing from it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from maillens.config import RRF_K
from maillens.rankers.base import Hit, Ranker


@dataclass(frozen=True, slots=True)
class FusedHit:
    doc_id: int
    score: float
    ranks: Mapping[str, int]  # ranker name -> 1-based rank (only rankers that listed it)
    contributions: Mapping[str, float]  # ranker name -> its share of ``score``


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[int]],
    *,
    k: int = RRF_K,
    weights: Mapping[str, float] | None = None,
) -> list[FusedHit]:
    """Fuse several ranked lists of document ids (best first, no duplicates within a list).

    Ties are broken by the best rank any ranker gave the document, then by document id, so the
    output is deterministic.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if weights is not None:
        unknown = set(weights) - set(rankings)
        if unknown:
            raise ValueError(f"weights given for unknown rankers: {sorted(unknown)}")

    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    shares: dict[int, dict[str, float]] = {}
    for name, ranking in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        if weight <= 0:
            raise ValueError(f"weight for {name!r} must be positive, got {weight}")
        seen: set[int] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                raise ValueError(f"ranker {name!r} lists document {doc_id} more than once")
            seen.add(doc_id)
            share = weight / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + share
            ranks.setdefault(doc_id, {})[name] = rank
            shares.setdefault(doc_id, {})[name] = share

    ordered = sorted(scores, key=lambda d: (-scores[d], min(ranks[d].values()), d))
    return [FusedHit(d, scores[d], ranks[d], shares[d]) for d in ordered]


class HybridRanker:
    """Runs several rankers and fuses their lists with RRF."""

    name = "hybrid"

    def __init__(
        self,
        members: Sequence[Ranker],
        *,
        k: int = RRF_K,
        depth: int = 200,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        if not members:
            raise ValueError("a hybrid ranker needs at least one member")
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.members = list(members)
        self.k = k
        self.depth = depth
        self.weights = weights

    def member_rankings(
        self, query: str, allowed: np.ndarray | None = None
    ) -> dict[str, list[int]]:
        return {
            m.name: [h.doc_id for h in m.search(query, self.depth, allowed=allowed)]
            for m in self.members
        }

    def search_detailed(
        self, query: str, k: int, *, allowed: np.ndarray | None = None
    ) -> list[FusedHit]:
        fused = reciprocal_rank_fusion(
            self.member_rankings(query, allowed), k=self.k, weights=self.weights
        )
        return fused[:k]

    def search(self, query: str, k: int, *, allowed: np.ndarray | None = None) -> list[Hit]:
        return [
            Hit(h.doc_id, h.score, rank)
            for rank, h in enumerate(self.search_detailed(query, k, allowed=allowed), start=1)
        ]
