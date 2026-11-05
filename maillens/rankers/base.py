"""Shared types for all rankers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class Hit:
    """One ranked result. ``rank`` is 1-based."""

    doc_id: int
    score: float
    rank: int


class Ranker(Protocol):
    """Anything that can turn a query into a ranked list of document ids."""

    name: str

    def search(self, query: str, k: int, *, allowed: np.ndarray | None = None) -> list[Hit]:
        """Top-``k`` documents for ``query``.

        ``allowed`` is an optional boolean mask over document ids (a metadata filter); documents
        outside it are never returned.
        """
        ...


def top_k_indices(scores: np.ndarray, k: int, *, positive_only: bool = False) -> np.ndarray:
    """Indices of the ``k`` best scores, best first.

    Deterministic: ties are broken by ascending index, including ties at the cut-off.
    Non-finite scores (``-inf`` is used to mask documents out) are never returned, and with
    ``positive_only`` neither are zero scores ("the document matched nothing").
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    valid = np.isfinite(scores)
    if positive_only:
        valid &= scores > 0
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return candidates
    cand_scores = scores[candidates]
    if candidates.size > k:
        kth_best = np.partition(cand_scores, candidates.size - k)[candidates.size - k]
        keep = cand_scores >= kth_best
        candidates, cand_scores = candidates[keep], cand_scores[keep]
    order = np.lexsort((candidates, -cand_scores))
    return candidates[order][:k]
