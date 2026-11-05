"""Ranking metrics with binary relevance, written from their textbook definitions.

Every function takes a ranked list of document ids (best first, no duplicates) and the set of
relevant ids. Metrics that divide by the number of relevant documents are undefined when there
are none, so they raise instead of quietly returning a misleading 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence, Set

RECALL_LEVELS: tuple[float, ...] = tuple(i / 10 for i in range(11))
METRIC_NAMES: tuple[str, ...] = ("P@10", "R@100", "MAP", "nDCG@10", "MRR", "Success@10")


def _require_relevant(relevant: Set[int]) -> None:
    if not relevant:
        raise ValueError("a query needs at least one relevant document")


def _require_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def precision_at_k(ranking: Sequence[int], relevant: Set[int], k: int) -> float:
    """Share of the top-k that is relevant (always divided by k, even if fewer were returned)."""
    _require_k(k)
    return sum(1 for d in ranking[:k] if d in relevant) / k


def recall_at_k(ranking: Sequence[int], relevant: Set[int], k: int) -> float:
    """Share of all relevant documents found in the top-k."""
    _require_k(k)
    _require_relevant(relevant)
    return sum(1 for d in ranking[:k] if d in relevant) / len(relevant)


def success_at_k(ranking: Sequence[int], relevant: Set[int], k: int) -> float:
    """1.0 if at least one relevant document is in the top-k."""
    _require_k(k)
    _require_relevant(relevant)
    return 1.0 if any(d in relevant for d in ranking[:k]) else 0.0


def reciprocal_rank(ranking: Sequence[int], relevant: Set[int], depth: int | None = None) -> float:
    """1 / rank of the first relevant document (0.0 if none within ``depth``)."""
    _require_relevant(relevant)
    for i, d in enumerate(ranking[:depth], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def average_precision(
    ranking: Sequence[int], relevant: Set[int], depth: int | None = None
) -> float:
    """Mean of the precision at each relevant hit; unretrieved relevant documents count as 0."""
    _require_relevant(relevant)
    hits, total = 0, 0.0
    for i, d in enumerate(ranking[:depth], start=1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def ndcg_at_k(ranking: Sequence[int], relevant: Set[int], k: int) -> float:
    """Normalised discounted cumulative gain with binary gains."""
    _require_k(k)
    _require_relevant(relevant)
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranking[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def interpolated_precision(
    ranking: Sequence[int],
    relevant: Set[int],
    levels: Sequence[float] = RECALL_LEVELS,
) -> list[float]:
    """11-point interpolated precision: at each recall level, the best precision at or beyond it."""
    _require_relevant(relevant)
    points: list[tuple[float, float]] = []  # (recall, precision) at every relevant hit
    hits = 0
    for i, d in enumerate(ranking, start=1):
        if d in relevant:
            hits += 1
            points.append((hits / len(relevant), hits / i))
    curve = []
    for level in levels:
        reachable = [p for r, p in points if r >= level - 1e-12]
        curve.append(max(reachable) if reachable else 0.0)
    return curve


def evaluate_ranking(ranking: Sequence[int], relevant: Set[int], *, depth: int) -> dict[str, float]:
    """All headline metrics for one query, from a ranking cut at ``depth``."""
    cut = ranking[:depth]
    return {
        "P@10": precision_at_k(cut, relevant, 10),
        "R@100": recall_at_k(cut, relevant, 100),
        "MAP": average_precision(cut, relevant),
        "nDCG@10": ndcg_at_k(cut, relevant, 10),
        "MRR": reciprocal_rank(cut, relevant),
        "Success@10": success_at_k(cut, relevant, 10),
    }
