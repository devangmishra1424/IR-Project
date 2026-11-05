import numpy as np
import pytest

from maillens.rankers.base import Hit
from maillens.rankers.rrf import HybridRanker, reciprocal_rank_fusion


def test_fusion_matches_hand_calculation():
    fused = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [3, 1, 4]}, k=60)
    assert [h.doc_id for h in fused] == [1, 3, 2, 4]
    by_doc = {h.doc_id: h for h in fused}
    assert by_doc[1].score == pytest.approx(1 / 61 + 1 / 62)
    assert by_doc[3].score == pytest.approx(1 / 63 + 1 / 61)
    assert by_doc[2].score == pytest.approx(1 / 62)
    assert by_doc[4].score == pytest.approx(1 / 63)


def test_fused_hit_explains_where_the_score_came_from():
    fused = reciprocal_rank_fusion({"a": [1, 2], "b": [2]}, k=60)
    top = fused[0]
    assert top.doc_id == 2
    assert dict(top.ranks) == {"a": 2, "b": 1}
    assert sum(top.contributions.values()) == pytest.approx(top.score)


def test_a_document_missing_from_a_list_earns_nothing_from_it():
    fused = reciprocal_rank_fusion({"a": [1], "b": [2]}, k=60)
    assert all(len(h.ranks) == 1 for h in fused)


def test_ties_break_by_best_rank_then_document_id():
    # docs 5 and 9 tie exactly; 5 wins on document id
    fused = reciprocal_rank_fusion({"a": [5], "b": [9]}, k=10)
    assert [h.doc_id for h in fused] == [5, 9]


def test_weights_shift_the_ordering():
    rankings = {"a": [1, 2], "b": [2, 1]}
    assert reciprocal_rank_fusion(rankings)[0].doc_id == 1  # tie -> lower id
    assert reciprocal_rank_fusion(rankings, weights={"b": 3.0})[0].doc_id == 2


def test_smaller_k_rewards_top_ranks_more_sharply():
    steep = reciprocal_rank_fusion({"a": [1, 2]}, k=1)
    flat = reciprocal_rank_fusion({"a": [1, 2]}, k=1000)
    assert steep[0].score / steep[1].score > flat[0].score / flat[1].score


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rankings": {"a": [1, 1]}}, "more than once"),
        ({"rankings": {"a": [1]}, "k": -1}, "k must"),
        ({"rankings": {"a": [1]}, "weights": {"zzz": 1.0}}, "unknown rankers"),
        ({"rankings": {"a": [1]}, "weights": {"a": 0.0}}, "positive"),
    ],
)
def test_invalid_input_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        reciprocal_rank_fusion(**kwargs)


def test_empty_rankings_fuse_to_nothing():
    assert reciprocal_rank_fusion({"a": [], "b": []}) == []


class StubRanker:
    def __init__(self, name: str, order: list[int]) -> None:
        self.name = name
        self._order = order

    def search(self, query: str, k: int, *, allowed: np.ndarray | None = None) -> list[Hit]:
        docs = [d for d in self._order if allowed is None or allowed[d]][:k]
        return [Hit(d, 1.0 / r, r) for r, d in enumerate(docs, start=1)]


def test_hybrid_ranker_fuses_member_lists_and_respects_the_filter():
    hybrid = HybridRanker([StubRanker("x", [0, 1, 2]), StubRanker("y", [2, 1, 0])], depth=3)
    # Docs 0 and 2 are first in one list and last in the other (1/61 + 1/63 each), which beats
    # doc 1's two middle places (2/62): RRF rewards a top rank more than it punishes a low one.
    assert [h.doc_id for h in hybrid.search("q", 3)] == [0, 2, 1]

    allowed = np.array([True, False, True])
    assert [h.doc_id for h in hybrid.search("q", 3, allowed=allowed)] == [0, 2]


def test_hybrid_ranker_needs_members_and_a_positive_depth():
    with pytest.raises(ValueError, match="at least one"):
        HybridRanker([])
    with pytest.raises(ValueError, match="depth"):
        HybridRanker([StubRanker("x", [0])], depth=0)
