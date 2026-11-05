import math

import pytest

from maillens.evaluation.metrics import (
    RECALL_LEVELS,
    average_precision,
    evaluate_ranking,
    interpolated_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    success_at_k,
)

# Worked example: ranking 1..5, relevant documents are 2 and 5.
RANKING = [1, 2, 3, 4, 5]
RELEVANT = {2, 5}


def test_precision_and_recall():
    assert precision_at_k(RANKING, RELEVANT, 5) == pytest.approx(0.4)
    assert precision_at_k(RANKING, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKING, RELEVANT, 5) == 1.0
    assert recall_at_k(RANKING, RELEVANT, 2) == 0.5


def test_precision_divides_by_k_even_when_fewer_results_returned():
    assert precision_at_k([7], {7}, 10) == pytest.approx(0.1)


def test_average_precision_counts_unretrieved_relevant_as_zero():
    # (1/2 + 2/5) / 2 relevant documents
    assert average_precision(RANKING, RELEVANT) == pytest.approx(0.45)
    # one relevant found at rank 1, the other never retrieved
    assert average_precision([9, 8], {9, 100}) == pytest.approx(0.5)


def test_ndcg_matches_hand_calculation():
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKING, RELEVANT, 5) == pytest.approx(dcg / ideal)


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k([2, 5, 1], RELEVANT, 3) == pytest.approx(1.0)


def test_reciprocal_rank_and_success():
    assert reciprocal_rank(RANKING, RELEVANT) == 0.5
    assert reciprocal_rank([1, 3], RELEVANT) == 0.0
    assert success_at_k(RANKING, RELEVANT, 2) == 1.0
    assert success_at_k(RANKING, RELEVANT, 1) == 0.0


def test_depth_truncates_the_ranking():
    assert reciprocal_rank(RANKING, RELEVANT, depth=1) == 0.0
    assert average_precision(RANKING, RELEVANT, depth=2) == pytest.approx(0.25)


def test_interpolated_precision_curve():
    curve = interpolated_precision(RANKING, RELEVANT)
    # points: (recall 0.5, precision 0.5) and (recall 1.0, precision 0.4)
    assert len(curve) == len(RECALL_LEVELS) == 11
    assert curve[:6] == [0.5] * 6  # recall levels 0.0 .. 0.5
    assert curve[6:] == [0.4] * 5  # recall levels 0.6 .. 1.0


def test_interpolated_precision_is_zero_when_recall_level_is_never_reached():
    curve = interpolated_precision([1, 2, 3], {3, 99})
    assert curve[:6] == [pytest.approx(1 / 3)] * 6
    assert curve[6:] == [0.0] * 5


def test_evaluate_ranking_bundles_all_metrics():
    out = evaluate_ranking(RANKING, RELEVANT, depth=1000)
    assert set(out) == {"P@10", "R@100", "MAP", "nDCG@10", "MRR", "Success@10"}
    assert out["MAP"] == pytest.approx(0.45)
    assert out["MRR"] == 0.5


@pytest.mark.parametrize(
    "call",
    [
        lambda: recall_at_k([1], set(), 10),
        lambda: average_precision([1], set()),
        lambda: ndcg_at_k([1], set(), 10),
        lambda: reciprocal_rank([1], set()),
        lambda: interpolated_precision([1], set()),
    ],
)
def test_metrics_reject_queries_without_relevant_documents(call):
    with pytest.raises(ValueError, match="relevant"):
        call()


@pytest.mark.parametrize("k", [0, -3])
def test_metrics_reject_non_positive_k(k):
    with pytest.raises(ValueError, match="positive"):
        precision_at_k([1], {1}, k)
    with pytest.raises(ValueError, match="positive"):
        ndcg_at_k([1], {1}, k)
