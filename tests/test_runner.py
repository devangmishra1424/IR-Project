import numpy as np
import pandas as pd
import pytest

from maillens.evaluation.queries import Query
from maillens.evaluation.runner import (
    fuse_runs,
    mean_with_ci,
    pr_curves,
    rank_queries,
    score_runs,
    significance_table,
    summarize,
)
from maillens.rankers.rrf import fused_order, reciprocal_rank_fusion


def q(qid, kind, relevant, role="test"):
    return Query(qid, f"text {qid}", kind, role, frozenset(relevant))


QUERIES = [q("a", "question", {2, 5}), q("b", "question", {1}), q("c", "subject", {7, 8}, "dev")]


# ---------------------------------------------------------------------------- fusion
def test_fused_order_matches_the_explainable_implementation_on_random_lists():
    rng = np.random.default_rng(0)
    for _ in range(200):
        lists = [list(rng.permutation(30)[: rng.integers(1, 30)]) for _ in range(3)]
        lists = [[int(d) for d in lst] for lst in lists]
        for k in (0, 1, 60):
            named = {f"r{i}": lst for i, lst in enumerate(lists)}
            expected = [h.doc_id for h in reciprocal_rank_fusion(named, k=k)]
            assert fused_order(lists, k=k) == expected


def test_fused_order_rejects_negative_k():
    with pytest.raises(ValueError, match="k must"):
        fused_order([[1, 2]], k=-1)


def test_fuse_runs_fuses_query_by_query_and_cuts_to_depth():
    runs = {"x": [[1, 2, 3], [4, 5]], "y": [[3, 2, 1], [5, 4]]}
    fused = fuse_runs(runs, ["x", "y"], k=60, depth=2)
    assert [f.tolist() for f in fused] == [[1, 3], [4, 5]]


@pytest.mark.parametrize(
    ("members", "message"),
    [([], "at least one"), (["x", "nope"], "unknown runs")],
)
def test_fuse_runs_validates_members(members, message):
    with pytest.raises(ValueError, match=message):
        fuse_runs({"x": [[1]]}, members)


def test_fuse_runs_requires_runs_over_the_same_queries():
    with pytest.raises(ValueError, match="same queries"):
        fuse_runs({"x": [[1]], "y": [[1], [2]]}, ["x", "y"])


def test_rank_queries_calls_the_retriever_with_each_query_text():
    seen = []

    def retrieve(text):
        seen.append(text)
        return [3, 1]

    run = rank_queries(QUERIES[:2], retrieve)
    assert [r.tolist() for r in run] == [[3, 1], [3, 1]]
    assert run[0].dtype == np.int32
    assert seen == ["text a", "text b"]


# ---------------------------------------------------------------------------- scoring
RUNS = {
    "good": [[2, 5, 9], [1, 9], [7, 8]],
    "bad": [[9, 8, 2], [9, 1], [9, 9 + 1]],
}


def test_score_runs_has_one_row_per_system_and_query_with_correct_values():
    df = score_runs(QUERIES, RUNS, depth=10)
    assert len(df) == 6
    good_a = df[(df["system"] == "good") & (df["qid"] == "a")].iloc[0]
    assert good_a["MAP"] == pytest.approx(1.0)
    assert good_a["first_relevant_rank"] == 1 and good_a["n_relevant"] == 2
    bad_a = df[(df["system"] == "bad") & (df["qid"] == "a")].iloc[0]
    assert bad_a["first_relevant_rank"] == 3
    assert bad_a["MRR"] == pytest.approx(1 / 3)
    never = df[(df["system"] == "bad") & (df["qid"] == "c")].iloc[0]
    assert never["first_relevant_rank"] == 0 and never["MAP"] == 0.0


def test_score_runs_rejects_a_run_with_the_wrong_number_of_rankings():
    with pytest.raises(ValueError, match="rankings for"):
        score_runs(QUERIES, {"x": [[1]]})


def test_summarize_averages_per_role_kind_and_system():
    s = summarize(score_runs(QUERIES, RUNS, depth=10))
    row = s[(s["system"] == "good") & (s["kind"] == "question")].iloc[0]
    assert row["queries"] == 2 and row["MAP"] == pytest.approx(1.0)
    assert set(s["role"]) == {"test", "dev"}


def test_pr_curves_average_the_interpolated_precision_per_group():
    curves = pr_curves(QUERIES[:2], {"good": RUNS["good"][:2]})
    assert len(curves) == 11
    assert curves["precision"].tolist() == pytest.approx([1.0] * 11)
    assert set(curves["kind"]) == {"question"} and set(curves["role"]) == {"test"}


# ---------------------------------------------------------------------------- significance
def test_significance_table_flags_a_clear_winner_and_not_a_tie():
    rng = np.random.default_rng(3)
    n = 120
    base = rng.uniform(0.2, 0.5, n)
    per_query = pd.concat(
        [
            pd.DataFrame(
                {
                    "system": name,
                    "qid": [f"q{i}" for i in range(n)],
                    "kind": "question",
                    "role": "test",
                    "MAP": vals,
                    "nDCG@10": vals,
                }
            )
            for name, vals in [
                ("strong", base + 0.15),
                ("weak", base),
                ("weak2", base + rng.normal(0, 0.001, n)),
            ]
        ]
    )
    table = significance_table(per_query, [("strong", "weak"), ("weak2", "weak")], n_resamples=1000)
    strong = table[(table["system_a"] == "strong") & (table["metric"] == "MAP")].iloc[0]
    assert strong["diff"] == pytest.approx(0.15, abs=1e-9)
    assert strong["p_bonferroni"] < 0.05 and strong["ci_low"] > 0
    tie = table[(table["system_a"] == "weak2") & (table["metric"] == "MAP")].iloc[0]
    assert tie["p_value"] > 0.05
    assert strong["p_bonferroni"] == pytest.approx(min(1.0, strong["p_value"] * 2))


def test_mean_with_ci_brackets_the_mean():
    mean, low, high = mean_with_ci([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert mean == pytest.approx(0.35) and low < mean < high
