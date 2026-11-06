import pandas as pd
import pytest

from maillens.evaluation.analysis import LENGTH_LABELS, length_analysis
from maillens.evaluation.queries import Query


def test_mrr_is_grouped_by_the_length_of_the_answer_email():
    emails = pd.DataFrame({"text": ["w " * 10, "w " * 100, "w " * 300, "w " * 900]})
    queries = [
        Query("t1", "q", "question", "test", frozenset({0})),
        Query("t2", "q", "question", "test", frozenset({1})),
        Query("t3", "q", "question", "test", frozenset({2})),
        Query("t4", "q", "question", "test", frozenset({3, 0})),  # mean length 455 words
        Query("d1", "q", "question", "dev", frozenset({0})),  # dev queries are ignored
        Query("s1", "q", "subject", "test", frozenset({0})),  # so are other query kinds
    ]
    rows = []
    for system, mrr in (("tfidf", 0.25), ("bm25", 1.0)):
        for query in queries:
            rows.append(
                {
                    "role": query.role,
                    "kind": query.kind,
                    "qid": query.qid,
                    "system": system,
                    "MRR": mrr if query.qid != "t1" else 0.5,
                }
            )
    table = length_analysis(pd.DataFrame(rows), queries, emails)

    assert set(table["answer_length"]) == set(LENGTH_LABELS)
    cell = table.set_index(["answer_length", "system"])
    assert cell.loc[("up to 50 words", "tfidf"), "MRR"] == pytest.approx(0.5)
    assert cell.loc[("up to 50 words", "bm25"), "questions"] == 1
    assert cell.loc[("over 400 words", "bm25"), "MRR"] == pytest.approx(1.0)
    assert table["questions"].sum() == 2 * 4  # 4 test questions, 2 systems
