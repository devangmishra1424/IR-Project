import math

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from maillens.index.inverted import InvertedIndex
from maillens.rankers.base import top_k_indices
from maillens.rankers.bm25 import Bm25Ranker
from maillens.rankers.tfidf import TfidfRanker
from maillens.text.analyzer import Analyzer

CORPUS = [
    "gas price forecast for california power market",
    "meeting moved to friday please confirm attendance",
    "california energy crisis and the power market response",
    "natural gas storage report for the week",
    "lunch on friday with the trading desk",
    "power purchase agreement draft attached for review",
    "review of the natural gas trading positions",
    "friday meeting agenda gas power california",
]
QUERIES = [
    "california power",
    "natural gas trading",
    "friday meeting",
    "unknownterm",
    "gas gas gas",
]


@pytest.fixture(scope="module")
def analyzer() -> Analyzer:
    return Analyzer()


@pytest.fixture(scope="module")
def index(analyzer) -> InvertedIndex:
    return InvertedIndex.build(analyzer(text) for text in CORPUS)


# ---------------------------------------------------------------------------- top-k helper
def test_top_k_breaks_ties_by_index_and_orders_best_first():
    scores = np.array([0.5, 0.9, 0.5, 0.9, 0.1])
    assert list(top_k_indices(scores, 3)) == [1, 3, 0]


def test_top_k_tie_at_the_cutoff_is_deterministic():
    scores = np.array([0.7, 0.7, 0.7, 0.7])
    assert list(top_k_indices(scores, 2)) == [0, 1]


def test_top_k_skips_masked_and_optionally_zero_scores():
    scores = np.array([0.0, -np.inf, 0.4])
    assert list(top_k_indices(scores, 5)) == [2, 0]
    assert list(top_k_indices(scores, 5, positive_only=True)) == [2]


def test_top_k_rejects_non_positive_k():
    with pytest.raises(ValueError, match="positive"):
        top_k_indices(np.array([1.0]), 0)


# ---------------------------------------------------------------------------- TF-IDF
@pytest.mark.parametrize("query", QUERIES)
def test_tfidf_matches_scikit_learn_cosine(analyzer, index, query):
    vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True)
    doc_matrix = vec.fit_transform(CORPUS)
    expected = (doc_matrix @ vec.transform([query]).T).toarray().ravel()
    ours = TfidfRanker(index, analyzer).score_all(query)
    assert np.allclose(ours, expected, atol=1e-5)


def test_tfidf_scores_are_cosines_between_zero_and_one(analyzer, index):
    scores = TfidfRanker(index, analyzer).score_all("california power market")
    assert scores.min() >= 0 and scores.max() <= 1 + 1e-6


# ---------------------------------------------------------------------------- BM25
def naive_bm25(docs: list[list[str]], query: list[str], k1: float, b: float) -> list[float]:
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    out = []
    for d in docs:
        score = 0.0
        for term in set(query):
            df = sum(1 for other in docs if term in other)
            tf = d.count(term)
            if df == 0 or tf == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
        out.append(score)
    return out


@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize(("k1", "b"), [(1.2, 0.75), (2.0, 0.3), (0.0, 0.75), (1.5, 1.0)])
def test_bm25_matches_a_naive_textbook_implementation(analyzer, index, query, k1, b):
    docs = [analyzer(t) for t in CORPUS]
    expected = naive_bm25(docs, analyzer(query), k1, b)
    ours = Bm25Ranker(index, analyzer, k1=k1, b=b).score_all(query)
    assert np.allclose(ours, expected, atol=1e-5)


def test_bm25_hand_computed_value():
    plain = Analyzer(stem=False, remove_stopwords=False)
    idx = InvertedIndex.build(plain(t) for t in ["a a b", "b c", "c c c d"])
    # N=3, df(a)=1 -> idf = ln(1 + 2.5/1.5); avgdl = 3; doc 0: tf=2, |d|=3 -> norm = 1.2
    expected = math.log(1 + 2.5 / 1.5) * 2 * 2.2 / (2 + 1.2)
    score = Bm25Ranker(idx, plain, k1=1.2, b=0.75).score_all("a")[0]
    assert score == pytest.approx(expected, rel=1e-5)
    assert score == pytest.approx(1.3486, abs=1e-3)


def test_bm25_ignores_repeated_query_words(analyzer, index):
    ranker = Bm25Ranker(index, analyzer)
    assert np.array_equal(ranker.score_all("gas"), ranker.score_all("gas gas gas"))


def test_bm25_parameter_validation(analyzer, index):
    with pytest.raises(ValueError, match="k1"):
        Bm25Ranker(index, analyzer, k1=-0.1)
    with pytest.raises(ValueError, match="b must"):
        Bm25Ranker(index, analyzer, b=1.5)


# ---------------------------------------------------------------------------- shared behaviour
@pytest.mark.parametrize("ranker_cls", [TfidfRanker, Bm25Ranker])
def test_unknown_or_empty_queries_return_nothing(analyzer, index, ranker_cls):
    ranker = ranker_cls(index, analyzer)
    assert ranker.search("unknownterm", 10) == []
    assert ranker.search("", 10) == []
    assert ranker.search("the and of", 10) == []  # only stop-words


@pytest.mark.parametrize("ranker_cls", [TfidfRanker, Bm25Ranker])
def test_search_returns_ranked_hits_and_only_matching_documents(analyzer, index, ranker_cls):
    hits = ranker_cls(index, analyzer).search("california power", 10)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
    assert {h.doc_id for h in hits} == {0, 2, 5, 7}  # the four documents mentioning either term
    assert hits[0].doc_id in {0, 2, 7}  # these three mention both terms ...
    assert hits[-1].doc_id == 5  # ... and doc 5 mentions only "power"


@pytest.mark.parametrize("ranker_cls", [TfidfRanker, Bm25Ranker])
def test_allowed_mask_filters_results(analyzer, index, ranker_cls):
    allowed = np.ones(len(CORPUS), dtype=bool)
    allowed[[0, 2]] = False
    hits = ranker_cls(index, analyzer).search("california power", 10, allowed=allowed)
    assert {h.doc_id for h in hits} == {5, 7}


@pytest.mark.parametrize("ranker_cls", [TfidfRanker, Bm25Ranker])
def test_explanation_sums_to_the_score(analyzer, index, ranker_cls):
    ranker = ranker_cls(index, analyzer)
    top = ranker.search("california power market", 1)[0]
    parts = ranker.explain("california power market", top.doc_id)
    assert sum(p.contribution for p in parts) == pytest.approx(top.score, rel=1e-5)
    assert parts == sorted(parts, key=lambda p: -p.contribution)
    assert all(p.tf >= 1 and p.df >= 1 for p in parts)


def test_tfidf_hand_computed_value():
    plain = Analyzer(stem=False, remove_stopwords=False)
    idx = InvertedIndex.build(plain(t) for t in ["a a b", "b c", "c c c d"])
    # idf(a) = ln(4/2) + 1 = 1.6931, idf(b) = ln(4/3) + 1 = 1.2877
    # doc 0 raw weights: a = (1 + ln 2) * 1.6931 = 2.8666, b = 1.2877 -> length 3.1425
    expected = (
        (1 + math.log(2))
        * (math.log(2) + 1)
        / math.sqrt(((1 + math.log(2)) * (math.log(2) + 1)) ** 2 + (math.log(4 / 3) + 1) ** 2)
    )
    score = TfidfRanker(idx, plain).score_all("a")[0]
    assert score == pytest.approx(expected, rel=1e-5)
    assert score == pytest.approx(0.9122, abs=1e-3)
