import numpy as np
import pytest

from maillens.index.inverted import InvertedIndex

DOCS = [
    ["a", "a", "b"],  # doc 0
    ["b", "c"],  # doc 1
    ["c", "c", "c", "d"],  # doc 2
    [],  # doc 3: analysed to nothing (e.g. only stop-words)
]


@pytest.fixture
def index() -> InvertedIndex:
    return InvertedIndex.build(DOCS)


def test_counts(index):
    assert index.n_docs == 4
    assert index.n_terms == 4
    assert index.n_postings == 6
    assert list(index.doc_len) == [3, 2, 4, 0]
    assert index.avgdl == pytest.approx(9 / 4)


def test_postings_are_sorted_by_document_and_carry_term_frequencies(index):
    docs, tfs = index.postings(index.term_id("c"))
    assert list(docs) == [1, 2]
    assert list(tfs) == [1, 3]
    docs, tfs = index.postings(index.term_id("a"))
    assert (list(docs), list(tfs)) == ([0], [2])


def test_document_frequency(index):
    assert {t: int(index.df[i]) for t, i in index.vocab.items()} == {"a": 1, "b": 2, "c": 2, "d": 1}


def test_unknown_term_has_no_id(index):
    assert index.term_id("zzz") is None


def test_posting_position_finds_or_misses(index):
    tid = index.term_id("b")
    pos = index.posting_position(tid, 1)
    assert pos is not None and index.tfs[pos] == 1
    assert index.posting_position(tid, 2) is None  # "b" is not in doc 2
    assert index.posting_position(tid, 3) is None  # doc 3 is empty


def test_term_of_posting_aligns_with_postings(index):
    terms = index.term_of_posting()
    assert terms.shape == index.doc_ids.shape
    for tid in range(index.n_terms):
        sl = index.posting_slice(tid)
        assert (terms[sl] == tid).all()


def test_save_and_load_round_trip(tmp_path, index):
    index.save(tmp_path / "idx")
    loaded = InvertedIndex.load(tmp_path / "idx")
    assert loaded.terms == index.terms
    for name in ("indptr", "doc_ids", "tfs", "doc_len"):
        assert np.array_equal(getattr(loaded, name), getattr(index, name))


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError, match="empty corpus"):
        InvertedIndex.build([])


def test_inconsistent_arrays_are_rejected():
    with pytest.raises(ValueError, match="indptr"):
        InvertedIndex(
            terms=["a"],
            indptr=np.array([0, 1, 2]),
            doc_ids=np.array([0], dtype=np.int32),
            tfs=np.array([1], dtype=np.int32),
            doc_len=np.array([1], dtype=np.int32),
        )
