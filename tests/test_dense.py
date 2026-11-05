import numpy as np
import pytest

from maillens.rankers.dense import (
    DenseRanker,
    chunk_inputs,
    embed_chunks,
    split_into_chunks,
)
from tests.helpers import HashingEncoder


# ---------------------------------------------------------------------------- chunking
def test_short_text_is_a_single_chunk():
    assert split_into_chunks("one two three", words=10, stride=5, max_chunks=4) == ["one two three"]


def test_empty_text_still_yields_one_chunk():
    assert split_into_chunks("   ", words=10, stride=5, max_chunks=4) == [""]


def test_windows_overlap_and_cover_the_whole_text():
    text = " ".join(str(i) for i in range(25))
    chunks = split_into_chunks(text, words=10, stride=8, max_chunks=10)
    assert chunks[0].split() == [str(i) for i in range(10)]
    assert chunks[1].split()[0] == "8"  # overlap of two words
    assert chunks[-1].split()[-1] == "24"  # nothing dropped at the end


def test_chunk_count_is_capped():
    text = " ".join(["w"] * 1000)
    assert len(split_into_chunks(text, words=10, stride=10, max_chunks=3)) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"words": 0, "stride": 1, "max_chunks": 1},
        {"words": 5, "stride": 6, "max_chunks": 1},
        {"words": 5, "stride": 5, "max_chunks": 0},
    ],
)
def test_bad_chunking_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        split_into_chunks("a b c", **kwargs)


def test_chunk_inputs_prefix_every_chunk_with_the_subject():
    out = chunk_inputs("Gas forecast", "alpha beta")
    assert out == ["Gas forecast. alpha beta"]
    assert chunk_inputs("", "alpha") == ["alpha"]


# ---------------------------------------------------------------------------- ranker
@pytest.fixture
def ranker() -> DenseRanker:
    enc = HashingEncoder()
    # doc 0 has two chunks (the second one is about solar), doc 1 one chunk, doc 2 one chunk
    texts = ["gas pipeline outage", "solar panel subsidy", "wind farm contract", "board lunch menu"]
    vectors = enc.encode(texts)
    chunk_doc = np.array([0, 0, 1, 2])
    return DenseRanker(vectors, chunk_doc, n_docs=3, encoder=enc)


def test_document_score_is_its_best_chunk(ranker):
    hits = ranker.search("solar panel", 3)
    assert hits[0].doc_id == 0  # found through its *second* chunk
    assert [h.rank for h in hits] == [1, 2, 3]


def test_best_chunk_reports_which_chunk_matched(ranker):
    chunk, similarity = ranker.best_chunk("solar panel", doc_id=0)
    assert chunk == 1
    assert 0 < similarity <= 1.0 + 1e-6


def test_dense_search_respects_the_allowed_mask(ranker):
    allowed = np.array([False, True, True])
    assert {h.doc_id for h in ranker.search("solar panel", 3, allowed=allowed)} == {1, 2}


def test_dense_always_ranks_every_allowed_document(ranker):
    assert len(ranker.search("anything at all", 10)) == 3


def test_constructor_rejects_inconsistent_inputs():
    enc = HashingEncoder()
    vectors = enc.encode(["a", "b"])
    with pytest.raises(ValueError, match="one vector per chunk"):
        DenseRanker(vectors, np.array([0]), n_docs=1, encoder=enc)
    with pytest.raises(ValueError, match="sorted"):
        DenseRanker(vectors, np.array([1, 0]), n_docs=2, encoder=enc)
    with pytest.raises(ValueError, match="at least one chunk"):
        DenseRanker(vectors, np.array([0, 0]), n_docs=2, encoder=enc)


# ---------------------------------------------------------------------------- embedding shards
class CountingEncoder(HashingEncoder):
    def __init__(self) -> None:
        super().__init__(dim=32)
        self.calls = 0

    def encode(self, texts, *, batch_size=64):
        self.calls += 1
        return super().encode(texts, batch_size=batch_size)


def test_embedding_resumes_from_saved_shards_without_recomputing(tmp_path):
    texts = [f"text number {i}" for i in range(10)]
    enc = CountingEncoder()
    first = embed_chunks(texts, enc, tmp_path, model_name="stub", shard_size=4)
    assert enc.calls == 3  # shards of 4 + 4 + 2
    again = embed_chunks(texts, enc, tmp_path, model_name="stub", shard_size=4)
    assert enc.calls == 3  # nothing recomputed
    assert first.shape == (10, 32) and first.dtype == np.float32
    assert np.array_equal(first, again)


def test_embedding_refuses_to_reuse_a_folder_for_different_input(tmp_path):
    enc = CountingEncoder()
    embed_chunks(["a", "b", "c"], enc, tmp_path, model_name="stub", shard_size=2)
    with pytest.raises(ValueError, match="different input"):
        embed_chunks(["a", "b", "DIFFERENT"], enc, tmp_path, model_name="stub", shard_size=2)


def test_embedding_detects_a_corrupt_shard(tmp_path):
    enc = CountingEncoder()
    embed_chunks(["a", "b", "c"], enc, tmp_path, model_name="stub", shard_size=2)
    np.save(tmp_path / "shard_00000.npy", np.zeros((1, 32), dtype=np.float16))
    with pytest.raises(ValueError, match="shape"):
        embed_chunks(["a", "b", "c"], enc, tmp_path, model_name="stub", shard_size=2)
