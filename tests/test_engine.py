from datetime import date

import numpy as np
import pandas as pd
import pytest

from maillens.engine import SYSTEMS, SearchEngine
from maillens.index.inverted import InvertedIndex
from maillens.pipeline import build_dense_chunks, index_text
from maillens.rankers.bm25 import Bm25Ranker
from maillens.rankers.dense import DenseRanker
from maillens.rankers.tfidf import TfidfRanker
from maillens.text.analyzer import Analyzer
from tests.helpers import HashingEncoder

ROWS = [
    # (subject, sender, day, thread, text)
    ("Gas storage report", "kay.mann@enron.com", 1, 0, "natural gas storage levels rose this week"),
    (
        "Re: Gas storage report",
        "jeff.dasovich@enron.com",
        2,
        0,
        "the storage injection numbers look high",
    ),
    (
        "California power market",
        "kay.mann@enron.com",
        3,
        1,
        "california power prices spiked after the outage",
    ),
    ("Lunch Friday", "tana.jones@enron.com", 4, 2, "lunch on friday at noon with the trading desk"),
    (
        "Contract draft",
        "jeff.dasovich@enron.com",
        5,
        3,
        "attached is the draft power purchase agreement",
    ),
    (
        "Re: Contract draft",
        "tana.jones@enron.com",
        6,
        3,
        "legal review complete the agreement is approved",
    ),
    ("Budget", "kay.mann@enron.com", 7, 4, "quarterly budget numbers attached for finance"),
    ("Re: Budget", "kay.mann@enron.com", 8, 4, "budget approved please proceed with the plan"),
]


@pytest.fixture(scope="module")
def engine() -> SearchEngine:
    emails = pd.DataFrame(
        {
            "doc_id": np.arange(len(ROWS), dtype=np.int32),
            "message_id": [f"<{i}@t>" for i in range(len(ROWS))],
            "subject": [r[0] for r in ROWS],
            "sender": [r[1] for r in ROWS],
            "recipients": [["x@enron.com"]] * len(ROWS),
            "date": pd.to_datetime([f"2001-03-{r[2]:02d} 10:00" for r in ROWS], utc=True),
            "mailbox": ["mann-k"] * 4 + ["jones-t"] * 4,
            "folder": ["inbox"] * len(ROWS),
            "thread_id": [r[3] for r in ROWS],
            "thread_size": [2, 2, 1, 1, 2, 2, 2, 2],
            "fresh": [r[4] for r in ROWS],
            "quoted": [""] * len(ROWS),
        }
    )
    emails["text"] = emails["fresh"]
    analyzer = Analyzer()
    index = InvertedIndex.build(
        analyzer(index_text(s, t)) for s, t in zip(emails["subject"], emails["text"], strict=True)
    )
    encoder = HashingEncoder()
    texts, chunk_doc = build_dense_chunks(emails)
    dense = DenseRanker(encoder.encode(texts), chunk_doc, len(emails), encoder)
    return SearchEngine(
        emails, analyzer, TfidfRanker(index, analyzer), Bm25Ranker(index, analyzer), dense, depth=6
    )


def test_compare_answers_with_all_four_rankers_and_reports_every_rank(engine):
    result = engine.compare("natural gas storage", 3)
    assert set(result.per_ranker) == set(SYSTEMS)
    assert all(1 <= len(hits) <= 3 for hits in result.per_ranker.values())
    assert result.per_ranker["bm25"][0].doc_id in (0, 1)  # the gas-storage thread
    top = result.per_ranker["hybrid"][0].doc_id
    assert result.ranks["hybrid"][top] == 1
    assert [h.doc_id for h in result.fused] == [h.doc_id for h in result.per_ranker["hybrid"]]
    # every shown document knows its rank in the lexical lists it appears in
    for system in ("tfidf", "bm25"):
        for hit in result.per_ranker[system]:
            assert result.ranks[system][hit.doc_id] == hit.rank


def test_the_hybrid_explains_its_score(engine):
    fused = engine.compare("power agreement", 3).fused[0]
    assert set(fused.ranks) <= {"tfidf", "bm25", "dense"}
    assert sum(fused.contributions.values()) == pytest.approx(fused.score)


def test_search_one_system_matches_the_comparison(engine):
    for system in SYSTEMS:
        assert [h.doc_id for h in engine.search("budget approved", system, 3)] == [
            h.doc_id for h in engine.compare("budget approved", 3).per_ranker[system]
        ]


@pytest.mark.parametrize(
    ("query", "k", "message"),
    [("   ", 3, "empty"), ("gas", 0, "between"), ("gas", 7, "between")],
)
def test_invalid_requests_are_rejected(engine, query, k, message):
    with pytest.raises(ValueError, match=message):
        engine.compare(query, k)


def test_unknown_system_is_rejected(engine):
    with pytest.raises(ValueError, match="unknown system"):
        engine.search("gas", "nope", 3)


def test_term_breakdown_exists_for_lexical_rankers_only(engine):
    parts = engine.explain_terms("natural gas storage", "bm25", 0)
    assert parts and parts[0].contribution >= parts[-1].contribution
    with pytest.raises(ValueError, match="tfidf"):
        engine.explain_terms("gas", "dense", 0)


def test_best_passage_returns_the_matched_chunk_text(engine):
    passage, similarity = engine.best_passage("lunch on friday", 3)
    assert "lunch" in passage and 0 < similarity <= 1.0 + 1e-6


def test_thread_lists_its_messages_oldest_first(engine):
    thread = engine.thread(3)
    assert list(thread["doc_id"]) == [4, 5]
    assert thread["date"].is_monotonic_increasing
    assert engine.thread(99).empty


def test_filters_restrict_every_ranker(engine):
    mask = engine.mask(sender_contains="tana.jones")
    result = engine.compare("agreement lunch", 5, allowed=mask)
    for hits in result.per_ranker.values():
        assert {h.doc_id for h in hits} <= {3, 5}
    assert engine.mask() is None
    assert list(engine.mask(date_from=date(2001, 3, 7))) == [False] * 6 + [True] * 2


def test_summary_describes_the_corpus(engine):
    summary = engine.summary()
    assert (summary.emails, summary.threads, summary.mailboxes) == (8, 5, 2)
    assert summary.chunks == 8 and summary.terms > 0
    assert summary.first < summary.last


def test_documents_must_be_addressed_by_position():
    emails = pd.DataFrame({"doc_id": [5, 6]})
    with pytest.raises(ValueError, match="row position"):
        SearchEngine(emails, Analyzer(), None, None, None)  # type: ignore[arg-type]


def test_loading_fails_fast_and_names_the_missing_file_without_touching_the_model(tmp_path):
    class ExplodingEncoder:
        @property
        def dim(self):
            raise AssertionError("the model must not be loaded when files are missing")

        def encode(self, texts, *, batch_size=64):
            raise AssertionError("the model must not be used when files are missing")

    with pytest.raises(FileNotFoundError) as caught:
        SearchEngine.load(
            processed_dir=tmp_path / "processed",
            artifacts_dir=tmp_path / "artifacts",
            encoder=ExplodingEncoder(),
        )
    assert "emails.parquet" in str(caught.value.filename)
