import pandas as pd
import pytest

from maillens.evaluation.queries import (
    Query,
    eligible_threads,
    keyword_query,
    load_queries,
    question_queries,
    save_queries,
    thread_queries,
)
from maillens.text.analyzer import Analyzer

A = Analyzer()
BANK = [
    "pipeline",
    "turbine",
    "refinery",
    "tariff",
    "settlement",
    "derivative",
    "hedging",
    "swap",
    "futures",
    "options",
    "forward",
    "basis",
    "spread",
    "volatility",
    "liquidity",
    "margin",
    "exposure",
    "collateral",
    "clearing",
    "delivery",
    "scheduling",
    "nomination",
    "imbalance",
    "capacity",
]


def thread_rows(tid, subject, n, start_doc, *, span_days=10, words=None):
    words = words or BANK[4 * tid : 4 * tid + 4]
    rows = []
    for i in range(n):
        rows.append(
            {
                "doc_id": start_doc + i,
                "message_id": f"<{start_doc + i}@t>",
                "subject_norm": subject,
                "thread_id": tid,
                "date": pd.Timestamp("2001-03-01", tz="UTC")
                + pd.Timedelta(days=i * span_days // n),
                "fresh": f"{' '.join(words)} extra{i}",
            }
        )
    return rows


@pytest.fixture
def emails() -> pd.DataFrame:
    rows = []
    doc = 0
    # six clean threads (ids 0-5), 5 messages each, with distinct subjects
    for tid in range(6):
        # (a number, not a bank word: the subject must not share words with the thread's text)
        rows += thread_rows(tid, f"gas storage report number {tid}", 5, doc)
        doc += 5
    rows += thread_rows(6, "tiny thread subject here", 3, doc)  # too small
    doc += 3
    rows += thread_rows(7, "slow moving thread subject", 5, doc, span_days=400)  # too long
    doc += 5
    rows += thread_rows(8, "shared subject across threads", 5, doc)  # subject reused
    doc += 5
    rows += thread_rows(9, "shared subject across threads", 5, doc)
    doc += 5
    rows += thread_rows(10, "short subject", 5, doc)  # < 3 words
    doc += 5
    return pd.DataFrame(rows)


def test_eligible_threads_applies_every_filter(emails):
    assert list(eligible_threads(emails)) == [0, 1, 2, 3, 4, 5]


def test_keyword_query_picks_shared_non_subject_words_in_their_written_form(emails):
    thread = emails[emails["thread_id"] == 0]
    query = keyword_query(thread, A, lambda stem: 5, n_docs=1000)
    assert sorted(query.split()) == sorted(BANK[0:4])  # the four words every message has
    assert "extra0" not in query  # appears in one message only


def test_keyword_query_never_uses_subject_words():
    thread = pd.DataFrame(
        {
            "subject_norm": ["pipeline outage report"] * 3,
            "fresh": ["pipeline turbine refinery tariff settlement"] * 3,
        }
    )
    query = keyword_query(thread, A, lambda stem: 5, n_docs=1000)
    assert "pipeline" not in query.split()


def test_keyword_query_is_none_when_the_thread_has_too_few_shared_words():
    thread = pd.DataFrame(
        {"subject_norm": ["a b c"] * 2, "fresh": ["turbine refinery", "turbine basis"]}
    )
    assert keyword_query(thread, A, lambda stem: 5, n_docs=1000) is None


def test_rarer_words_rank_above_common_ones():
    thread = pd.DataFrame({"subject_norm": ["x y z"] * 2, "fresh": ["common rarer words here"] * 2})
    freq = {"common": 900, "rarer": 3}
    query = keyword_query(thread, A, lambda stem: freq.get(stem, 50), n_docs=1000, n_terms=2)
    assert query.split()[0] == "rarer"


def test_question_queries_map_message_ids_to_document_ids():
    emails = pd.DataFrame({"message_id": ["<a>", "<b>", "<c>"], "doc_id": [0, 1, 2]})
    answers = pd.DataFrame(
        {"qid": ["test-1"], "question": ["who?"], "role": ["test"], "answers": [["<a>", "<c>"]]}
    )
    (query,) = question_queries(emails, answers)
    assert query == Query("test-1", "who?", "question", "test", frozenset({0, 2}))
    answers["answers"] = [["<zzz>"]]
    with pytest.raises(KeyError):
        question_queries(emails, answers)


def test_thread_queries_pair_subject_and_keyword_on_the_same_threads(emails):
    queries = thread_queries(emails, A, lambda stem: 5, dev_threads=2, test_threads=3, seed=1)
    assert len(queries) == 10
    by_thread: dict[str, list[Query]] = {}
    for query in queries:
        by_thread.setdefault(query.qid.rsplit("-", 1)[1], []).append(query)
    assert len(by_thread) == 5
    for pair in by_thread.values():
        assert {p.kind for p in pair} == {"subject", "keyword"}
        assert pair[0].relevant == pair[1].relevant and len(pair[0].relevant) == 5
        assert pair[0].role == pair[1].role
    assert sum(1 for q in queries if q.role == "dev") == 4
    assert sum(1 for q in queries if q.role == "test") == 6
    # subject queries are the thread subject, so every e-mail of the thread contains them
    subject_query = next(q for q in queries if q.kind == "subject")
    assert subject_query.text.startswith("gas storage report number")


def test_thread_queries_are_deterministic_and_fail_loudly_when_threads_run_out(emails):
    first = thread_queries(emails, A, lambda stem: 5, dev_threads=2, test_threads=3, seed=9)
    again = thread_queries(emails, A, lambda stem: 5, dev_threads=2, test_threads=3, seed=9)
    assert first == again
    with pytest.raises(ValueError, match="usable threads"):
        thread_queries(emails, A, lambda stem: 5, dev_threads=4, test_threads=4)


def test_queries_survive_a_save_load_round_trip(tmp_path):
    queries = [Query("q1", "text one", "question", "dev", frozenset({3, 9}))]
    save_queries(queries, tmp_path / "sub" / "queries.parquet")
    assert load_queries(tmp_path / "sub" / "queries.parquet") == queries
