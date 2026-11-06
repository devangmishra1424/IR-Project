import pandas as pd

from maillens.app import ui
from maillens.rankers.base import Hit

WHEN = pd.Timestamp("2001-03-09 10:00", tz="UTC")


def card(**overrides):
    args = {
        "rank": 1,
        "subject": "Gas prices",
        "sender": "kay.mann@enron.com",
        "when": WHEN,
        "thread_size": 3,
        "snippet_html": "plain <mark>gas</mark>",
        "ranks": {"hybrid": 1, "bm25": 2, "dense": None, "tfidf": 7},
        "current": "bm25",
    }
    return ui.card_html(**{**args, **overrides})


def test_card_shows_subject_sender_date_and_thread_size():
    out = card()
    assert "Gas prices" in out and "kay.mann@enron.com" in out
    assert "9 Mar 2001" in out and "thread of 3" in out


def test_single_message_threads_are_not_advertised():
    assert "thread of" not in card(thread_size=1)


def test_untrusted_fields_are_escaped_but_the_prepared_snippet_is_kept():
    out = card(subject='<img src=x onerror="alert(1)">', sender="<script>x</script>")
    assert "<img" not in out and "<script>" not in out
    assert "&lt;img" in out and "&lt;script&gt;" in out
    assert "<mark>gas</mark>" in out  # snippet_html is trusted: already escaped upstream


def test_empty_subject_gets_a_placeholder():
    assert "(no subject)" in card(subject="   ")


def test_rank_chips_cover_every_ranker_and_mark_the_current_one():
    out = ui.rank_chips_html({"hybrid": 1, "bm25": 2, "dense": None, "tfidf": 7}, "bm25")
    for label in ("Hybrid RRF #1", "BM25 #2", "Dense –", "TF-IDF #7"):
        assert label in out
    assert out.count("ml-chip here") == 1


def test_column_header_uses_the_ranker_colour():
    out = ui.column_header_html("bm25")
    assert "#eb6834" in out and "BM25" in out


def test_collapse_threads_keeps_the_best_hit_and_counts_the_rest():
    hits = [Hit(10, 3.0, 1), Hit(11, 2.5, 2), Hit(20, 2.0, 3), Hit(12, 1.0, 4)]
    thread_of = {10: 1, 11: 1, 20: 2, 12: 1}.__getitem__
    collapsed = ui.collapse_threads(hits, thread_of)
    assert [(h.doc_id, hidden) for h, hidden in collapsed] == [(10, 2), (20, 0)]


def test_top_overlap_counts_shared_documents_per_pair():
    lists = {"hybrid": [1, 2, 3], "bm25": [1, 2, 9], "dense": [7, 8, 9], "tfidf": [1, 5, 6]}
    overlap = {(a, b): n for a, b, n in ui.top_overlap(lists, 3)}
    assert overlap[("hybrid", "bm25")] == 2
    assert overlap[("bm25", "dense")] == 1
    assert overlap[("hybrid", "dense")] == 0
    assert len(overlap) == 6


def test_top_overlap_only_looks_at_the_top_n():
    overlap = {(a, b): n for a, b, n in ui.top_overlap({"hybrid": [1, 2], "bm25": [2, 1]}, 1)}
    assert overlap == {("hybrid", "bm25"): 0}


def test_agreement_line_mentions_the_pairs():
    out = ui.agreement_html({"hybrid": [1, 2], "bm25": [1, 3]}, 2)
    assert "Hybrid RRF ∩ BM25" in out and "<b>1/2</b>" in out
