import pytest

from maillens.snippets import best_window, highlight, query_terms
from maillens.text.analyzer import Analyzer

A = Analyzer()


def test_highlight_marks_every_inflection_of_a_query_word():
    terms = query_terms("negotiate contract", A)
    out = highlight("Negotiating contracts and contract talks", terms, A)
    assert out == (
        "<mark>Negotiating</mark> <mark>contracts</mark> and <mark>contract</mark> talks"
    )


def test_highlight_escapes_untrusted_html():
    terms = query_terms("gas", A)
    out = highlight('<script>alert("x")</script> gas & more', terms, A)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "<mark>gas</mark>" in out
    assert "&amp; more" in out


def test_highlight_without_matches_is_just_escaped_text():
    assert highlight("a < b", set(), A) == "a &lt; b"


def test_stopwords_in_the_query_are_not_highlighted():
    terms = query_terms("the gas", A)
    assert "<mark>the</mark>" not in highlight("the gas", terms, A)


def test_short_text_is_returned_whole_with_whitespace_collapsed():
    assert best_window("one   two\nthree", set(), A, width=300) == "one two three"


def test_window_centres_on_the_densest_cluster_of_matches():
    filler = "lorem ipsum dolor sit amet " * 40
    text = f"gas appears once here. {filler} california power market crisis {filler}"
    terms = query_terms("california power market crisis", A)
    window = best_window(text, terms, A, width=200)
    assert "california power market crisis" in window
    assert window.startswith("…") and window.endswith("…")
    assert len(window) <= 215


def test_window_falls_back_to_the_start_when_nothing_matches():
    text = "alpha " * 200
    window = best_window(text, {"zzz"}, A, width=100)
    assert window.startswith("alpha alpha")
    assert window.endswith("…")


def test_window_rejects_tiny_widths():
    with pytest.raises(ValueError, match="width"):
        best_window("text", set(), A, width=5)
