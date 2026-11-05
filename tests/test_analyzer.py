import pytest

from maillens.text.analyzer import Analyzer


def test_lowercases_removes_stopwords_and_stems():
    assert Analyzer()("The Quick brown foxes were jumping") == ["quick", "brown", "fox", "jump"]


def test_numbers_and_dates_survive_as_separate_tokens():
    assert Analyzer()("hours 6 on 3/9/01") == ["hour", "6", "3", "9", "01"]


def test_options_can_switch_stemming_and_stopwords_off():
    plain = Analyzer(stem=False, remove_stopwords=False)
    assert plain("The foxes were jumping") == ["the", "foxes", "were", "jumping"]


def test_overlong_tokens_are_dropped():
    assert Analyzer(max_token_len=10)("short " + "x" * 11) == ["short"]


def test_empty_and_punctuation_only_text_gives_no_tokens():
    assert Analyzer()("") == []
    assert Analyzer()("--- !!! ???") == []


def test_inflected_forms_of_a_word_meet_at_the_same_stem():
    a = Analyzer()
    assert a("negotiating") == a("Negotiation") != []


def test_invalid_max_token_len_is_rejected():
    with pytest.raises(ValueError, match="max_token_len"):
        Analyzer(max_token_len=0)
