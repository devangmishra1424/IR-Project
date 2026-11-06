from datetime import date

import pandas as pd
import pytest

from maillens.filters import build_mask


@pytest.fixture
def docs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sender": [
                "jeff.skilling@enron.com",
                "kay.mann@enron.com",
                None,
                "JEFF.dasovich@enron.com",
            ],
            "mailbox": ["skilling-j", "mann-k", "mann-k", "dasovich-j"],
            "date": pd.to_datetime(
                ["2001-01-10 00:00", "2001-03-05 23:30", "2001-03-06 00:00", "2002-01-01 00:00"],
                utc=True,
            ),
        }
    )


def test_no_filters_means_no_mask(docs):
    assert build_mask(docs) is None
    assert build_mask(docs, sender_contains="   ") is None


def test_sender_filter_is_case_insensitive_and_tolerates_missing_values(docs):
    assert list(build_mask(docs, sender_contains="JEFF")) == [True, False, False, True]


def test_special_characters_in_the_sender_filter_are_not_a_regex(docs):
    assert not build_mask(docs, sender_contains="(.*").any()


def test_mailbox_filter_is_exact(docs):
    assert list(build_mask(docs, mailbox="mann-k")) == [False, True, True, False]


def test_date_bounds_are_inclusive_of_the_whole_end_day(docs):
    mask = build_mask(docs, date_from=date(2001, 3, 5), date_to=date(2001, 3, 5))
    assert list(mask) == [False, True, False, False]


def test_filters_combine_with_and(docs):
    mask = build_mask(docs, sender_contains="jeff", date_from=date(2002, 1, 1))
    assert list(mask) == [False, False, False, True]
