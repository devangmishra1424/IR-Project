import numpy as np
import pandas as pd
import pytest

from maillens.prep.threads import assign_threads


def make(rows):
    """rows: (subject_norm, 'YYYY-MM-DD', sender, [recipients])"""
    return pd.DataFrame(
        {
            "subject_norm": [r[0] for r in rows],
            "date": pd.to_datetime([r[1] for r in rows], utc=True),
            "sender": [r[2] for r in rows],
            "recipients": [r[3] for r in rows],
        }
    )


def groups(ids):
    """Partition of row positions implied by thread ids (order-independent form)."""
    out: dict[int, list[int]] = {}
    for pos, tid in enumerate(ids):
        out.setdefault(int(tid), []).append(pos)
    return sorted(out.values())


def test_replies_with_shared_people_form_one_thread():
    df = make(
        [
            ("gas price outlook", "2001-03-01", "a", ["b"]),
            ("gas price outlook", "2001-03-02", "b", ["a"]),
            ("gas price outlook", "2001-03-04", "a", ["b", "c"]),
        ]
    )
    assert groups(assign_threads(df)) == [[0, 1, 2]]


def test_a_long_silence_starts_a_new_thread():
    df = make(
        [
            ("quarterly report", "2001-01-01", "a", ["b"]),
            ("quarterly report", "2001-06-01", "a", ["b"]),
        ]
    )
    assert groups(assign_threads(df, gap_days=30)) == [[0], [1]]


def test_the_gap_is_measured_from_the_latest_message_so_slow_threads_can_chain():
    df = make(
        [
            ("contract draft", "2001-01-01", "a", ["b"]),
            ("contract draft", "2001-01-25", "b", ["a"]),
            ("contract draft", "2001-02-20", "a", ["b"]),  # 26 days after the previous one
        ]
    )
    assert groups(assign_threads(df, gap_days=30)) == [[0, 1, 2]]


def test_same_subject_but_different_people_are_different_conversations():
    df = make(
        [
            ("budget review", "2001-03-01", "a", ["b"]),
            ("budget review", "2001-03-02", "x", ["y"]),
        ]
    )
    assert groups(assign_threads(df)) == [[0], [1]]


def test_generic_and_empty_subjects_never_thread():
    df = make(
        [
            ("meeting", "2001-03-01", "a", ["b"]),
            ("meeting", "2001-03-01", "a", ["b"]),
            ("", "2001-03-01", "a", ["b"]),
            ("", "2001-03-01", "a", ["b"]),
        ]
    )
    assert groups(assign_threads(df, min_subject_terms=2)) == [[0], [1], [2], [3]]


def test_a_thread_stops_growing_at_the_size_cap():
    rows = [("daily digest report", f"2001-03-{d:02d}", "a", ["b"]) for d in range(1, 8)]
    sizes = sorted(len(g) for g in groups(assign_threads(make(rows), max_thread_size=3)))
    assert sizes == [1, 3, 3]


def test_result_does_not_depend_on_input_order():
    rows = [
        ("gas price outlook", "2001-03-01", "a", ["b"]),
        ("gas price outlook", "2001-03-02", "b", ["a"]),
        ("budget review", "2001-03-01", "x", ["y"]),
        ("gas price outlook", "2001-09-01", "a", ["b"]),
    ]
    forward = groups(assign_threads(make(rows)))
    perm = [3, 0, 2, 1]
    shuffled = assign_threads(make([rows[i] for i in perm]))
    # map positions in the shuffled frame back to the original row numbers
    remapped = sorted(sorted(perm[p] for p in g) for g in groups(shuffled))
    assert remapped == forward


def test_every_row_gets_an_id_and_singletons_are_unique():
    df = make([("only one here", "2001-03-01", "a", ["b"])])
    ids = assign_threads(df)
    assert ids.dtype == np.int64 and ids.shape == (1,) and ids[0] >= 0


def test_missing_dates_are_rejected():
    df = make([("gas price outlook", "2001-03-01", "a", ["b"])])
    df.loc[0, "date"] = pd.NaT
    with pytest.raises(ValueError, match="date"):
        assign_threads(df)


def test_invalid_parameters_are_rejected():
    df = make([("gas price outlook", "2001-03-01", "a", ["b"])])
    with pytest.raises(ValueError):
        assign_threads(df, gap_days=0)
    with pytest.raises(ValueError):
        assign_threads(df, max_thread_size=0)
