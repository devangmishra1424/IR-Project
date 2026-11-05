"""Thread reconstruction.

The Enron export carries no In-Reply-To / References headers, so a conversation is rebuilt from
what is left: the normalised subject, the people involved and time. A message joins an existing
thread when

* it has the same normalised subject,
* it was sent within ``gap_days`` of that thread's latest message, and
* it shares at least one participant (sender or recipient) with the thread so far.

Three guards keep the heuristic honest: subjects that are too generic to identify a
conversation ("hi", "meeting", empty) never thread; a thread stops growing at
``max_thread_size`` (automated daily digests would otherwise chain for years); and the result
does not depend on the order of the input rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

_TERM = re.compile(r"[a-z0-9]+")
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
_SECONDS_PER_DAY = 86_400.0


@dataclass
class _Thread:
    thread_id: int
    last_sent: float
    people: set[str]
    size: int


def _is_too_generic(subject_norm: str, min_terms: int) -> bool:
    return len(_TERM.findall(subject_norm)) < min_terms


def assign_threads(
    meta: pd.DataFrame,
    *,
    gap_days: float = 30.0,
    min_subject_terms: int = 2,
    max_thread_size: int = 500,
) -> np.ndarray:
    """Thread id (int64) for every row of ``meta``.

    ``meta`` needs the columns ``subject_norm`` (str), ``date`` (UTC timestamps, no missing
    values), ``sender`` (str) and ``recipients`` (list of str). Ids are unique per thread but
    otherwise arbitrary.
    """
    if gap_days <= 0 or min_subject_terms < 0 or max_thread_size < 1:
        raise ValueError("gap_days and max_thread_size must be positive")
    if meta["date"].isna().any():
        raise ValueError("every message needs a date to be threaded")

    meta = meta.reset_index(drop=True)
    subjects = meta["subject_norm"].to_numpy()
    sent = ((meta["date"] - _EPOCH).dt.total_seconds()).to_numpy()
    senders = meta["sender"].to_numpy()
    recipients = meta["recipients"].to_numpy()
    gap = gap_days * _SECONDS_PER_DAY

    thread_of = np.full(len(meta), -1, dtype=np.int64)
    next_id = 0

    def new_thread(pos: int, people: set[str]) -> _Thread:
        nonlocal next_id
        thread = _Thread(next_id, sent[pos], people, 1)
        thread_of[pos] = next_id
        next_id += 1
        return thread

    group_sizes = meta["subject_norm"].map(meta["subject_norm"].value_counts()).to_numpy()
    order = meta.sort_values(["subject_norm", "date"], kind="stable").index.to_numpy()
    previous_subject: str | None = None
    open_threads: list[_Thread] = []
    generic = True
    for pos in order:
        subject = subjects[pos]
        if subject != previous_subject:
            previous_subject = subject
            open_threads = []
            generic = _is_too_generic(subject, min_subject_terms)
        if generic or group_sizes[pos] == 1:
            new_thread(pos, set())  # never reused: nothing can join a singleton
            continue

        people = {senders[pos], *recipients[pos]}
        for thread in reversed(open_threads):  # most recently started first
            if (
                sent[pos] - thread.last_sent <= gap
                and thread.size < max_thread_size
                and thread.people & people
            ):
                thread.last_sent = max(thread.last_sent, sent[pos])
                thread.people |= people
                thread.size += 1
                thread_of[pos] = thread.thread_id
                break
        else:
            open_threads.append(new_thread(pos, people))
    return thread_of
