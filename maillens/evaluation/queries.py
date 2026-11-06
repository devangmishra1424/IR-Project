"""Query sets and relevance judgements.

The Enron corpus ships without relevance labels, so two automatic protocols are used (the README
discusses their limits):

* ``question`` - natural-language questions from a published dataset. The e-mail(s) that answer a
  question are its relevant documents (a "known-item" task).
* ``subject`` and ``keyword`` - built from conversation threads. Every e-mail of the thread is
  relevant. A subject query is the thread's subject line; a keyword query is a handful of the
  thread's most characteristic words *that do not appear in the subject*, so it is not a trivial
  copy of the field the index already stores. Both kinds are built for the same threads, which
  makes them directly comparable.

Each query is ``dev`` (used only to tune parameters) or ``test`` (used only for reporting).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from maillens import config
from maillens.prep.corpus import ANSWERS_FILE
from maillens.text.analyzer import Analyzer

QUERIES_FILE = "queries.parquet"
KINDS = ("question", "subject", "keyword")


@dataclass(frozen=True, slots=True)
class Query:
    qid: str
    text: str
    kind: str  # "question" | "subject" | "keyword"
    role: str  # "dev" | "test"
    relevant: frozenset[int]  # document ids


# ----------------------------------------------------------------------------- questions
def question_queries(emails: pd.DataFrame, answers: pd.DataFrame) -> list[Query]:
    """One query per question; relevant documents are the e-mails that answer it."""
    doc_of = dict(zip(emails["message_id"], emails["doc_id"], strict=True))
    out = []
    for row in answers.itertuples(index=False):
        relevant = frozenset(int(doc_of[m]) for m in row.answers)  # KeyError = corpus out of sync
        out.append(Query(str(row.qid), str(row.question), "question", str(row.role), relevant))
    return out


# ----------------------------------------------------------------------------- threads
def eligible_threads(
    emails: pd.DataFrame,
    *,
    min_size: int = 4,
    max_size: int = 30,
    max_span_days: int = 45,
    min_subject_words: int = 3,
) -> pd.Index:
    """Thread ids that make a clean test of thread retrieval.

    Real conversations: several messages, within weeks of each other, with a descriptive subject
    that no other e-mail in the corpus shares (so the thread *is* the full set of e-mails a
    subject query could be expected to find).
    """
    grouped = emails.groupby("thread_id")
    size = grouped.size()
    span = (grouped["date"].max() - grouped["date"].min()).dt.days
    subject = grouped["subject_norm"].first()
    subject_total = subject.map(emails["subject_norm"].value_counts())
    keep = (
        size.between(min_size, max_size)
        & (span <= max_span_days)
        & (subject_total == size)
        & (subject.str.split().str.len() >= min_subject_words)
    )
    return size.index[keep]


def keyword_query(
    thread: pd.DataFrame,
    analyzer: Analyzer,
    doc_freq: Callable[[str], int],
    n_docs: int,
    *,
    n_terms: int = 4,
) -> str | None:
    """The thread's most characteristic words outside its subject, or None if there are too few.

    A word qualifies when it occurs in at least two of the thread's messages (own text only, not
    quoted) and is not in the subject. Candidates are ranked by
    ``messages-containing-it * idf``, so words that are both thread-wide and rare rank first.
    The query uses the words as written, not their stems.
    """
    subject_terms = set(analyzer(str(thread["subject_norm"].iloc[0])))
    messages_with: Counter[str] = Counter()
    surface: dict[str, Counter[str]] = {}
    for body in thread["fresh"]:
        stems_in_message: set[str] = set()
        for token in analyzer.raw_tokens(body):
            if len(token) < 3 or not token.isalpha():
                continue
            stems = analyzer(token)
            if not stems or stems[0] in subject_terms:
                continue
            stem = stems[0]
            stems_in_message.add(stem)
            surface.setdefault(stem, Counter())[token] += 1
        messages_with.update(stems_in_message)

    scored = []
    for stem, count in messages_with.items():
        df = doc_freq(stem)
        if count >= 2 and df > 0:
            scored.append((count * math.log(n_docs / df), stem))
    if len(scored) < n_terms:
        return None
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return " ".join(surface[stem].most_common(1)[0][0] for _, stem in scored[:n_terms])


def thread_queries(
    emails: pd.DataFrame,
    analyzer: Analyzer,
    doc_freq: Callable[[str], int],
    *,
    dev_threads: int = 100,
    test_threads: int = 300,
    seed: int = config.SEED,
) -> list[Query]:
    """Subject and keyword queries for randomly chosen eligible threads (dev first, then test)."""
    members = emails.groupby("thread_id").indices
    rng = np.random.default_rng(seed)
    out: list[Query] = []
    chosen = {"dev": 0, "test": 0}
    for thread_id in rng.permutation(eligible_threads(emails).to_numpy()):
        role = "dev" if chosen["dev"] < dev_threads else "test"
        if chosen[role] >= (dev_threads if role == "dev" else test_threads):
            break
        thread = emails.iloc[members[thread_id]]
        words = keyword_query(thread, analyzer, doc_freq, len(emails))
        if words is None:
            continue  # keep both kinds on exactly the same threads
        relevant = frozenset(int(d) for d in thread["doc_id"])
        out.append(
            Query(
                f"{role}-subject-{thread_id}",
                thread["subject_norm"].iloc[0],
                "subject",
                role,
                relevant,
            )
        )
        out.append(Query(f"{role}-keyword-{thread_id}", words, "keyword", role, relevant))
        chosen[role] += 1
    if chosen["dev"] < dev_threads or chosen["test"] < test_threads:
        raise ValueError(f"only found {chosen} usable threads; lower the requested counts")
    return out


# ----------------------------------------------------------------------------- persistence
def save_queries(queries: list[Query], path: Path) -> None:
    frame = pd.DataFrame(
        {
            "qid": [q.qid for q in queries],
            "text": [q.text for q in queries],
            "kind": [q.kind for q in queries],
            "role": [q.role for q in queries],
            "relevant": [sorted(q.relevant) for q in queries],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def load_queries(path: Path) -> list[Query]:
    frame = pd.read_parquet(path)
    return [
        Query(r.qid, r.text, r.kind, r.role, frozenset(int(d) for d in r.relevant))
        for r in frame.itertuples(index=False)
    ]


def build_all_queries(
    emails: pd.DataFrame,
    analyzer: Analyzer,
    doc_freq: Callable[[str], int],
    processed_dir: Path = config.PROCESSED_DIR,
    *,
    dev_threads: int = 100,
    test_threads: int = 300,
    seed: int = config.SEED,
) -> list[Query]:
    """Build the question and thread query sets, save them, and return them."""
    answers = pd.read_parquet(processed_dir / ANSWERS_FILE)
    queries = question_queries(emails, answers) + thread_queries(
        emails,
        analyzer,
        doc_freq,
        dev_threads=dev_threads,
        test_threads=test_threads,
        seed=seed,
    )
    save_queries(queries, processed_dir / QUERIES_FILE)
    return queries
