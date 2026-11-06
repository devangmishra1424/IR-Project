"""Follow-up analyses of the evaluation results (no new retrieval runs needed)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from maillens.evaluation.queries import Query

LENGTH_BINS = [0, 50, 150, 400, np.inf]
LENGTH_LABELS = ["up to 50 words", "51-150 words", "151-400 words", "over 400 words"]


def length_analysis(
    per_query: pd.DataFrame, queries: Sequence[Query], emails: pd.DataFrame
) -> pd.DataFrame:
    """MRR of every system on the test questions, by how long the answer e-mail is.

    Cosine normalisation in TF-IDF divides by the length of the whole e-mail, so it should hurt
    when the answer is long; BM25 normalises length only partly. This table checks that. The
    length of a question's answer is the mean word count of its relevant e-mails.
    """
    test = per_query[(per_query["role"] == "test") & (per_query["kind"] == "question")]
    words = emails["text"].str.split().str.len().to_numpy()
    by_qid = {q.qid: q for q in queries}
    length = {
        qid: float(np.mean([words[d] for d in by_qid[qid].relevant]))
        for qid in test["qid"].unique()
    }
    frame = test.assign(words=test["qid"].map(length))
    frame["answer_length"] = pd.cut(frame["words"], bins=LENGTH_BINS, labels=LENGTH_LABELS)
    grouped = frame.groupby(["answer_length", "system"], observed=True)["MRR"]
    out = grouped.agg(questions="size", MRR="mean").reset_index()
    out["answer_length"] = out["answer_length"].astype(str)
    return out
