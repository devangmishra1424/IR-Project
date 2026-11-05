"""The published Enron question set, used as an independent source of test queries.

Every row is an LLM-written natural-language question about one e-mail, with the Message-ID of
the e-mail that answers it. We use the dataset's own ``test`` split for the final numbers and a
small random draw from its ``train`` split as a development set for tuning, so nothing is tuned
on the questions it is later judged on.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from maillens import config

log = logging.getLogger(__name__)

DEV_QUESTIONS = 300
MIN_REALISM = 0.7  # the dataset's own 0-1 "how realistic is this question" score


def load_questions(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """All questions with columns qid, question, message_ids (list[str]), how_realistic, split."""
    base = raw_dir / "questions" / "data"
    frames = []
    for split in ("train", "test"):
        table = pq.read_table(base / f"{split}-00000-of-00001.parquet")
        df = table.to_pandas()
        df["split"] = split
        df["qid"] = [f"{split}-{i}" for i in df["id"]]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["message_ids"] = [[str(m).strip() for m in ids] for ids in out["message_ids"]]
    return out[["qid", "question", "message_ids", "how_realistic", "split"]]


def pick_questions(
    questions: pd.DataFrame,
    *,
    dev_n: int = DEV_QUESTIONS,
    min_realism: float = MIN_REALISM,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """The questions used for evaluation, labelled ``dev`` (tuning) or ``test`` (reporting).

    Questions below the realism threshold or without an answer e-mail are dropped. The dev set is
    a fixed random draw from the dataset's train split; the test set is its whole test split.
    """
    usable = questions[
        (questions["how_realistic"] >= min_realism) & (questions["message_ids"].map(len) > 0)
    ]
    test = usable[usable["split"] == "test"].assign(role="test")
    train = usable[usable["split"] == "train"]
    if len(train) < dev_n:
        raise ValueError(f"only {len(train)} usable train questions, need {dev_n} for the dev set")
    rng = np.random.default_rng(seed)
    dev = train.iloc[np.sort(rng.choice(len(train), size=dev_n, replace=False))].assign(role="dev")
    return pd.concat([dev, test], ignore_index=True)
