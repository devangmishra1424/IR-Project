import json
from pathlib import Path

import pandas as pd
import pytest

from maillens import config
from maillens.prep.corpus import (
    body_fingerprint,
    build_corpus,
    choose_sample,
    load_emails,
    load_metadata,
)

FOOTER = (
    "\n**********************************************************************\n"
    "This e-mail is the property of Enron Corp. and may contain confidential and privileged "
    "material for the sole use of the intended recipient.\n"
    "**********************************************************************\n"
)


def email(n, subject, sender, to, day, body, file_name="smith-j/inbox/1.", cc=("",), date=None):
    return {
        "message_id": f"<{n}@thyme>",
        "subject": subject,
        "from": sender,
        "to": list(to),
        "cc": list(cc),
        "bcc": [""],
        "date": date or pd.Timestamp(f"2001-03-{day:02d} 10:00", tz="UTC"),
        "body": body,
        "file_name": file_name,
    }


def write_raw(raw: Path) -> None:
    """Three shards: a 3-message thread, singles, and each kind of duplicate / bad row."""
    shard0 = [
        email(1, "Budget review", "a@x.com", ["b@x.com"], 1, "Please review the budget."),
        email(2, "Re: Budget review", "b@x.com", ["a@x.com"], 2, "Looks fine." + FOOTER),
        email(3, "Lunch plans", "c@x.com", ["d@x.com"], 3, "Noon?"),
        email(4, "Gas outlook", "e@x.com", ["f@x.com"], 4, "<b>Prices</b> are rising."),
    ]
    shard1 = [
        # the same e-mail as #3 filed in a mirror folder, with a different Message-ID
        email(12, "Lunch plans", "c@x.com", ["d@x.com"], 3, "Noon?", "smith-j/all_documents/9."),
        email(5, "Re: Budget review", "a@x.com", ["b@x.com"], 4, "Thanks, approved."),
        email(6, "Weekly digest report", "g@x.com", ["h@x.com"], 5, "Digest one"),
        email(7, "Contract draft", "i@x.com", ["j@x.com"], 6, "Draft attached."),
    ]
    shard2 = [
        email(8, "Q3 results", "k@x.com", ["l@x.com"], 7, "Results are in."),
        email(9, "Offsite agenda", "m@x.com", ["n@x.com"], 8, "Agenda below."),
        email(10, "Visa paperwork", "o@x.com", ["p@x.com"], 9, "Forms attached."),
        email(11, "Parking update", "q@x.com", ["r@x.com"], 10, "Lot B closed."),
        # the very same Message-ID as #4, stored again under a mirror folder
        email(
            4,
            "Gas outlook",
            "e@x.com",
            ["f@x.com"],
            4,
            "<b>Prices</b> are rising.",
            "smith-j/discussion_threads/3.",
        ),
        # placeholder timestamp, as found in the real data
        email(
            13,
            "Old thing",
            "s@x.com",
            ["t@x.com"],
            1,
            "ancient",
            date=pd.Timestamp("1980-01-01", tz="UTC"),
        ),
    ]
    base = raw / "enron-emails" / "data"
    base.mkdir(parents=True)
    for i, rows in enumerate([shard0, shard1, shard2]):
        pd.DataFrame(rows).to_parquet(base / f"train-0000{i}-of-00003.parquet", index=False)

    qdir = raw / "questions" / "data"
    qdir.mkdir(parents=True)

    def questions(rows):
        return pd.DataFrame(
            [
                {
                    "id": i,
                    "question": q,
                    "answer": "a",
                    "message_ids": ids,
                    "how_realistic": r,
                    "inbox_address": "x",
                    "query_date": "2001-03-01",
                }
                for i, q, ids, r in rows
            ]
        )

    questions(
        [
            (0, "What did b say about the budget?", ["<2@thyme>"], 0.9),  # answer sits in a thread
            (1, "Which agenda was sent?", ["<9@thyme>"], 0.8),
            (2, "A vague question", ["<7@thyme>"], 0.3),  # below realism cut-off
            (3, "Where is lunch?", ["<12@thyme>"], 0.9),  # points at the mirror copy
        ]
    ).to_parquet(qdir / "test-00000-of-00001.parquet", index=False)
    questions(
        [
            (0, "What is outlook for gas?", ["<4@thyme>"], 0.9),
            (1, "What lot is closed?", ["<11@thyme>"], 0.9),
            (2, "Who approved?", ["<5@thyme>"], 0.9),
        ]
    ).to_parquet(qdir / "train-00000-of-00001.parquet", index=False)


@pytest.fixture
def raw(tmp_path) -> Path:
    write_raw(tmp_path / "raw")
    return tmp_path / "raw"


def test_body_fingerprint_ignores_case_and_whitespace_but_not_content():
    assert body_fingerprint("Hello   World\n") == body_fingerprint("hello world")
    assert body_fingerprint("hello world") != body_fingerprint("hello there")
    assert body_fingerprint(None) == body_fingerprint("")


def test_load_metadata_handles_every_kind_of_duplicate_and_bad_row(raw):
    shards = [raw / "enron-emails" / f for f in config.EMAIL_SHARDS]
    meta, stats, canonical = load_metadata(shards)

    assert stats["raw_rows"] == 14
    assert stats["dropped_implausible_date"] == 1  # the 1980 placeholder
    assert stats["duplicate_message_ids"] == 1  # #4 stored twice under one ID
    assert stats["duplicate_content"] == 1  # #12 is a copy of #3
    assert stats["unique_messages"] == len(meta) == 11

    assert "<12@thyme>" not in set(meta["message_id"])
    assert canonical["<12@thyme>"] == "<3@thyme>"  # the mirror copy maps to the kept one
    assert canonical["<3@thyme>"] == "<3@thyme>"
    assert meta.loc[meta["message_id"] == "<3@thyme>", "folder"].item() == "inbox"
    assert meta.loc[meta["message_id"] == "<4@thyme>", "folder"].item() == "inbox"
    assert all("" not in r for r in meta["recipients"])  # the [''] cc is not a participant
    assert meta.loc[meta["message_id"] == "<2@thyme>", "subject_norm"].item() == "budget review"


def test_choose_sample_always_keeps_required_threads_whole():
    meta = pd.DataFrame(
        {
            "message_id": [f"m{i}" for i in range(10)],
            "thread_id": [0, 0, 0, 1, 2, 3, 4, 5, 6, 7],
        }
    )
    mask = choose_sample(meta, {"m1"}, target=5, seed=3)
    assert mask[:3].all()  # thread 0 (m0..m2) kept entirely
    assert 5 <= mask.sum() <= 7  # topped up with whole threads
    assert choose_sample(meta, {"m1"}, target=5, seed=3).tolist() == mask.tolist()
    with pytest.raises(ValueError, match="target"):
        choose_sample(meta, set(), target=0, seed=1)


def test_build_corpus_end_to_end(raw, tmp_path):
    out = tmp_path / "out"
    stats = build_corpus(raw_dir=raw, out_dir=out, target=7, seed=1, dev_questions=2)

    df = load_emails(out)
    assert json.loads((out / "corpus_stats.json").read_text()) == stats
    assert list(df["doc_id"]) == list(range(len(df)))
    assert df["message_id"].is_unique
    assert len(df) >= 7

    # answers (and their whole threads) are in the corpus: messages 1, 2, 5 share a conversation
    assert {"<2@thyme>", "<9@thyme>", "<3@thyme>"} <= set(df["message_id"])
    assert {"<1@thyme>", "<2@thyme>", "<5@thyme>"} <= set(df["message_id"])

    # threads are consecutive ids, contiguous rows, with correct sizes
    assert df["thread_id"].is_monotonic_increasing
    assert sorted(df["thread_id"].unique()) == list(range(df["thread_id"].nunique()))
    assert (df.groupby("thread_id")["thread_id"].transform("size") == df["thread_size"]).all()
    budget = df[df["message_id"].isin(["<1@thyme>", "<2@thyme>", "<5@thyme>"])]
    assert budget["thread_id"].nunique() == 1 and budget["thread_size"].iloc[0] == 3

    # cleaning was applied
    row2 = df[df["message_id"] == "<2@thyme>"].iloc[0]
    assert row2["fresh"] == "Looks fine."
    assert "enron" not in row2["text"].lower()
    assert df.loc[df["message_id"] == "<4@thyme>", "fresh"].item() == "Prices are rising."

    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].min() >= pd.Timestamp("1997-01-01", tz="UTC")

    # questions: 2 dev + 3 realistic test (the 0.3-realism one is excluded); the question aimed
    # at a mirror copy resolves to the copy that was kept
    assert stats["questions_selected"] == stats["questions_answerable"] == 5
    assert stats["answer_messages"] == 5
    answers = pd.read_parquet(out / "question_answers.parquet")
    assert set(answers["role"]) == {"dev", "test"}
    lunch = answers[answers["question"] == "Where is lunch?"].iloc[0]
    assert list(lunch["answers"]) == ["<3@thyme>"]


def test_build_corpus_is_deterministic(raw, tmp_path):
    a = build_corpus(raw_dir=raw, out_dir=tmp_path / "a", target=7, seed=5, dev_questions=2)
    b = build_corpus(raw_dir=raw, out_dir=tmp_path / "b", target=7, seed=5, dev_questions=2)
    assert a == b
    assert load_emails(tmp_path / "a")["message_id"].tolist() == (
        load_emails(tmp_path / "b")["message_id"].tolist()
    )
