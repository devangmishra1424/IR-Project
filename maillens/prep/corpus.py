"""Build the evaluation corpus from the raw parquet shards.

1. Read the metadata of every message, plus a fingerprint of its body. The Enron dump stores
   the same e-mail many times (once per folder it was filed in: ``all_documents``,
   ``discussion_threads``, ``sent``, ...) under *different* Message-IDs, so duplicates are found
   by content, not by ID. Every original ID is mapped to the copy that is kept.
2. Rebuild conversation threads.
3. Sample *whole threads* until the corpus has about ``target`` e-mails, always including every
   e-mail that a selected question is about. Sampling whole threads keeps thread-level
   relevance judgements complete; sampling single e-mails would shred every conversation.
4. Read the bodies of the chosen e-mails only, clean them, and write one parquet file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from maillens import config
from maillens.prep.cleaning import clean_body, normalize_subject
from maillens.prep.questions import DEV_QUESTIONS, load_questions, pick_questions
from maillens.prep.threads import assign_threads

log = logging.getLogger(__name__)

META_COLUMNS = ["message_id", "subject", "from", "to", "cc", "date", "file_name"]
BATCH_ROWS = 25_000
RAW_BODY_CAP = 5 * config.MAX_BODY_CHARS  # bounds memory while reading; cleaning trims further
EMAILS_FILE = "emails.parquet"
ANSWERS_FILE = "question_answers.parquet"
RAW_BODIES_FILE = "raw_bodies.parquet"
STATS_FILE = "corpus_stats.json"

# The corpus covers 1997-2002; a handful of rows carry placeholder dates (1980, 2043, ...).
PLAUSIBLE_START = pd.Timestamp("1997-01-01", tz="UTC")
PLAUSIBLE_END = pd.Timestamp("2003-01-01", tz="UTC")

_WHITESPACE = re.compile(r"\s+")
_FINGERPRINT_CHARS = 2000


def _iter_batches(paths: Sequence[Path], columns: list[str]) -> Iterator[pa.RecordBatch]:
    for path in paths:
        yield from pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=columns)


def _list_of_str(value: object) -> list[str]:
    """Missing list values arrive as None or NaN, and empty cc/bcc lists as ``['']``.

    Blank entries are dropped: an empty string shared by two messages must not count as a shared
    participant when threads are rebuilt.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [s for s in (str(v).strip() for v in value) if s]  # type: ignore[attr-defined]


def body_fingerprint(body: str | None) -> int:
    """64-bit hash of the body's first 2000 characters, ignoring case and whitespace."""
    # A missing body arrives as None or NaN (NaN is truthy), so test the type explicitly.
    text = _WHITESPACE.sub(" ", body if isinstance(body, str) else "").strip().lower()
    text = text[:_FINGERPRINT_CHARS]
    digest = hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


# Folder preference when several copies of one e-mail exist: a real mailbox folder first, the
# catch-all mirrors (which copy everything) last.
_FOLDER_PRIORITY = {
    "inbox": 0,
    "notes_inbox": 0,
    "sent": 0,
    "sent_items": 0,
    "_sent_mail": 0,
    "all_documents": 2,
    "discussion_threads": 2,
}


def _read_with_fingerprints(shards: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for batch in _iter_batches(shards, [*META_COLUMNS, "body"]):
        frame = batch.to_pandas()
        frame["fingerprint"] = np.fromiter(
            (body_fingerprint(b) for b in frame["body"]), dtype=np.uint64, count=len(frame)
        )
        frames.append(frame.drop(columns="body"))
    return pd.concat(frames, ignore_index=True)


def load_metadata(shards: Sequence[Path]) -> tuple[pd.DataFrame, dict[str, int], pd.Series]:
    """Distinct e-mails (metadata only), corpus counts, and a map original-ID -> kept-ID.

    The map covers every message that survived the date check, so a question that points at a
    duplicate copy still finds the copy that was kept.
    """
    raw = _read_with_fingerprints(shards)
    stats = {"raw_rows": len(raw)}

    kept = raw.dropna(subset=["message_id", "date"]).copy()
    stats["dropped_missing_id_or_date"] = len(raw) - len(kept)
    in_range = (kept["date"] >= PLAUSIBLE_START) & (kept["date"] < PLAUSIBLE_END)
    stats["dropped_implausible_date"] = int((~in_range).sum())
    kept = kept[in_range].copy()

    kept["message_id"] = kept["message_id"].str.strip()
    subjects = kept["subject"].fillna("")
    kept["subject_norm"] = [normalize_subject(s) for s in subjects]
    paths = kept["file_name"].fillna("").str.split("/")
    kept["mailbox"] = paths.str[0].fillna("")
    kept["folder"] = paths.str[1].fillna("")
    kept["_priority"] = kept["folder"].map(_FOLDER_PRIORITY).fillna(1).astype(int)

    # Same Message-ID twice (does not happen in this dump, but costs nothing to guard).
    kept = kept.sort_values("_priority", kind="stable")
    by_id = kept.drop_duplicates("message_id", keep="first")
    stats["duplicate_message_ids"] = len(kept) - len(by_id)

    # Same e-mail filed in several folders: identical sender, minute, subject and body.
    key = ["from", "_minute", "subject_norm", "fingerprint"]
    by_id = by_id.assign(_minute=by_id["date"].dt.floor("min"))
    canonical_id = by_id.groupby(key, sort=False)["message_id"].transform("first")
    kept_rows = by_id[by_id["message_id"] == canonical_id].sort_index()
    stats["duplicate_content"] = len(by_id) - len(kept_rows)
    stats["unique_messages"] = len(kept_rows)
    canonical = pd.Series(canonical_id.to_numpy(), index=by_id["message_id"].to_numpy())

    meta = pd.DataFrame(
        {
            "message_id": kept_rows["message_id"],
            "subject": kept_rows["subject"].fillna(""),
            "subject_norm": kept_rows["subject_norm"],
            "sender": kept_rows["from"].fillna(""),
            "recipients": [
                _list_of_str(to) + _list_of_str(cc)
                for to, cc in zip(kept_rows["to"], kept_rows["cc"], strict=True)
            ],
            "date": pd.to_datetime(kept_rows["date"], utc=True),
            "mailbox": kept_rows["mailbox"],
            "folder": kept_rows["folder"],
        }
    ).reset_index(drop=True)
    return meta, stats, canonical


def choose_sample(
    meta: pd.DataFrame, required_ids: set[str], *, target: int, seed: int
) -> np.ndarray:
    """Boolean row mask: threads holding a required e-mail, then random threads up to ``target``.

    A thread is included with the same probability whatever its size, so every e-mail has the
    same chance of being drawn and the sample keeps the corpus's mix of single messages and long
    conversations.
    """
    if target < 1:
        raise ValueError("target must be positive")
    sizes = meta.groupby("thread_id").size()
    must = pd.Index(meta.loc[meta["message_id"].isin(required_ids), "thread_id"].unique())
    chosen = set(must)
    total = int(sizes.loc[must].sum()) if len(must) else 0
    if total < target:
        rng = np.random.default_rng(seed)
        candidates = rng.permutation(sizes.index.difference(must).to_numpy())
        running = total + np.cumsum(sizes.loc[candidates].to_numpy())
        take = int(np.searchsorted(running, target)) + 1  # first prefix that reaches target
        chosen.update(candidates[:take].tolist())
    return meta["thread_id"].isin(chosen).to_numpy()


def read_bodies(shards: Sequence[Path], wanted: set[str]) -> dict[str, str]:
    """Bodies (capped) of the wanted Message-IDs; the first copy of a repeated ID wins."""
    value_set = pa.array(sorted(wanted), type=pa.string())
    found: dict[str, str] = {}
    for batch in _iter_batches(shards, ["message_id", "body"]):
        hit = pc.is_in(batch.column("message_id"), value_set=value_set)
        if not pc.any(hit).as_py():
            continue
        picked = batch.filter(hit)
        for mid, body in zip(
            picked.column("message_id").to_pylist(),
            picked.column("body").to_pylist(),
            strict=True,
        ):
            found.setdefault(mid.strip(), (body or "")[:RAW_BODY_CAP])
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"{len(missing)} chosen messages were not found when reading bodies")
    return found


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def build_corpus(
    *,
    raw_dir: Path = config.RAW_DIR,
    out_dir: Path = config.PROCESSED_DIR,
    target: int = config.SAMPLE_TARGET,
    seed: int = config.SEED,
    dev_questions: int = DEV_QUESTIONS,
) -> dict[str, int]:
    """Run the whole preparation; writes the e-mails, the question answers and the statistics."""
    shards = [raw_dir / "enron-emails" / f for f in config.EMAIL_SHARDS]
    meta, stats, canonical = load_metadata(shards)
    log.info(
        "metadata: %d raw rows -> %d distinct e-mails (%d content duplicates, %d bad dates)",
        stats["raw_rows"],
        len(meta),
        stats["duplicate_content"],
        stats["dropped_implausible_date"],
    )

    meta["thread_id"] = assign_threads(meta)
    stats["threads_total"] = int(meta["thread_id"].nunique())

    # Questions point at original Message-IDs; translate them to the copies that were kept.
    questions = pick_questions(load_questions(raw_dir), dev_n=dev_questions, seed=seed)
    questions["kept_ids"] = [
        sorted({canonical[m] for m in ids if m in canonical.index})
        for ids in questions["message_ids"]
    ]
    answerable = questions[questions["kept_ids"].map(len) > 0].reset_index(drop=True)
    required = {m for ids in answerable["kept_ids"] for m in ids}
    stats.update(
        questions_selected=len(questions),
        questions_answerable=len(answerable),
        answer_messages=len(required),
    )

    sample = meta.loc[choose_sample(meta, required, target=target, seed=seed)].copy()
    log.info("sampled %d messages in %d threads", len(sample), sample["thread_id"].nunique())

    bodies = read_bodies(shards, set(sample["message_id"]))
    cleaned = [clean_body(bodies[m]) for m in sample["message_id"]]
    sample["fresh"] = [c.fresh for c in cleaned]
    sample["quoted"] = [c.quoted for c in cleaned]

    sample = sample.sort_values(["thread_id", "date", "message_id"], kind="stable")
    sample = sample.reset_index(drop=True)
    sample["thread_id"] = pd.factorize(sample["thread_id"])[0].astype(np.int64)
    sample["thread_size"] = sample.groupby("thread_id")["thread_id"].transform("size")
    sample.insert(0, "doc_id", np.arange(len(sample), dtype=np.int32))

    stats.update(
        sample_messages=len(sample),
        sample_threads=int(sample["thread_id"].nunique()),
        sample_empty_bodies=int(((sample["fresh"] + sample["quoted"]).str.len() == 0).sum()),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(sample, out_dir / EMAILS_FILE)
    # The untouched bodies are kept for the cleaning ablation (raw vs cleaned text).
    raw_bodies = [bodies[m] for m in sample["message_id"]]
    _write_parquet(
        pd.DataFrame({"doc_id": sample["doc_id"], "raw_body": raw_bodies}),
        out_dir / RAW_BODIES_FILE,
    )
    _write_parquet(
        answerable[["qid", "question", "role", "kept_ids"]].rename(columns={"kept_ids": "answers"}),
        out_dir / ANSWERS_FILE,
    )
    (out_dir / STATS_FILE).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("wrote %s (%d rows)", out_dir / EMAILS_FILE, len(sample))
    return stats


def load_emails(out_dir: Path = config.PROCESSED_DIR) -> pd.DataFrame:
    """The prepared corpus; ``text`` is the body the indexes see (fresh, then quoted)."""
    df = pd.read_parquet(out_dir / EMAILS_FILE)
    df["text"] = (df["fresh"] + "\n" + df["quoted"]).str.strip()
    return df


def load_raw_bodies(out_dir: Path = config.PROCESSED_DIR) -> pd.Series:
    """The uncleaned (length-capped) bodies, indexed by doc_id."""
    return pd.read_parquet(out_dir / RAW_BODIES_FILE).set_index("doc_id")["raw_body"]
