"""The search engine behind the app and the CLI: four rankers over one prepared corpus."""

from __future__ import annotations

import errno
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from maillens import config
from maillens.filters import build_mask
from maillens.index.inverted import InvertedIndex
from maillens.pipeline import DENSE_DIR, SPARSE_DIR, load_dense_ranker
from maillens.prep.corpus import EMAILS_FILE, load_emails
from maillens.rankers.base import Hit, Ranker
from maillens.rankers.bm25 import Bm25Ranker
from maillens.rankers.dense import (
    DenseRanker,
    Encoder,
    SentenceTransformerEncoder,
    split_into_chunks,
)
from maillens.rankers.rrf import FusedHit, reciprocal_rank_fusion
from maillens.rankers.sparse import TermContribution
from maillens.rankers.tfidf import TfidfRanker
from maillens.text.analyzer import Analyzer

log = logging.getLogger(__name__)

PARAMS_FILE = "params.json"
MEMBERS = ("tfidf", "bm25", "dense")  # the rankers fused by the hybrid
SYSTEMS = (*MEMBERS, "hybrid")


def load_params(artifacts_dir: Path = config.ARTIFACTS_DIR) -> dict[str, dict[str, float]]:
    """Tuned ranker parameters written by the evaluation; the defaults if it has not run yet."""
    path = artifacts_dir / PARAMS_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"bm25": {"k1": config.BM25_K1, "b": config.BM25_B}}


@dataclass(frozen=True)
class Comparison:
    """The same query answered by all four rankers."""

    per_ranker: dict[str, list[Hit]]  # tfidf / bm25 / dense / hybrid, each cut to k
    fused: list[FusedHit]  # the hybrid's hits with the ranks and shares behind each score
    # system -> {document id -> rank} over each ranker's whole candidate list (not just the top k),
    # so a result can show where the *other* rankers placed it.
    ranks: dict[str, dict[int, int]]


@dataclass(frozen=True)
class CorpusSummary:
    """Headline numbers about the indexed corpus."""

    emails: int
    threads: int
    mailboxes: int
    first: pd.Timestamp
    last: pd.Timestamp
    terms: int  # distinct index terms
    chunks: int  # embedded passages


class SearchEngine:
    def __init__(
        self,
        emails: pd.DataFrame,
        analyzer: Analyzer,
        tfidf: TfidfRanker,
        bm25: Bm25Ranker,
        dense: DenseRanker,
        *,
        rrf_k: int = config.RRF_K,
        depth: int = config.UI_DEPTH,
    ) -> None:
        if not np.array_equal(emails["doc_id"].to_numpy(), np.arange(len(emails))):
            raise ValueError("doc_id must equal the row position")
        self.emails = emails
        self.analyzer = analyzer
        self.rankers: dict[str, Ranker] = {"tfidf": tfidf, "bm25": bm25, "dense": dense}
        self.tfidf, self.bm25, self.dense = tfidf, bm25, dense
        self.rrf_k = rrf_k
        self.depth = depth
        self._thread_ids = emails["thread_id"].to_numpy()  # sorted: the corpus is stored by thread

    @classmethod
    def load(
        cls,
        *,
        processed_dir: Path = config.PROCESSED_DIR,
        artifacts_dir: Path = config.ARTIFACTS_DIR,
        encoder: Encoder | None = None,
    ) -> SearchEngine:
        # Fail fast, before the slow parts (corpus, indexes, the embedding model) are loaded.
        for required in (
            processed_dir / EMAILS_FILE,
            artifacts_dir / SPARSE_DIR / "main" / "terms.json",
            artifacts_dir / DENSE_DIR / "meta.json",
        ):
            if not required.exists():
                raise FileNotFoundError(errno.ENOENT, "build this stage first", str(required))
        emails = load_emails(processed_dir)
        analyzer = Analyzer()
        index = InvertedIndex.load(artifacts_dir / SPARSE_DIR / "main")
        if index.n_docs != len(emails):
            raise ValueError(
                f"sparse index has {index.n_docs} documents but the corpus has {len(emails)}; "
                "rebuild with `python -m maillens index`"
            )
        params = load_params(artifacts_dir)
        encoder = encoder or SentenceTransformerEncoder()
        return cls(
            emails,
            analyzer,
            TfidfRanker(index, analyzer),
            Bm25Ranker(index, analyzer, **params["bm25"]),
            load_dense_ranker(len(emails), encoder, artifacts_dir / DENSE_DIR),
        )

    # ------------------------------------------------------------------ searching
    def _check(self, query: str, k: int) -> None:
        if not query.strip():
            raise ValueError("the query is empty")
        if not 1 <= k <= self.depth:
            raise ValueError(f"k must be between 1 and {self.depth}, got {k}")

    def compare(self, query: str, k: int, *, allowed: np.ndarray | None = None) -> Comparison:
        """Answer ``query`` with every ranker; the hybrid fuses the three lists found here."""
        self._check(query, k)
        lists = {
            name: self.rankers[name].search(query, self.depth, allowed=allowed) for name in MEMBERS
        }
        fused = reciprocal_rank_fusion(
            {name: [h.doc_id for h in hits] for name, hits in lists.items()}, k=self.rrf_k
        )
        per_ranker = {name: hits[:k] for name, hits in lists.items()}
        per_ranker["hybrid"] = [
            Hit(h.doc_id, h.score, rank) for rank, h in enumerate(fused[:k], start=1)
        ]
        ranks = {name: {h.doc_id: h.rank for h in hits} for name, hits in lists.items()}
        ranks["hybrid"] = {h.doc_id: rank for rank, h in enumerate(fused, start=1)}
        return Comparison(per_ranker, fused[:k], ranks)

    def search(
        self, query: str, system: str, k: int, *, allowed: np.ndarray | None = None
    ) -> list[Hit]:
        if system not in SYSTEMS:
            raise ValueError(f"unknown system {system!r}; choose from {SYSTEMS}")
        return self.compare(query, k, allowed=allowed).per_ranker[system]

    # ------------------------------------------------------------------ explanations
    def explain_terms(self, query: str, system: str, doc_id: int) -> list[TermContribution]:
        """Per-term score breakdown for the two lexical rankers."""
        if system == "tfidf":
            return self.tfidf.explain(query, doc_id)
        if system == "bm25":
            return self.bm25.explain(query, doc_id)
        raise ValueError("term breakdowns exist for 'tfidf' and 'bm25' only")

    def best_passage(self, query: str, doc_id: int) -> tuple[str, float]:
        """The chunk of the e-mail the dense ranker matched, and its cosine similarity."""
        chunk_index, similarity = self.dense.best_chunk(query, doc_id)
        chunks = split_into_chunks(str(self.emails["text"].iloc[doc_id]))
        return chunks[chunk_index], similarity

    # ------------------------------------------------------------------ corpus access
    def email(self, doc_id: int) -> pd.Series:
        return self.emails.iloc[doc_id]

    def thread(self, thread_id: int) -> pd.DataFrame:
        """All messages of a conversation, oldest first."""
        lo = int(np.searchsorted(self._thread_ids, thread_id, side="left"))
        hi = int(np.searchsorted(self._thread_ids, thread_id, side="right"))
        return self.emails.iloc[lo:hi].sort_values("date")

    def mask(
        self,
        *,
        sender_contains: str = "",
        mailbox: str = "",
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> np.ndarray | None:
        return build_mask(
            self.emails,
            sender_contains=sender_contains,
            mailbox=mailbox,
            date_from=date_from,
            date_to=date_to,
        )

    def summary(self) -> CorpusSummary:
        return CorpusSummary(
            emails=len(self.emails),
            threads=int(self.emails["thread_id"].nunique()),
            mailboxes=int(self.emails["mailbox"].nunique()),
            first=self.emails["date"].min(),
            last=self.emails["date"].max(),
            terms=self.tfidf.index.n_terms,
            chunks=int(self.dense.vectors.shape[0]),
        )
