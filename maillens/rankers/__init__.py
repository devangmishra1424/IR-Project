"""The four rankers required by the assignment: TF-IDF, BM25, dense and hybrid RRF."""

from maillens.rankers.base import Hit, Ranker
from maillens.rankers.bm25 import Bm25Ranker
from maillens.rankers.dense import DenseRanker, Encoder
from maillens.rankers.rrf import FusedHit, HybridRanker, reciprocal_rank_fusion
from maillens.rankers.tfidf import TfidfRanker

__all__ = [
    "Bm25Ranker",
    "DenseRanker",
    "Encoder",
    "FusedHit",
    "Hit",
    "HybridRanker",
    "Ranker",
    "TfidfRanker",
    "reciprocal_rank_fusion",
]
