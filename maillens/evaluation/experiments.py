"""The full evaluation protocol: tune on the dev queries, report on the test queries.

1. Tune BM25's ``k1`` and ``b`` on the **dev** queries only.
2. Run every ranker once over all queries and derive the fused variants from those runs.
3. Score and aggregate on the **test** queries: the main comparison table, 95% confidence
   intervals, precision-recall curves, paired significance tests, and the ablations (RRF's ``k``,
   which rankers to fuse, BM25 tuning, and indexing raw vs cleaned text).
"""

from __future__ import annotations

import json
import logging
import platform
import time
from datetime import UTC, datetime
from importlib import metadata
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from maillens import config
from maillens.engine import PARAMS_FILE, SearchEngine
from maillens.evaluation.queries import Query
from maillens.evaluation.runner import (
    Run,
    fuse_runs,
    mean_with_ci,
    pr_curves,
    score_runs,
    significance_table,
    summarize,
)
from maillens.index.inverted import InvertedIndex
from maillens.rankers.base import top_k_indices
from maillens.rankers.bm25 import Bm25Ranker
from maillens.rankers.dense import DenseRanker
from maillens.rankers.sparse import SparseRanker
from maillens.rankers.tfidf import TfidfRanker

log = logging.getLogger(__name__)

MAIN_SYSTEMS = ("tfidf", "bm25", "dense", "hybrid")
BM25_GRID_K1 = (0.6, 0.9, 1.2, 1.5, 2.0)
BM25_GRID_B = (0.3, 0.5, 0.75, 0.9)
RRF_K_SWEEP = (1, 5, 10, 20, 40, 60, 100, 200, 500)
COMPARISONS = (
    ("hybrid", "bm25"),
    ("hybrid", "dense"),
    ("hybrid", "tfidf"),
    ("bm25", "dense"),
    ("bm25", "tfidf"),
    ("dense", "tfidf"),
)


def sparse_run(ranker: SparseRanker, queries: list[Query], depth: int) -> Run:
    """Top-``depth`` documents of a lexical ranker for every query (non-matching ones omitted)."""
    return [
        top_k_indices(ranker.score_all(q.text), depth, positive_only=True).astype(np.int32)
        for q in queries
    ]


def dense_run(dense: DenseRanker, queries: list[Query], depth: int) -> Run:
    """Top-``depth`` documents of the dense ranker; all queries are encoded in one batch."""
    vectors = dense.encoder.encode([q.text for q in queries], batch_size=64)
    return dense.rank_vectors(vectors, depth)


def tune_bm25(
    index: InvertedIndex, engine: SearchEngine, dev: list[Query], depth: int
) -> tuple[dict[str, float], pd.DataFrame]:
    """Grid-search ``k1`` and ``b`` by the mean (over query kinds) of the dev MAP."""
    rows = []
    for k1, b in product(BM25_GRID_K1, BM25_GRID_B):
        run = sparse_run(Bm25Ranker(index, engine.analyzer, k1=k1, b=b), dev, depth)
        by_kind = score_runs(dev, {"bm25": run}, depth=depth).groupby("kind")["MAP"].mean()
        rows.append({"k1": k1, "b": b, **by_kind.to_dict(), "objective": float(by_kind.mean())})
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["objective", "k1", "b"], ascending=[False, True, True]).iloc[0]
    return {"k1": float(best["k1"]), "b": float(best["b"])}, grid


def _versions() -> dict[str, str]:
    names = ("numpy", "pandas", "torch", "sentence-transformers", "scikit-learn", "nltk")
    return {name: metadata.version(name) for name in names}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.4f")


def run_evaluation(
    engine: SearchEngine,
    raw_index: InvertedIndex,
    queries: list[Query],
    *,
    results_dir: Path = config.RESULTS_DIR,
    artifacts_dir: Path = config.ARTIFACTS_DIR,
    processed_dir: Path = config.PROCESSED_DIR,
    depth: int = config.EVAL_DEPTH,
) -> dict[str, object]:
    """Run the whole protocol and write every table to ``results_dir``."""
    started = time.time()
    results_dir.mkdir(parents=True, exist_ok=True)
    index = engine.tfidf.index
    dev = [q for q in queries if q.role == "dev"]
    test = [q for q in queries if q.role == "test"]
    if not dev or not test:
        raise ValueError("need both dev and test queries")
    log.info("queries: %d dev, %d test", len(dev), len(test))

    # 1. tune on dev -----------------------------------------------------------------------
    best, grid = tune_bm25(index, engine, dev, depth)
    _write_csv(grid, results_dir / "bm25_tuning.csv")
    (artifacts_dir / PARAMS_FILE).write_text(json.dumps({"bm25": best}), encoding="utf-8")
    log.info("BM25 tuned on dev: %s", best)
    bm25 = Bm25Ranker(index, engine.analyzer, **best)

    # 2. one run per ranker, then derived variants -------------------------------------------
    runs: dict[str, Run] = {
        "tfidf": sparse_run(engine.tfidf, queries, depth),
        "bm25": sparse_run(bm25, queries, depth),
        "dense": dense_run(engine.dense, queries, depth),
    }
    log.info("rankers done in %.0fs", time.time() - started)
    runs["hybrid"] = fuse_runs(runs, ["tfidf", "bm25", "dense"], k=config.RRF_K, depth=depth)
    runs["hybrid_bm25_dense"] = fuse_runs(runs, ["bm25", "dense"], k=config.RRF_K, depth=depth)
    runs["bm25_default"] = sparse_run(Bm25Ranker(index, engine.analyzer), queries, depth)
    runs["tfidf_raw"] = sparse_run(TfidfRanker(raw_index, engine.analyzer), queries, depth)
    runs["bm25_raw"] = sparse_run(Bm25Ranker(raw_index, engine.analyzer, **best), queries, depth)
    for k in RRF_K_SWEEP:
        if k != config.RRF_K:
            runs[f"hybrid_k{k}"] = fuse_runs(runs, ["tfidf", "bm25", "dense"], k=k, depth=depth)
        else:
            runs[f"hybrid_k{k}"] = runs["hybrid"]
    log.info("fusions done in %.0fs", time.time() - started)

    # 3. score --------------------------------------------------------------------------------
    per_query = score_runs(queries, runs, depth=depth)
    summary = summarize(per_query)
    test_summary = summary[summary["role"] == "test"]
    _write_csv(
        test_summary[test_summary["system"].isin(MAIN_SYSTEMS)], results_dir / "metrics_main.csv"
    )
    _write_csv(summary[summary["role"] == "dev"], results_dir / "metrics_dev.csv")
    _write_csv(test_summary, results_dir / "metrics_all_systems.csv")

    ci_rows = []
    test_pq = per_query[per_query["role"] == "test"]
    for (kind, system), group in test_pq.groupby(["kind", "system"]):
        row: dict[str, object] = {"kind": kind, "system": system, "queries": len(group)}
        for metric in ("MAP", "nDCG@10", "MRR", "P@10", "R@100"):
            mean, low, high = mean_with_ci(group[metric].to_numpy())
            row.update({metric: mean, f"{metric}_low": low, f"{metric}_high": high})
        ci_rows.append(row)
    _write_csv(pd.DataFrame(ci_rows), results_dir / "metrics_ci.csv")

    test_idx = [i for i, q in enumerate(queries) if q.role == "test"]
    curve_runs = {
        name: [runs[name][i] for i in test_idx] for name in (*MAIN_SYSTEMS, "hybrid_bm25_dense")
    }
    _write_csv(pr_curves(test, curve_runs), results_dir / "pr_curves.csv")

    significance = significance_table(per_query, COMPARISONS)
    _write_csv(significance, results_dir / "significance.csv")

    sweep = test_summary[test_summary["system"].str.startswith("hybrid_k")]
    _write_csv(
        sweep[["kind", "system", "MAP", "nDCG@10", "MRR", "P@10", "R@100"]],
        results_dir / "rrf_k_sweep.csv",
    )
    ablation_systems = (
        "tfidf",
        "tfidf_raw",
        "bm25",
        "bm25_default",
        "bm25_raw",
        "dense",
        "hybrid",
        "hybrid_bm25_dense",
    )
    _write_csv(
        test_summary[test_summary["system"].isin(ablation_systems)], results_dir / "ablation.csv"
    )

    per_query.to_csv(results_dir / "per_query.csv.gz", index=False, float_format="%.4f")
    _write_csv(
        pd.DataFrame(
            {
                "qid": [q.qid for q in queries],
                "text": [q.text for q in queries],
                "kind": [q.kind for q in queries],
                "role": [q.role for q in queries],
                "n_relevant": [len(q.relevant) for q in queries],
            }
        ),
        results_dir / "queries.csv",
    )

    corpus_stats = json.loads((processed_dir / "corpus_stats.json").read_text("utf-8"))
    info = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(time.time() - started),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": _versions(),
        "bm25_tuned": best,
        "rrf_k": config.RRF_K,
        "eval_depth": depth,
        "embedding_model": config.EMBED_MODEL,
        "chunking": {
            "words": config.CHUNK_WORDS,
            "stride": config.CHUNK_STRIDE,
            "max_chunks": config.MAX_CHUNKS_PER_EMAIL,
        },
        "queries": {
            f"{role}/{kind}": int(n)
            for (role, kind), n in pd.Series([(q.role, q.kind) for q in queries])
            .value_counts()
            .items()
        },
        "corpus": corpus_stats,
        "index": {
            "documents": index.n_docs,
            "terms": index.n_terms,
            "postings": index.n_postings,
            "dense_chunks": int(engine.dense.vectors.shape[0]),
        },
    }
    (results_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.info("evaluation finished in %.0fs; results in %s", time.time() - started, results_dir)
    return info
