"""Run rankers over query sets, score the runs, and aggregate the results.

A *run* is one ranked list of document ids per query, in query order. Runs from the individual
rankers are produced once; every fused variant (different ``k``, different member sets) is then
derived from them without searching again.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from maillens.config import EVAL_DEPTH, RRF_K
from maillens.evaluation.metrics import (
    METRIC_NAMES,
    RECALL_LEVELS,
    evaluate_ranking,
    interpolated_precision,
)
from maillens.evaluation.queries import Query
from maillens.evaluation.stats import paired_bootstrap_ci, paired_randomization_pvalue
from maillens.rankers.rrf import fused_order

Ranking = Sequence[int] | np.ndarray
# One ranking per query. A Sequence (not list), so lists of arrays and lists of lists both fit.
Run = Sequence[Ranking]
Retriever = Callable[[str], Sequence[int]]


def _ints(ranking: Ranking) -> list[int]:
    """Plain Python ints: iterating a numpy array yields numpy scalars, which are much slower."""
    return ranking.tolist() if isinstance(ranking, np.ndarray) else list(ranking)


def rank_queries(queries: Sequence[Query], retrieve: Retriever) -> Run:
    """Ranked document ids (compact int32 arrays) for every query."""
    return [np.asarray(retrieve(q.text), dtype=np.int32) for q in queries]


def fuse_runs(
    runs: Mapping[str, Run],
    members: Sequence[str],
    *,
    k: int = RRF_K,
    depth: int = EVAL_DEPTH,
    weights: Mapping[str, float] | None = None,
) -> Run:
    """Reciprocal rank fusion of the named runs, query by query (unlisted runs weigh 1)."""
    if not members:
        raise ValueError("need at least one run to fuse")
    unknown = [m for m in members if m not in runs]
    if unknown:
        raise ValueError(f"unknown runs: {unknown}")
    n_queries = len(runs[members[0]])
    if any(len(runs[m]) != n_queries for m in members):
        raise ValueError("runs must cover the same queries")
    positional = None if weights is None else [weights.get(m, 1.0) for m in members]
    return [
        np.asarray(
            fused_order([_ints(runs[m][i]) for m in members], k=k, weights=positional)[:depth],
            dtype=np.int32,
        )
        for i in range(n_queries)
    ]


def score_runs(
    queries: Sequence[Query], runs: Mapping[str, Run], *, depth: int = EVAL_DEPTH
) -> pd.DataFrame:
    """One row per (system, query) with every headline metric."""
    rows: list[dict[str, object]] = []
    for system, run in runs.items():
        if len(run) != len(queries):
            raise ValueError(f"run {system!r} has {len(run)} rankings for {len(queries)} queries")
        for query, ranking in zip(queries, run, strict=True):
            cut = _ints(ranking)[:depth]
            first = next((i for i, d in enumerate(cut, start=1) if d in query.relevant), 0)
            rows.append(
                {
                    "system": system,
                    "qid": query.qid,
                    "kind": query.kind,
                    "role": query.role,
                    "n_relevant": len(query.relevant),
                    "first_relevant_rank": first,
                    **evaluate_ranking(cut, query.relevant, depth=depth),
                }
            )
    return pd.DataFrame(rows)


def pr_curves(queries: Sequence[Query], runs: Mapping[str, Run]) -> pd.DataFrame:
    """Mean 11-point interpolated precision per (system, role, kind) and recall level."""
    rows: list[dict[str, object]] = []
    for system, run in runs.items():
        per_group: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
        for query, ranking in zip(queries, run, strict=True):
            per_group[(query.role, query.kind)].append(
                interpolated_precision(_ints(ranking), query.relevant)
            )
        for (role, kind), curves in per_group.items():
            mean = np.mean(curves, axis=0)
            rows.extend(
                {
                    "system": system,
                    "role": role,
                    "kind": kind,
                    "recall": level,
                    "precision": float(p),
                }
                for level, p in zip(RECALL_LEVELS, mean, strict=True)
            )
    return pd.DataFrame(rows)


def summarize(per_query: pd.DataFrame) -> pd.DataFrame:
    """Mean of every metric per (role, kind, system), with the number of queries."""
    grouped = per_query.groupby(["role", "kind", "system"], sort=False)
    out = grouped[list(METRIC_NAMES)].mean()
    out["queries"] = grouped.size()
    return out.reset_index()


def mean_with_ci(
    values: Sequence[float], *, n_resamples: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, lower, upper) with a 95% percentile-bootstrap interval over queries."""
    return paired_bootstrap_ci(values, np.zeros(len(values)), n_resamples=n_resamples, seed=seed)


def significance_table(
    per_query: pd.DataFrame,
    comparisons: Sequence[tuple[str, str]],
    *,
    metrics: Sequence[str] = ("MAP", "nDCG@10"),
    role: str = "test",
    n_resamples: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired tests for each (kind, metric, comparison) on the ``role`` queries.

    ``p_bonferroni`` multiplies the randomisation p-value by the number of comparisons made
    within one (kind, metric) family, the conservative correction for testing several pairs.
    """
    rows = []
    data = per_query[per_query["role"] == role]
    for kind, subset in data.groupby("kind"):
        for metric in metrics:
            wide = subset.pivot(index="qid", columns="system", values=metric)
            for a, b in comparisons:
                x, y = wide[a], wide[b]
                diff, low, high = paired_bootstrap_ci(x, y, n_resamples=n_resamples, seed=seed)
                p = paired_randomization_pvalue(x, y, n_permutations=n_resamples, seed=seed)
                rows.append(
                    {
                        "kind": kind,
                        "metric": metric,
                        "system_a": a,
                        "system_b": b,
                        "mean_a": float(x.mean()),
                        "mean_b": float(y.mean()),
                        "diff": diff,
                        "ci_low": low,
                        "ci_high": high,
                        "p_value": p,
                        "p_bonferroni": min(1.0, p * len(comparisons)),
                        "queries": len(x),
                    }
                )
    return pd.DataFrame(rows)
