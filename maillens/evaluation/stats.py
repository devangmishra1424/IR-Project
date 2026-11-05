"""Paired significance tests over per-query scores.

Two systems are evaluated on the same queries, so the right question is about the per-query
*differences*: is the mean difference larger than chance would produce? Two complementary
answers are given: a bootstrap confidence interval for the mean difference, and a sign-flip
randomisation test for a p-value (Smucker, Allan & Carterette, CIKM 2007, recommend the latter
for IR).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_BATCH = 1000  # resamples drawn at a time, so memory stays bounded for large query sets


def _paired_differences(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("a and b must be one-dimensional and the same length")
    if x.size < 2:
        raise ValueError("need at least two paired observations")
    return x - y


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(mean of a - b, lower, upper) with a percentile-bootstrap interval over queries."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    diffs = _paired_differences(a, b)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, _BATCH):
        size = min(_BATCH, n_resamples - start)
        picks = rng.integers(0, diffs.size, size=(size, diffs.size))
        means[start : start + size] = diffs[picks].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(diffs.mean()), float(low), float(high)


def paired_randomization_pvalue(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided sign-flip test of "the mean paired difference is zero".

    Under the null each query's difference is as likely to be positive as negative, so random
    sign flips show how large a mean difference arises by chance. The +1 terms keep the p-value
    away from an impossible exact zero.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    diffs = _paired_differences(a, b)
    observed = abs(diffs.mean())
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    at_least_as_extreme = 0
    for start in range(0, n_permutations, _BATCH):
        size = min(_BATCH, n_permutations - start)
        signs = rng.choice((-1.0, 1.0), size=(size, diffs.size))
        at_least_as_extreme += int((np.abs((signs * diffs).mean(axis=1)) >= observed - 1e-15).sum())
    return (at_least_as_extreme + 1) / (n_permutations + 1)
