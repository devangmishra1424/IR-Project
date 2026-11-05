import numpy as np
import pytest

from maillens.evaluation.stats import paired_bootstrap_ci, paired_randomization_pvalue


def test_identical_systems_are_not_significantly_different():
    scores = [0.1, 0.5, 0.9, 0.3, 0.7]
    mean, low, high = paired_bootstrap_ci(scores, scores, n_resamples=500)
    assert (mean, low, high) == (0.0, 0.0, 0.0)
    assert paired_randomization_pvalue(scores, scores, n_permutations=500) == 1.0


def test_a_consistent_improvement_is_detected():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.2, 0.6, size=200)
    better = base + 0.1 + rng.normal(0, 0.02, size=200)
    mean, low, high = paired_bootstrap_ci(better, base, n_resamples=2000)
    assert 0.09 < mean < 0.11
    assert low > 0.0  # the interval excludes zero
    assert low < mean < high
    assert paired_randomization_pvalue(better, base, n_permutations=2000) < 0.01


def test_pure_noise_is_not_flagged():
    rng = np.random.default_rng(2)
    a, b = rng.uniform(size=300), rng.uniform(size=300)
    assert paired_randomization_pvalue(a, b, n_permutations=2000) > 0.05


def test_results_are_reproducible_for_a_given_seed():
    a, b = [0.2, 0.4, 0.6, 0.8], [0.1, 0.5, 0.2, 0.3]
    assert paired_bootstrap_ci(a, b, n_resamples=300, seed=7) == paired_bootstrap_ci(
        a, b, n_resamples=300, seed=7
    )
    assert paired_randomization_pvalue(a, b, n_permutations=300, seed=7) == (
        paired_randomization_pvalue(a, b, n_permutations=300, seed=7)
    )


def test_pvalue_is_never_exactly_zero():
    assert paired_randomization_pvalue([1.0] * 30, [0.0] * 30, n_permutations=200) > 0.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: paired_bootstrap_ci([1.0], [1.0]),
        lambda: paired_bootstrap_ci([1.0, 2.0], [1.0]),
        lambda: paired_bootstrap_ci([1.0, 2.0], [1.0, 2.0], confidence=1.5),
        lambda: paired_bootstrap_ci([1.0, 2.0], [1.0, 2.0], n_resamples=0),
        lambda: paired_randomization_pvalue([1.0, 2.0], [1.0, 2.0], n_permutations=0),
        lambda: paired_randomization_pvalue([[1.0, 2.0]], [[1.0, 2.0]]),
    ],
)
def test_invalid_input_is_rejected(call):
    with pytest.raises(ValueError):
        call()
