import numpy as np
import pytest

from david_stats import bootstrap_ci, paired_diff_ci, spans_zero


def test_ci_contains_mean_and_ordered():
    v = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    m, lo, hi = bootstrap_ci(v)
    assert lo <= m <= hi and m == 0.6


def test_ci_degenerate():
    assert bootstrap_ci([1.0] * 20) == (1.0, 1.0, 1.0)


def test_ci_width_shrinks_with_n():
    rng = np.random.default_rng(0)
    small = rng.integers(0, 2, 50)
    big = rng.integers(0, 2, 2000)
    w = lambda v: (lambda r: r[2] - r[1])(bootstrap_ci(v))
    assert w(big) < w(small)


def test_paired_diff_zero_for_same():
    v = np.random.default_rng(1).integers(0, 2, 100)
    d, lo, hi = paired_diff_ci(v, v)
    assert d == 0 and spans_zero(lo, hi)


def test_paired_diff_detects_shift():
    a = np.ones(200)
    b = np.zeros(200)
    d, lo, hi = paired_diff_ci(a, b)
    assert d == 1 and not spans_zero(lo, hi)


def test_paired_shape_mismatch():
    with pytest.raises(ValueError):
        paired_diff_ci([1, 2], [1])


def test_ci_reproducible():
    v = [0, 1, 1, 0, 1]
    assert bootstrap_ci(v, seed=3) == bootstrap_ci(v, seed=3)
