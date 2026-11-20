import numpy as np
import pytest

import grapher_metrics as gm


def test_pr_at_k_all_gold_needed():
    assert gm.perfect_recall_at_k([1, 2, 3], {1, 2}, 2) == 1
    assert gm.perfect_recall_at_k([1, 3, 2], {1, 2}, 2) == 0


def test_pr_at_k_edge_cases():
    assert gm.perfect_recall_at_k([1, 2], {1, 2}, 0) == 0
    assert gm.perfect_recall_at_k([1], {1, 2}, 5) == 0       # 候補が足りない
    assert gm.perfect_recall_at_k([1, 2], {1, 2}, 100) == 1  # k が候補数より大きい
    with pytest.raises(ValueError):
        gm.perfect_recall_at_k([1], set(), 1)


def test_pr_leq_recall():
    r = [5, 1, 9, 2]
    assert gm.perfect_recall_at_k(r, {1, 2}, 3) <= gm.recall_at_k(r, {1, 2}, 3)
    assert gm.recall_at_k(r, {1, 2}, 3) == 0.5


def test_bootstrap_ci_contains_mean_and_constant():
    x = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1.0])
    m, lo, hi = gm.bootstrap_ci(x, 2000)
    assert lo <= m <= hi
    assert gm.bootstrap_ci(np.ones(20)) == (1.0, 1.0, 1.0)


def test_paired_diff_detects_shift():
    a = np.ones(200); b = np.zeros(200)
    r = gm.paired_diff_ci(a, b, 500)
    assert r["significant"] and r["diff"] == 1
    z = gm.paired_diff_ci(b, b, 500)
    assert not z["significant"]


def test_regression_counts():
    assert gm.regression_counts([1, 0, 1, 0], [1, 1, 0, 0]) == {"worse": 1, "better": 1, "same": 2}


def test_hub_share():
    deg = np.array([10, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert gm.hub_share([0, 1, 2], deg, 3) == pytest.approx(1 / 3)
    assert gm.hub_share([1, 2, 3], deg, 3) == 0
    assert gm.hub_share([0, 1], np.zeros(10), 2) == 0
