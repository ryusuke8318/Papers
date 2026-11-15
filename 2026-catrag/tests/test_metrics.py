import numpy as np

import catrag_metrics as cm


def test_recall_and_fcr():
    ranked = [5, 1, 9, 3, 7, 2]
    gold = {1, 3}
    assert cm.recall_at_k(ranked, gold, 2) == 0.5
    assert cm.recall_at_k(ranked, gold, 5) == 1.0
    assert cm.full_chain_at_k(ranked, gold, 2) == 0.0
    assert cm.full_chain_at_k(ranked, gold, 4) == 1.0


def test_joint_proxy_needs_both():
    texts = ["Paris is in France", "Berlin", "x", "y"]
    assert cm.joint_proxy_at_k([0, 1], {0, 1}, 2, "France", texts) == 1.0
    assert cm.joint_proxy_at_k([0, 2], {0, 1}, 2, "France", texts) == 0.0
    assert cm.joint_proxy_at_k([0, 1], {0, 1}, 2, "Spain", texts) == 0.0


def test_bootstrap_ci_contains_mean_and_paired_diff():
    rng = np.random.default_rng(0)
    a = rng.binomial(1, 0.7, 300).astype(float)
    m, lo, hi = cm.bootstrap_ci(a, 2000, 0)
    assert lo < m < hi
    r = cm.paired_diff_ci(a, a, 2000)
    assert r["diff"] == 0 and not r["significant"]
    b = np.maximum(a - 0.5 * (rng.random(300) < 0.5), 0)
    assert cm.paired_diff_ci(a, b, 2000)["significant"]
