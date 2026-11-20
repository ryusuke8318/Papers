"""ブートストラップ区間(numpy のみ)。"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05):
    """平均と、質問を再抽出したときの (1-alpha) 区間。"""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff_ci(a, b, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05):
    """同じ質問での a - b の平均と区間。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    return bootstrap_ci(a - b, n_boot, seed, alpha)


def spans_zero(lo: float, hi: float) -> bool:
    return lo <= 0.0 <= hi
