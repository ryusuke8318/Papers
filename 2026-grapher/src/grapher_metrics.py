"""Perfect Recall@K (PR@K)、ブートストラップ区間、対応ありの差、ハブの指標。"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def perfect_recall_at_k(ranked: Sequence[int], gold: set[int], k: int) -> float:
    """正解がすべて上位 k に入っていれば 1。gold が空なら定義できないので ValueError。"""
    if not gold:
        raise ValueError("gold is empty")
    if k <= 0:
        return 0.0
    return float(gold <= set(ranked[:k]))


def recall_at_k(ranked: Sequence[int], gold: set[int], k: int) -> float:
    if not gold:
        raise ValueError("gold is empty")
    return len(set(ranked[:k]) & gold) / len(gold)


def bootstrap_ci(x, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_diff_ci(a, b, n_boot: int = 5000, seed: int = 0) -> dict:
    """a - b の平均と 95% 区間(質問ごとの対応あり)。区間が 0 をまたぐなら「差なし」(同等の証明ではない)。"""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    mean, lo, hi = bootstrap_ci(d, n_boot, seed)
    return {"diff": mean, "lo": lo, "hi": hi, "significant": bool(lo > 0 or hi < 0)}


def regression_counts(a, b) -> dict:
    """質問ごとの 0/1 の指標 a(手法)と b(基準)を比べる。a が悪化した質問数と改善した質問数。"""
    a = np.asarray(a); b = np.asarray(b)
    return {"worse": int((a < b).sum()), "better": int((a > b).sum()), "same": int((a == b).sum())}


def hub_share(ranked_local: Sequence[int], deg: np.ndarray, k: int, top_frac: float = 0.1) -> float:
    """上位 k のうち、候補内で次数が上位 top_frac(同点は含む)に入る節点の割合。

    ranked_local は候補内の添字を点数順に並べたもの。次数が 0 だけの候補集合では 0 を返す。
    """
    if deg.max() <= 0:
        return 0.0
    thr = np.quantile(deg, 1 - top_frac)
    thr = max(thr, 1)
    hubs = set(np.nonzero(deg >= thr)[0].tolist())
    top = list(ranked_local)[:k]
    return sum(1 for i in top if i in hubs) / len(top)
