"""検索の指標(Recall@k、Full Chain Retrieval、Joint Success の代用)とブートストラップ。"""
from __future__ import annotations

import re
import string
from typing import Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], gold: set[int], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold)


def full_chain_at_k(ranked: Sequence[int], gold: set[int], k: int) -> float:
    """正解の支持段落が全部、上位 k に入っていれば 1(論文の FCR)。"""
    return float(gold <= set(ranked[:k]))


def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answer_in_context(answer: str, texts: Sequence[str]) -> bool:
    a = _norm(answer)
    return bool(a) and any(a in _norm(t) for t in texts)


def joint_proxy_at_k(ranked: Sequence[int], gold: set[int], k: int, answer: str, texts: Sequence[str]) -> float:
    """Joint Success の代用。QA の生成は行わず、「FCR を満たし、正解の文字列が上位 k 件に含まれる」で数える。"""
    return float(full_chain_at_k(ranked, gold, k) and answer_in_context(answer, [texts[i] for i in ranked[:k]]))


def bootstrap_ci(x: np.ndarray, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict[str, float]:
    """a - b の平均と 95% 区間(質問ごとの対応あり)。区間が 0 をまたぐなら有意でない。"""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    mean, lo, hi = bootstrap_ci(d, n_boot, seed)
    return {"diff": mean, "lo": lo, "hi": hi, "significant": bool(lo > 0 or hi < 0)}
