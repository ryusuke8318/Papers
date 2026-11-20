"""GraphER の中核: 候補集合上のグラフ構築、PPR、Graph Cohesive Smoothing (GCS)、ハイブリッド検索の点数。

依存は numpy と scipy のみ。名前付き実体の抽出と埋め込みは呼び出し側が行う。
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import scipy.sparse as sp

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an the of in on at to and or is was were are be by for with as from that this it its his her their he she they".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    """スパース行列による BM25 (Robertson の idf、k1=1.2、b=0.75)。1 質問の全文書の点数を 1 回の行列積で出す。"""

    def __init__(self, docs: Sequence[str], k1: float = 1.2, b: float = 0.75):
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lens = np.zeros(len(docs))
        for i, d in enumerate(docs):
            toks = tokenize(d)
            lens[i] = len(toks)
            cnt: dict[int, int] = {}
            for t in toks:
                j = vocab.setdefault(t, len(vocab))
                cnt[j] = cnt.get(j, 0) + 1
            for j, c in cnt.items():
                rows.append(i), cols.append(j), vals.append(c)
        n = len(docs)
        tf = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float64)
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        avg = max(lens.mean(), 1e-9)
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * lens[tf.row] / avg)
        w = tf.data * (k1 + 1) / denom * self.idf[tf.col]
        self.mat = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).tocsc()
        self.vocab = vocab

    def scores(self, query: str) -> np.ndarray:
        ids = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not ids:
            return np.zeros(self.mat.shape[0])
        return np.asarray(self.mat[:, ids].sum(axis=1)).ravel()


def hybrid_scores(cos: np.ndarray, bm25: np.ndarray, w_cos: float = 0.7, w_bm25: float = 0.3) -> np.ndarray:
    """0.7 * cosine + 0.3 * 正規化 BM25。BM25 は質問ごとに、全文書での最大値で割る(最大が 0 なら全部 0)。"""
    m = bm25.max()
    nb = bm25 / m if m > 0 else np.zeros_like(bm25)
    return w_cos * cos + w_bm25 * nb


def top_candidates(score: np.ndarray, n: int = 200) -> np.ndarray:
    n = min(n, len(score))
    idx = np.argpartition(-score, n - 1)[:n]
    return idx[np.argsort(-score[idx], kind="stable")]


def adjacency_from_entities(ents: Sequence[frozenset]) -> np.ndarray:
    """論文の辺の重み: A_ij = |E_i ∩ E_j| / |E_j|(E_j は j の固有な実体の集合)。対角は 0、E_j が空なら A_ij = 0。

    A は一般に非対称で、列 j ごとに分母が決まる。
    """
    n = len(ents)
    vocab: dict = {}
    rows, cols = [], []
    for i, e in enumerate(ents):
        for x in e:
            rows.append(i)
            cols.append(vocab.setdefault(x, len(vocab)))
    if not vocab:
        return np.zeros((n, n))
    B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(vocab)))
    shared = (B @ B.T).toarray()
    sizes = np.asarray(B.sum(axis=1)).ravel()
    A = np.zeros((n, n))
    nz = sizes > 0
    A[:, nz] = shared[:, nz] / sizes[nz]
    np.fill_diagonal(A, 0.0)
    return A


def adjacency_from_pairs(n: int, pairs: Sequence[tuple[int, int]], weight: float = 1.0) -> np.ndarray:
    """構造的近接・文脈的近接(外部キー、隣接チャンク)の辺。無向で重み一定。"""
    A = np.zeros((n, n))
    for i, j in pairs:
        if i != j:
            A[i, j] = A[j, i] = weight
    return A


def row_normalize(A: np.ndarray) -> np.ndarray:
    """W_ij = A_ij / sum_k A_ik (A_ij > 0 のとき)。辺のない行は 0 のまま。"""
    s = A.sum(axis=1, keepdims=True)
    W = np.zeros_like(A, dtype=np.float64)
    np.divide(A, s, out=W, where=s > 0)
    return W


def gcs(A: np.ndarray, s: np.ndarray, alpha: float, tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
    """Graph Cohesive Smoothing (論文の Algorithm 1)。

    p <- alpha * s + (1 - alpha) * W p を L1 残差が tol を下回るまで繰り返し、max(p, s) を返す。
    alpha は seed への重み。alpha = 1 なら p = s。
    """
    s = np.asarray(s, dtype=np.float64)
    W = row_normalize(A)
    p = s.copy()
    for _ in range(max_iter):
        q = alpha * s + (1 - alpha) * (W @ p)
        done = np.abs(q - p).sum() < tol
        p = q
        if done:
            break
    return np.maximum(p, s)


def ppr(A: np.ndarray, s: np.ndarray, alpha: float, tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
    """Personalized PageRank。alpha は再起動確率(論文の damping factor と同じ位置づけ)。

    遷移確率は P = row_normalize(A)。p <- alpha * v + (1 - alpha) * P^T p。
    v = s / sum(s)(seed を確率分布に直したもの)。出る辺のない節点の質量は v に戻す。
    確率が辺をたどって集まるので、入る辺の多い節点(ハブ)に質量が溜まる。
    """
    s = np.asarray(s, dtype=np.float64)
    tot = s.sum()
    v = s / tot if tot > 0 else np.full(len(s), 1.0 / len(s))
    P = row_normalize(A)
    dangling = (A.sum(axis=1) <= 0)
    p = v.copy()
    for _ in range(max_iter):
        leak = p[dangling].sum()
        q = alpha * v + (1 - alpha) * (P.T @ p + leak * v)
        done = np.abs(q - p).sum() < tol
        p = q
        if done:
            break
    return p


def rank_from_scores(cand: np.ndarray, score: np.ndarray) -> list[int]:
    """候補の全体 ID を点数の降順で返す。同点は seed の順位(cand の並び)を優先する。"""
    order = np.argsort(-score, kind="stable")
    return [int(cand[i]) for i in order]


def degrees(A: np.ndarray) -> np.ndarray:
    """候補グラフでの次数(辺の数、A_ij>0 または A_ji>0 の相手の数)。"""
    B = (A > 0) | (A.T > 0)
    return B.sum(axis=1)
