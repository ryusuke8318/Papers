"""整合度の評価指標(numpy のみ)。

- mutual k-NN: Huh et al. の PRH ベンチマークと同じ、近傍集合の重なり。何も当てはめない。
- R@1 / R@5: 検証ペアで 1 つの(半)直交写像を当てはめ、テストの画像→キャプションを検索する。
"""
from __future__ import annotations

import numpy as np


def _normalize(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)


def knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    Xn = _normalize(X)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)  # 自分自身は近傍に含めない
    return np.argpartition(-S, k, axis=1)[:, :k]


def mutual_knn(A: np.ndarray, B: np.ndarray, k: int = 10) -> float:
    """A, B は同じ n 個の対象の 2 つの表現。近傍集合の重なりの平均 (偶然の水準は k/(n-1))。"""
    n = len(A)
    ia, ib = knn_indices(A, k), knn_indices(B, k)
    ma = np.zeros((n, n), dtype=bool)
    mb = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    ma[rows, ia] = True
    mb[rows, ib] = True
    return float((ma & mb).sum(1).mean() / k)


def fit_whitening(X: np.ndarray, eps: float = 1e-2):
    """ZCA 白色化 (平均, 行列)。eps は固有値の平均に対する相対値。"""
    mu = X.mean(0)
    C = np.cov(X - mu, rowvar=False)
    s, V = np.linalg.eigh(C)
    s = np.clip(s, 0, None)
    W = (V / np.sqrt(s + eps * s.mean())) @ V.T
    return mu, W


def apply_whitening(X: np.ndarray, wh) -> np.ndarray:
    mu, W = wh
    return (X - mu) @ W


def fit_semi_orthogonal(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A (n×da) を B (n×db) へ写す、直交行を持つ da×db 行列 (da<=db)。直交 Procrustes の一般化。"""
    U, _, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    return U @ Vt


def retrieval_recall(A_mapped: np.ndarray, B: np.ndarray, ks=(1, 5)) -> dict:
    """B の i 番目(画像)から A_mapped の i 番目(キャプション)を探す recall。"""
    S = _normalize(B) @ _normalize(A_mapped).T  # 画像 × キャプション
    true = np.diag(S)[:, None]
    rank = (S > true).sum(1)  # 真の候補より類似度が高いものの数
    return {f"R@{k}": float((rank < k).mean()) for k in ks}


def evaluate_alignment(txt_train, img_train, txt_val, img_val, txt_test, img_test, k: int = 10,
                       eps: float = 1e-2) -> dict:
    """テキスト側(da 次元)と画像側(db 次元)の整合度。

    白色化は訓練画像から推定し、写像は検証ペアだけで当てはめ、テストで評価する。
    """
    wt = fit_whitening(txt_train, eps)
    wi = fit_whitening(img_train, eps)
    W = fit_semi_orthogonal(apply_whitening(txt_val, wt), apply_whitening(img_val, wi))
    t_test = apply_whitening(txt_test, wt)
    i_test = apply_whitening(img_test, wi)
    out = {"mutual_knn": mutual_knn(txt_test, img_test, k)}
    out.update(retrieval_recall(t_test @ W, i_test))
    out["chance_mutual_knn"] = k / (len(txt_test) - 1)
    out["chance_R@1"] = 1 / len(txt_test)
    return out
