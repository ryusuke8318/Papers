"""Balestriero & LeCun (2026) の理論部分を、有限な世界の上で数値的に確かめる小規模実装。

対象:
  Prop. 1  白色化 JEPA は positive pair のグラフのスペクトル埋め込みを学ぶ
  Thm. 1   positive が共通なら、別々に学習したエンコーダは回転を除いて一致する
  Thm. 2   positive のグラフが違うと、不一致はカーネル差で上から抑えられる
  Prop. 2  回転は k 組のペアで決まる(直交 Procrustes)

依存は numpy のみ。実行: python src/platonic_theory.py
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- 世界の生成
def make_kernel(rng, n, n_clusters, p, noise=0.02):
    """周辺分布が p に一致する対称な positive-pair 分布 Pi を作る。

    クラスタ内で positive が起きやすい構造にして、スペクトルに gap を作る。
    """
    labels = np.arange(n) % n_clusters
    W = noise * rng.random((n, n))
    W += (labels[:, None] == labels[None, :]) * (0.5 + rng.random((n, n)))
    W = (W + W.T) / 2
    s = np.sqrt(p)
    for _ in range(2000):  # 対称 Sinkhorn: diag(s) W diag(s) の行和を p に合わせる
        s = np.sqrt(s * p / (W @ s))
    Pi = s[:, None] * W * s[None, :]
    assert np.allclose(Pi.sum(1), p, atol=1e-8)
    return Pi


def normalized_adjacency(Pi, p):
    d = 1.0 / np.sqrt(p)
    return d[:, None] * Pi * d[None, :]


def spectral_embedding(Pi, p, k):
    """Prop. 1 の Phi_pi(z) = (v_1(z), ..., v_k(z)) / sqrt(p(z)) と固有値を返す。"""
    K = normalized_adjacency(Pi, p)
    mu, V = np.linalg.eigh(K)
    order = np.argsort(-mu)
    mu, V = mu[order], V[:, order]
    return V[:, 1 : k + 1] / np.sqrt(p)[:, None], mu


# ---------------------------------------------------------------- JEPA の学習
def jepa_loss(F, Pi, p, lam):
    """l = ||a-b||^2 の予測項 + 二次の anti-collapse(E f=0, E ff^T=I)を罰則で課す。"""
    k = F.shape[1]
    D = np.diag(p)
    L = np.diag(Pi.sum(1)) - Pi
    pred = 2 * np.trace(F.T @ L @ F)
    C = F.T @ D @ F
    m = F.T @ p
    return pred + lam * (np.sum((C - np.eye(k)) ** 2) + m @ m)


def train_jepa(Pi, p, k, rng, lam=20.0, steps=20000, lr=0.01):
    """エンコーダを埋め込み表として直接学習する(単射なレンダリングでは座標の違いだけ)。"""
    n = len(p)
    F = rng.normal(size=(n, k)) / np.sqrt(k)
    D = np.diag(p)
    L = np.diag(Pi.sum(1)) - Pi
    for _ in range(steps):
        C = F.T @ D @ F
        m = F.T @ p
        grad = 4 * L @ F + lam * (4 * D @ F @ (C - np.eye(k)) + 2 * np.outer(p, m))
        F -= lr * grad
    return F


# ---------------------------------------------------------------- 回転の整合
def procrustes(A, B):
    """min_Q ||A - B Q^T||_F を与える直交行列 Q (A ≈ B Q^T)。"""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt  # Q such that A ≈ B @ Q.T


def rot_error(A, B, weights=None):
    """回転を最適化した後の RMS 距離。weights は世界状態の確率 p(z)。"""
    Q = procrustes(A, B)
    R = A - B @ Q.T
    w = np.ones(len(A)) if weights is None else weights
    return np.sqrt(np.sum(w * np.sum(R**2, axis=1)))


# ---------------------------------------------------------------- 実験
def exp_prop1_thm1(rng, n=30, k=3, n_clusters=4):
    p = rng.dirichlet(np.ones(n) * 5)
    Pi = make_kernel(rng, n, n_clusters, p)
    Phi, mu = spectral_embedding(Pi, p, k)
    print(f"[世界] |Z|={n}, k={k}, 固有値 mu_1..mu_{k+1} = {np.round(mu[1:k+2], 3)}"
          f", gap mu_k - mu_(k+1) = {mu[k]-mu[k+1]:.3f}")

    # Prop. 1: 最小損失 = 2 * sum(1 - mu_i)
    pred_min = 2 * np.sum(1 - mu[1 : k + 1])
    loss_at_phi = jepa_loss(Phi, Pi, p, lam=0.0)
    print(f"[Prop.1] 理論上の最小値 {pred_min:.6f}  Phi での予測損失 {loss_at_phi:.6f}")
    assert np.isclose(pred_min, loss_at_phi)

    # 学習した埋め込み(別の乱数初期化)が Phi と回転を除いて一致するか
    F1 = train_jepa(Pi, p, k, rng)
    e1 = rot_error(F1, Phi, p)
    print(f"[Prop.1] 勾配法で学習した埋め込みと Phi の回転後誤差: {e1:.2e}")

    # Thm. 1: 別々に学習した 3 つのモダリティ。レンダリング g_m は状態の並べ替え
    perms = [rng.permutation(n) for _ in range(3)]
    Fs = []
    for g in perms:
        Pi_m = Pi[np.ix_(g, g)]  # モダリティ m の観測は g(z) と索引される
        p_m = p[g]
        F_m = train_jepa(Pi_m, p_m, k, rng)
        F_world = np.empty_like(F_m)
        F_world[g] = F_m  # 観測の添字から世界状態の添字へ戻す(評価のためだけ)
        Fs.append(F_world)
    for i in range(3):
        for j in range(i + 1, 3):
            print(f"[Thm.1] モダリティ {i} と {j} の回転後誤差: {rot_error(Fs[i], Fs[j], p):.2e}")
    # 回転なしだと一致しない(= 回転の自由度が残っている)ことも確認
    raw = np.sqrt(np.sum(p * np.sum((Fs[0] - Fs[1]) ** 2, axis=1)))
    print(f"[Thm.1] 回転を合わせる前の誤差(参考): {raw:.3f}")
    return p, Pi, Phi, Fs


def exp_prop2(rng, Fs, p, k):
    """Prop. 2: k 組のペアだけで回転が決まり、全データでの Procrustes と一致する。"""
    A, B = Fs[0], Fs[1]
    idx = rng.choice(len(A), size=k, replace=False)
    Q_k = procrustes(A[idx], B[idx])
    Q_all = procrustes(A, B)
    print(f"[Prop.2] {k} 組のペアで推定した回転と全ペアの回転の差 (Frobenius): "
          f"{np.linalg.norm(Q_k - Q_all):.2e}")


def exp_thm2(rng, n=24, k=3, trials=200):
    """Thm. 2: 別のカーネル(無関係なカーネルを重み w で混ぜる)でも誤差は上界以下。"""
    worst = 0.0
    for _ in range(trials):
        p = rng.dirichlet(np.ones(n) * 5)
        Pi1 = make_kernel(rng, n, rng.integers(2, 6), p)
        Pi_other = make_kernel(rng, n, rng.integers(2, 6), p)
        w = 10 ** rng.uniform(-4, 0)
        Pi2 = (1 - w) * Pi1 + w * Pi_other
        Phi1, _ = spectral_embedding(Pi1, p, k)
        Phi2, mu2 = spectral_embedding(Pi2, p, k)
        delta = mu2[k] - mu2[k + 1]
        dK = normalized_adjacency(Pi1, p) - normalized_adjacency(Pi2, p)
        bound = 2 * np.sqrt(2) / delta * min(
            np.sqrt(k) * np.linalg.norm(dK, 2), np.linalg.norm(dK, "fro")
        )
        err = rot_error(Phi1, Phi2, p)
        worst = max(worst, err / bound)
        assert err <= bound, (err, bound)
    print(f"[Thm.2] {trials} 個の世界すべてで 誤差 <= 上界。誤差/上界 の最大値 = {worst:.2f}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    k = 3
    p, Pi, Phi, Fs = exp_prop1_thm1(rng, k=k)
    exp_prop2(rng, Fs, p, k)
    exp_thm2(rng)
