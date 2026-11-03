"""Zhang et al. (2024) RAFT の要点を合成データで確かめる小規模実装。

対象:
  Fig.6  golden のみで学習すると distractor への頑健性が落ちる
         (distractor 混じりで学習した selector が test 時の文書数変動に強い)
  Fig.5  学習データの P% にだけ golden を入れ、残りは distractor のみにする
         (P<100% が不完全な検索への備えになり、memorization が fallback になる)
  Tab.2  Chain-of-Thought (引用つき推論) の補助 supervision が精度を上げる

依存は numpy のみ。実行: python src/raft_toy.py
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------- selector (Fig.6)
def make_qa_batch(rng, n, dim, n_dist):
    """各問に golden 1 件 + distractor n_dist 件。golden は query 依存。"""
    Q = rng.normal(size=(n, dim))
    G = Q + 0.5 * rng.normal(size=(n, dim))
    D = rng.normal(size=(n, n_dist, dim))
    return Q, G, D


def selector_scores(Q, G, D, W):
    """候補 [golden, distractors...] に対する score。golden の index は 0。"""
    C = np.concatenate([G[:, None, :], D], axis=1)  # (n, 1+n_dist, dim)
    return np.einsum("nd,nkd->nk", Q @ W, C)


def train_selector(Q, G, D, dim, steps=500, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(dim, dim)) * 0.01
    n = len(Q)
    for _ in range(steps):
        S = selector_scores(Q, G, D, W)
        P = softmax(S)
        Y = np.zeros_like(P)
        Y[:, 0] = 1.0
        Grad_S = (P - Y) / n  # (n, k)
        C = np.concatenate([G[:, None, :], D], axis=1)
        # dS/dW = Q^T outer C の平均
        Grad_W = np.einsum("nd,nk,nke->de", Q, Grad_S, C)
        W -= lr * Grad_W
    return W


def eval_selector(W, rng, dim, n_test=1000, k_test=4):
    Q, G, D = make_qa_batch(rng, n_test, dim, k_test)
    S = selector_scores(Q, G, D, W)
    return float((S.argmax(axis=1) == 0).mean())


def exp_distractors():
    dim = 16
    rng = np.random.default_rng(0)
    Qtr, Gtr, Dtr = make_qa_batch(rng, 2000, dim, 4)
    # golden のみ: 候補が golden 1 件だけなので loss=0、W は初期値のまま
    W_golden = train_selector(Qtr, Gtr, Dtr[:, :0, :], dim)
    W_raft = train_selector(Qtr, Gtr, Dtr, dim)
    print("[Fig.6] test 文書数を変えた時の golden 選択精度:")
    for k in [0, 1, 3, 4, 9]:
        a_g = eval_selector(W_golden, rng, dim, k_test=k)
        a_r = eval_selector(W_raft, rng, dim, k_test=k)
        print(f"  test distractor {k}: golden-only {a_g:.3f} / RAFT {a_r:.3f}")


# ---------------------------------------------------------------- P% mixing (Fig.5)
def train_memorizer(Q, Y, dim, n_class, steps=1000, lr=1.0, seed=1):
    """Q -> 答え の線形分類器。golden なしサンプルだけで学習する想定。"""
    rng = np.random.default_rng(seed)
    n = len(Q)
    if n == 0:
        return None  # 学習データなし: chance 回答
    W = rng.normal(size=(dim, n_class)) * 0.01
    b = np.zeros(n_class)
    Yoh = np.zeros((n, n_class))
    Yoh[np.arange(n), Y] = 1.0
    for _ in range(steps):
        P = softmax(Q @ W + b)
        G = (P - Yoh) / n
        W -= lr * Q.T @ G
        b -= lr * G.sum(axis=0)
    return (W, b)


def eval_memorizer(params, Q, Y):
    if params is None:
        return 1.0 / 20
    W, b = params
    return float((softmax(Q @ W + b).argmax(axis=1) == Y).mean())


def exp_p_ratio():
    dim, n_class = 16, 20
    rng = np.random.default_rng(2)
    proto = rng.normal(size=(n_class, dim))
    n_train = 500
    Ytr = rng.integers(0, n_class, size=n_train)
    Qtr = proto[Ytr] + 0.5 * rng.normal(size=(n_train, dim))
    reader_acc = 0.90  # golden がある時の reader 精度 (oracle として固定)
    results = {}
    for P in [1.0, 0.8]:
        n_gold = int(n_train * P)
        # golden なしサンプルだけで memorizer を学習
        mem = train_memorizer(Qtr[n_gold:], Ytr[n_gold:], dim, n_class)
        n_test = 2000
        Yte = rng.integers(0, n_class, size=n_test)
        Qte = proto[Yte] + 0.5 * rng.normal(size=(n_test, dim))
        mem_acc = eval_memorizer(mem, Qte, Yte)
        results[P] = mem_acc
        print(f"[Fig.5] P={P:.1f}: memorizer 精度 {mem_acc:.3f} "
              f"(学習サンプル {n_train - n_gold} 件)")
    for r_fail in [0.0, 0.3]:
        a100 = (1 - r_fail) * reader_acc + r_fail * results[1.0]
        a80 = (1 - r_fail) * reader_acc + r_fail * results[0.8]
        print(f"[Fig.5] 検索失敗率 {r_fail:.1f} での総合精度: "
              f"P=100% {a100:.3f} / P=80% {a80:.3f}")


# ---------------------------------------------------------------- CoT (Tab.2)
def exp_cot():
    """答えのみ (flat 20-way) と CoT 風の階層予測 (グループ推論→絞り込み) の比較。

    CoT の代理: 先に 5-way のグループ (大分類) を推論し、その中で 4-way の
    答えを選ぶ。使う学習データはどちらも同じ 60 件。グループ推論が
    1 グループ 12 件の supervision を共有の手がかりとして効かせる。
    """
    dim = 16
    rng = np.random.default_rng(4)
    group_proto = rng.normal(size=(5, dim))
    proto = group_proto[np.arange(20) // 4] + 0.5 * rng.normal(size=(20, dim))
    n_train, n_test = 60, 2000  # 少データ (1 クラス 3 件、1 グループ 12 件)
    Ytr = np.repeat(np.arange(20), n_train // 20)
    rng.shuffle(Ytr)
    Qtr = proto[Ytr] + 1.0 * rng.normal(size=(n_train, dim))
    Yte = rng.integers(0, 20, size=n_test)
    Qte = proto[Yte] + 1.0 * rng.normal(size=(n_test, dim))
    accs = {False: [], True: []}
    for trial in range(5):
        # 答えのみ: flat 20-way
        flat = train_memorizer(Qtr, Ytr, dim, 20, seed=100 + trial)
        accs[False].append(eval_memorizer(flat, Qte, Yte))
        # CoT: グループ推論 (5-way) + グループ内絞り込み (4-way x5)
        grp = train_memorizer(Qtr, Ytr // 4, dim, 5, seed=200 + trial)
        assert grp is not None
        Wg, bg = grp
        Gte = softmax(Qte @ Wg + bg).argmax(axis=1)
        pred = np.empty(n_test, dtype=int)
        for g in range(5):
            m = Ytr // 4 == g
            loc = train_memorizer(Qtr[m], Ytr[m] % 4, dim, 4,
                                seed=300 + trial * 5 + g)
            assert loc is not None
            Wl, bl = loc
            idx = np.where(Gte == g)[0]
            pred[idx] = g * 4 + softmax(Qte[idx] @ Wl + bl).argmax(axis=1)
        accs[True].append(float((pred == Yte).mean()))
    for use_cot in [False, True]:
        tag = "CoTあり" if use_cot else "答えのみ"
        print(f"[Tab.2] {tag}: テスト精度 {np.mean(accs[use_cot]):.3f} "
              f"(5 seed 平均)")
    print(f"[Tab.2] CoT の改善: +{np.mean(accs[True]) - np.mean(accs[False]):.3f}")


if __name__ == "__main__":
    exp_distractors()
    exp_p_ratio()
    exp_cot()
