"""RankRAG (Yu et al., 2024) の小規模実装。依存は numpy のみ。

論文の主張の「向き」だけを合成データで確かめる。絶対値の再現は狙わない。
合成データの生成過程・固定定数・oracle はすべて仮定であり、コード中に
仮定として明示する。LLM のファインチューニングそのものは再現しない。

対応:
  exp_rerank_gain    -> Tab.3 (w/o reranking で全タスク低下, Avg 52.6 -> 49.8)
  exp_data_efficiency -> Tab.6 + Fig.4 (約50k で 503k の専門 ranker を上回る、
                        ranking 未学習の生成モデルを ranker に使うと悪化)
  exp_topn_tradeoff  -> Fig.1 対 §5.5/Fig.6 (rerank 後は小さい k=5 で飽和、
                        rerank なしは k を増やしてもノイズで頭打ち)
各実験は DEGENERATE ベースライン(学習なし・rerank なし)を含み、
それが失敗することを確認する。
"""
import numpy as np

SEED = 7
N_TOP = 100      # 仮定: 論文の 8B 設定 (top-N=100) を模した検索深さ
K_GEN = 5        # 仮定: 論文の生成時 top-k=5
MISLEAD = 0.03   # 仮定: top-k 内の distractor 1 件あたり回答を誤らせる確率
N_TEST = 2000


def make_corpus(rng, n_queries, n_cand):
    """合成コーパス。各クエリに golden 1 件 + distractor (n_cand-1) 件。

    仮定: 各 passage は 3 信号を持つ。
      s_lex   = rel + noise      (語彙一致っぽい信号、golden 寄り)
      s_dense = 0.6*rel + noise (bi-encoder っぽい弱い信号。retriever はこれだけ使う)
      s_ans   = rel + 小 noise   (oracle: 答えの有無。統一 (x,c,y) 学習で
                ranking head が触れると仮定した QA 由来の転移信号)
    """
    rel = np.zeros((n_queries, n_cand))
    rel[:, 0] = 1.0  # 0 番目が golden
    s_lex = rel + rng.normal(0, 0.9, rel.shape)
    s_dense = 0.6 * rel + rng.normal(0, 1.0, rel.shape)
    s_ans = rel + rng.normal(0, 0.5, rel.shape)
    return rel, s_lex, s_dense, s_ans


def train_logreg(X, y, rng, iters=2000, lr=0.5):
    """numpy のフルバッチ勾配降下による logistic regression。"""
    w = rng.normal(0, 0.1, X.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g) / len(y)
        b -= lr * g.mean()
    return w, b


def sample_pairs(rng, n_pairs, n_queries=400):
    """ranking 学習ペアを合成。(クエリ、passage特徴、ラベル) を返す。"""
    rel, s_lex, s_dense, s_ans = make_corpus(rng, n_queries, 20)
    idx_q = rng.integers(0, n_queries, n_pairs)
    idx_c = rng.integers(0, 20, n_pairs)
    F = np.stack([s_lex[idx_q, idx_c], s_dense[idx_q, idx_c],
                  s_ans[idx_q, idx_c]], axis=1)
    return F, rel[idx_q, idx_c]


def recall_at_k(order, k):
    """order: (n_queries, n_cand) のランク順序(golden は index 0)。golden が top-k に入る率。"""
    return float((order[:, :k] == 0).any(axis=1).mean())


def em_of(order, k, rng):
    """生成 EM の代理: golden が top-k にある AND distractor に誤らされない。

    仮定: 誤り要因は (a) golden が top-k 外 (recall 不足)、(b) top-k 内の
    distractor が各々確率 MISLEAD で回答を誤らせる (ノイズ混入)。
    (b) は Fig.1 の「k を増やすとノイズで飽和」の再現用。
    """
    has_gold = (order[:, :k] == 0).any(axis=1)
    n_dis = k - has_gold.astype(int)  # top-k 内の distractor 数
    misled = rng.random(len(order)) < (1.0 - (1.0 - MISLEAD) ** n_dis)
    return float((has_gold & ~misled).mean())


def eval_split(rng, w_rank, b_rank, use_ans, n_test=N_TEST):
    """retriever 順 vs rerank 順の order を返す。w_rank=None は学習なし ranker。"""
    rel, s_lex, s_dense, s_ans = make_corpus(rng, n_test, N_TOP)
    order_ret = np.argsort(-s_dense, axis=1)  # backbone retriever (s_dense のみ)
    if w_rank is None:
        order_rerank = np.array([rng.permutation(N_TOP) for _ in range(n_test)])
    else:
        F = np.stack([s_lex, s_dense, s_ans], axis=-1)  # (n, cand, 3)
        if not use_ans:
            F = F[:, :, :2]
        score = F @ w_rank + b_rank
        order_rerank = np.argsort(-score, axis=1)
    return order_ret, order_rerank


def exp_rerank_gain():
    """Tab.3: rerank あり vs なし (w/o reranking は全タスクで低下)。"""
    rng = np.random.default_rng(SEED)
    F, y = sample_pairs(rng, 400)
    w, b = train_logreg(F, y, rng)  # RankRAG 相当: 少量 + s_ans 付き
    order_ret, order_rerank = eval_split(rng, w, b, use_ans=True)
    r = np.random.default_rng(SEED + 100)
    em_no = em_of(order_ret, K_GEN, r)
    em_yes = em_of(order_rerank, K_GEN, np.random.default_rng(SEED + 101))
    print(f"[Tab.3] rerankなし EM@{K_GEN}: {em_no:.3f} (DEGENERATE)")
    print(f"[Tab.3] rerankあり EM@{K_GEN}: {em_yes:.3f} / 改善: {em_yes - em_no:+.3f}")


def exp_data_efficiency():
    """Tab.6 + Fig.4: 少量+統一学習が 10 倍量の専門 ranker を上回り、
    ranking 未学習モデルを ranker にすると backbone より悪化する。"""
    rng = np.random.default_rng(SEED + 1)
    F_small, y_small = sample_pairs(rng, 400)       # RankRAG 相当: ~50k の代理 (1x)
    F_big, y_big = sample_pairs(rng, 4000)          # 専門 ranker 相当: 10x 量
    w_joint, b_joint = train_logreg(F_small, y_small, rng)          # s_ans あり
    w_spec, b_spec = train_logreg(F_big[:, :2], y_big, rng)         # s_ans なし
    order_ret, _ = eval_split(rng, None, 0.0, False)
    r_base = recall_at_k(order_ret, K_GEN)
    _, order_zero = eval_split(np.random.default_rng(SEED + 2), None, 0.0, False)
    r_zero = recall_at_k(order_zero, K_GEN)  # DEGENERATE: 未学習 ranker (ランダム)
    _, order_spec = eval_split(np.random.default_rng(SEED + 3), w_spec, b_spec, False)
    r_spec = recall_at_k(order_spec, K_GEN)
    _, order_joint = eval_split(np.random.default_rng(SEED + 4), w_joint, b_joint, True)
    r_joint = recall_at_k(order_joint, K_GEN)
    print(f"[Tab.6] backbone retriever R@{K_GEN}: {r_base:.3f}")
    print(f"[Tab.6] 未学習 ranker R@{K_GEN}: {r_zero:.3f} (DEGENERATE、backbone を下回る)")
    print(f"[Tab.6] 専門 ranker(10x量) R@{K_GEN}: {r_spec:.3f}")
    print(f"[Tab.6] 統一学習(1x量) R@{K_GEN}: {r_joint:.3f}")


def exp_topn_tradeoff():
    """Fig.1 対 §5.5: rerank 後は小さい k で飽和する。"""
    rng = np.random.default_rng(SEED + 5)
    F, y = sample_pairs(rng, 400)
    w, b = train_logreg(F, y, rng)
    order_ret, order_rerank = eval_split(rng, w, b, use_ans=True)
    for k in (1, 2, 5, 10, 20):
        r = np.random.default_rng(SEED + 200 + k)
        e_no = em_of(order_ret, k, r)
        e_yes = em_of(order_rerank, k, np.random.default_rng(SEED + 300 + k))
        tag = " <- DEGENERATE(小さい k で崩壊)" if (k <= 2) else ""
        print(f"[Fig.1/6] k={k:2d}: rerankなし {e_no:.3f}{tag} / rerankあり {e_yes:.3f}")


if __name__ == "__main__":
    exp_rerank_gain()
    exp_data_efficiency()
    exp_topn_tradeoff()
