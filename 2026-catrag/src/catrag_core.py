"""CatRAG(Lau et al., 2026)の縮小再現で使うグラフ・PPR・活性化の拡散の実装。

LLM を使わず、次の置き換えをしている。
- OpenIE の知識グラフ -> 固有表現の候補(大文字始まりの連続語・年・記事タイトルの文字列照合)と、
  文内の共起で作るグラフ。記事タイトルは全文に暗黙に共起する(主題の代用)。
- Fact の検索 -> 文(sentence)の埋め込み検索。上位の文に含まれる語句を起点にする。
- LLM の 4 段階判定 -> 「質問と、辺の根拠になる文の類似度の順位」を 4 段階に分けた倍率。
- 検証済み triple による支持 -> 「質問との類似度の順位が r_key 以内の文が、その段落でその語句を含む」こと。

依存は numpy と scipy のみ。埋め込みは呼び出し側が計算して渡す。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------- 語句の抽出

_TOKEN = re.compile(r"\w+(?:['’\-]\w+)*", re.UNICODE)
_CONNECT = {"of", "the", "de", "la", "le", "von", "van", "der", "del", "du", "da", "di", "for"}
_STOP_CAP = {
    "the", "a", "an", "in", "on", "at", "he", "she", "it", "his", "her", "its", "they", "this", "that",
    "these", "those", "after", "before", "during", "when", "while", "as", "by", "for", "from", "with",
    "although", "however", "but", "and", "or", "if", "who", "whom", "whose", "what", "which", "where",
    "why", "how", "is", "are", "was", "were", "do", "does", "did", "both", "name", "there", "their",
    "we", "i", "you", "of", "to", "not", "also", "since", "one", "some", "many", "most", "other",
    "during", "between", "among", "into", "under", "over", "about", "than", "then", "so", "such",
}


def _strip_possessive(tok: str) -> str:
    for suf in ("'s", "’s"):
        if tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def _is_cap(tok: str) -> bool:
    if tok[0].isupper():
        return True
    return tok.isdigit() and len(tok) == 4 and 1000 <= int(tok) <= 2099


def normalize_key(text: str) -> str:
    """括弧書き(例: 'Ed Wood (film)')を除き、小文字・所有格なしの空白区切りにする。"""
    text = re.sub(r"\([^)]*\)", " ", text)
    toks = [_strip_possessive(t).lower() for t, _, _ in tokenize(text)]
    return " ".join(t for t in toks if t)


def extract_entities(text: str) -> list[str]:
    """大文字始まりの連続語(間に of/the 等の接続語を許す)と 4 桁の年を、正規化したキーで返す。
    句読点(イニシャルの後の '.' を除く)をはさむ語は、同じ語句に含めない。"""
    toks = [(_strip_possessive(t), s, e) for t, s, e in tokenize(text)]

    def joined(i: int) -> bool:  # 語 i-1 と語 i が、空白だけ(またはイニシャルの '.')で続いているか
        gap = text[toks[i - 1][2] : toks[i][1]].strip()
        return gap == "" or (gap == "." and len(toks[i - 1][0]) == 1)

    out: list[str] = []
    i, n = 0, len(toks)
    while i < n:
        if not _is_cap(toks[i][0]):
            i += 1
            continue
        j = i + 1
        while j < n:
            if _is_cap(toks[j][0]) and joined(j):
                j += 1
                continue
            k = j
            while k < n and toks[k][0].lower() in _CONNECT and k - j < 2 and joined(k):
                k += 1
            if k > j and k < n and _is_cap(toks[k][0]) and joined(k):
                j = k + 1
            else:
                break
        run = [t for t, _, _ in toks[i:j]]
        while run and run[0].lower() in _STOP_CAP:
            run = run[1:]
        if run:
            key = " ".join(t.lower() for t in run)
            if len(key) >= 2:
                out.append(key)
        i = j
    return out


def find_ngram_keys(text: str, keys: set[str], max_n: int = 8) -> list[str]:
    toks = [_strip_possessive(t).lower() for t, _, _ in tokenize(text)]
    found = []
    for i in range(len(toks)):
        for n in range(1, max_n + 1):
            if i + n > len(toks):
                break
            k = " ".join(toks[i : i + n])
            if k in keys:
                found.append(k)
    return found


# ---------------------------------------------------------------- グラフ

@dataclass
class GraphIndex:
    n_e: int
    n_p: int
    ent_keys: list[str]
    ent_id: dict[str, int]
    W: sp.csr_matrix                      # 静的な重み。行 = 出る側。列 < n_e は語句、>= n_e は段落
    sent_pass: np.ndarray                 # 文 -> 段落番号
    sent_ents: list[np.ndarray]           # 文 -> 語句 id
    ent_sents: list[np.ndarray]           # 語句 -> 文 id
    ent_npass: np.ndarray                 # 語句を含む段落の数
    pair_sents: dict[tuple[int, int], list[int]]  # (u<v) の語句対 -> 共起する文
    title_key: list[str] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return self.n_e + self.n_p

    def weighted_degree(self) -> np.ndarray:
        return np.asarray(self.W.sum(axis=1)).ravel()

    def match_query_entities(self, question: str) -> list[int]:
        return sorted({self.ent_id[k] for k in extract_entities(question) if k in self.ent_id})


def build_graph(passages: Sequence[dict], max_ents_per_sent: int = 12) -> GraphIndex:
    """passages: [{'title': str, 'sents': [str, ...]}, ...]。段落 i の節点番号は n_e + i。"""
    title_keys = [normalize_key(p["title"]) for p in passages]
    title_set = {k for k in title_keys if k}
    ent_id: dict[str, int] = {}
    ent_keys: list[str] = []

    def eid(k: str) -> int:
        if k not in ent_id:
            ent_id[k] = len(ent_keys)
            ent_keys.append(k)
        return ent_id[k]

    for k in title_keys:
        if k:
            eid(k)

    sent_pass: list[int] = []
    sent_ents_raw: list[list[int]] = []
    for pi, p in enumerate(passages):
        own = title_keys[pi]
        for s in p["sents"]:
            keys = list(dict.fromkeys(extract_entities(s) + find_ngram_keys(s, title_set)))
            if own:
                keys = [own] + [k for k in keys if k != own]
            keys = keys[:max_ents_per_sent]
            sent_pass.append(pi)
            sent_ents_raw.append([eid(k) for k in keys])

    n_e, n_p = len(ent_keys), len(passages)
    rows, cols, vals = [], [], []
    pair_sents: dict[tuple[int, int], list[int]] = {}
    ent_sent_lists: list[list[int]] = [[] for _ in range(n_e)]
    ent_pass_sets: list[set[int]] = [set() for _ in range(n_e)]
    for si, ents in enumerate(sent_ents_raw):
        pi = sent_pass[si]
        for e in ents:
            ent_sent_lists[e].append(si)
            ent_pass_sets[e].add(pi)
        for u, v in combinations(sorted(set(ents)), 2):
            pair_sents.setdefault((u, v), []).append(si)
    for (u, v), ss in pair_sents.items():
        w = float(len(ss))
        rows += [u, v]
        cols += [v, u]
        vals += [w, w]
    for e in range(n_e):
        for pi in ent_pass_sets[e]:
            rows += [e, n_e + pi]
            cols += [n_e + pi, e]
            vals += [1.0, 1.0]
    N = n_e + n_p
    W = sp.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float64)
    W.sum_duplicates()
    return GraphIndex(
        n_e=n_e, n_p=n_p, ent_keys=ent_keys, ent_id=ent_id, W=W,
        sent_pass=np.asarray(sent_pass, dtype=np.int64),
        sent_ents=[np.asarray(x, dtype=np.int64) for x in sent_ents_raw],
        ent_sents=[np.asarray(x, dtype=np.int64) for x in ent_sent_lists],
        ent_npass=np.asarray([len(s) for s in ent_pass_sets], dtype=np.float64),
        pair_sents=pair_sents, title_key=title_keys,
    )


# ---------------------------------------------------------------- PPR

def ppr(W: sp.csr_matrix, seed: np.ndarray, d: float = 0.5, iters: int = 100, tol: float = 1e-10) -> np.ndarray:
    """v <- (1-d) s + d T^T v(T は W の行を正規化したもの)。論文の式(1)。行の和が 0 の節点は質量を落とす。"""
    s = np.asarray(seed, dtype=np.float64)
    s = s / s.sum()
    rs = np.asarray(W.sum(axis=1)).ravel()
    inv = np.divide(1.0, rs, out=np.zeros_like(rs), where=rs > 0)
    Wt = W.T
    v = s.copy()
    for _ in range(iters):
        nv = (1 - d) * s + d * (Wt @ (v * inv))
        if np.abs(nv - v).sum() < tol:
            v = nv
            break
        v = nv
    return v


def ppr_exact(W: sp.csr_matrix, seed: np.ndarray, d: float = 0.5) -> np.ndarray:
    """解析解 v = (1-d)(I - d T^T)^{-1} s(テスト用。小さなグラフ向け)。"""
    Wd = W.toarray()
    rs = Wd.sum(axis=1, keepdims=True)
    T = np.divide(Wd, rs, out=np.zeros_like(Wd), where=rs > 0)
    s = np.asarray(seed, dtype=np.float64)
    s = s / s.sum()
    return (1 - d) * np.linalg.solve(np.eye(len(s)) - d * T.T, s)


# ---------------------------------------------------------------- CatRAG の 3 要素

@dataclass(frozen=True)
class CatConfig:
    d: float = 0.5                  # 論文の表 6: 減衰 0.5
    k_fact: int = 10                # 起点を作る上位の文の数
    fact_temp: float = 0.05         # 文の類似度 -> 起点の重み exp((s-max)/T)
    lam: float = 0.3                # 段落の起点の質量(語句の起点の質量を 1 としたときの比)
    n_pseed: int = 20
    pseed_temp: float = 0.05
    # (1) Symbolic Anchoring
    use_anchor: bool = False
    eps: float = 0.2
    # (2) Query-Aware Dynamic Edge Weighting
    use_edge: bool = False
    n_seed: int = 5
    k_edge: int = 15
    tier_ranks: tuple[int, int, int] = (10, 100, 1000)   # Direct / High / Weak の文の順位の上限
    weak_min: float = 0.2           # 段階 I で外れた辺に掛ける倍率(表 7 の Weak の下限)
    # (3) Key-Fact Passage Weight Enhancement
    use_key: bool = False
    beta: float = 2.5
    r_key: int = 20


def tier_weight(rank: int, ranks: tuple[int, int, int]) -> float:
    """辺の根拠の文の類似度順位 -> 倍率 φ(表 7)。Weak と High は順位の対数で線形補間する。"""
    r1, r2, r3 = ranks
    if rank <= r1:
        return 5.0
    if rank <= r2:
        t = (math.log(rank) - math.log(r1)) / max(math.log(r2) - math.log(r1), 1e-9)
        return 3.0 - 1.0 * t
    if rank <= r3:
        t = (math.log(rank) - math.log(r2)) / max(math.log(r3) - math.log(r2), 1e-9)
        return 0.3 - 0.1 * t
    return 0.0


def seed_entities(idx: GraphIndex, sent_sim: np.ndarray, cfg: CatConfig) -> np.ndarray:
    """上位 k_fact の文に含まれる語句に重みを置く(重み = exp((s-max)/T) / 語句を含む段落数)。和は 1。"""
    top = np.argpartition(-sent_sim, min(cfg.k_fact, len(sent_sim) - 1))[: cfg.k_fact]
    smax = sent_sim[top].max()
    w = np.zeros(idx.n_e)
    for s in top:
        ws = math.exp((sent_sim[s] - smax) / cfg.fact_temp)
        for e in idx.sent_ents[s]:
            w[e] += ws
    w = w / np.maximum(idx.ent_npass, 1.0)
    return w / w.sum()


def passage_seeds(dense: np.ndarray, cfg: CatConfig) -> np.ndarray:
    top = np.argsort(-dense)[: cfg.n_pseed]
    w = np.zeros(len(dense))
    w[top] = np.exp((dense[top] - dense[top].max()) / cfg.pseed_temp)
    return w / w.sum()


def build_seed(idx: GraphIndex, sent_sim: np.ndarray, dense: np.ndarray, anchors: Iterable[int], cfg: CatConfig):
    """起点ベクトル(全節点)と、語句の起点だけの重みを返す。"""
    ent_w = seed_entities(idx, sent_sim, cfg)
    seed = np.zeros(idx.n_nodes)
    seed[: idx.n_e] = ent_w
    anchors = [a for a in anchors]
    if cfg.use_anchor and anchors:
        inv = np.array([1.0 / max(idx.ent_npass[a], 1.0) for a in anchors])
        inv = inv / inv.sum() * cfg.eps       # 起点全体の質量 eps を、段落数の逆数で配分
        for a, x in zip(anchors, inv):
            seed[a] += x
    seed[idx.n_e :] = cfg.lam * passage_seeds(dense, cfg)
    return seed, ent_w


def edge_sim_rank(idx: GraphIndex, u: int, v: int, sent_rank: np.ndarray) -> int:
    ss = idx.pair_sents[(u, v) if u < v else (v, u)]
    return int(min(sent_rank[s] for s in ss))


def modify_graph(idx: GraphIndex, ent_w: np.ndarray, sent_rank: np.ndarray, cfg: CatConfig) -> sp.csr_matrix:
    """質問ごとの辺の重みを持つ W を返す(静的な W は変更しない)。変更するのは seed 語句から出る辺だけ(非対称)。"""
    if not (cfg.use_edge or cfg.use_key):
        return idx.W
    Wq = idx.W.copy()
    seeds = np.argsort(-ent_w)[: cfg.n_seed]
    seeds = [int(u) for u in seeds if ent_w[u] > 0]
    n_e = idx.n_e
    for u in seeds:
        a, b = Wq.indptr[u], Wq.indptr[u + 1]
        cols = Wq.indices[a:b]
        data = Wq.data[a:b]            # Wq.data の view。直接書き換える
        if cfg.use_edge:
            rel = np.where(cols < n_e)[0]
            if len(rel):
                ranks = np.array([edge_sim_rank(idx, u, int(cols[i]), sent_rank) for i in rel])
                keep = np.ones(len(rel), dtype=bool)
                if len(rel) > cfg.k_edge:                         # 段階 I: 上位 k_edge の辺だけを段階 II へ
                    keep[:] = False
                    keep[np.argsort(ranks, kind="stable")[: cfg.k_edge]] = True
                for ii, i in enumerate(rel):
                    if keep[ii]:
                        data[i] *= tier_weight(int(ranks[ii]), cfg.tier_ranks)
                    else:
                        data[i] *= cfg.weak_min
        if cfg.use_key:
            for i in np.where(cols >= n_e)[0]:
                p = int(cols[i]) - n_e
                ss = idx.ent_sents[u]
                if len(ss) and any(idx.sent_pass[s] == p and sent_rank[s] <= cfg.r_key for s in ss):
                    data[i] *= 1.0 + cfg.beta
    return Wq


def ranks_from_sim(sim: np.ndarray) -> np.ndarray:
    """類似度の降順の順位(1 始まり)。"""
    order = np.argsort(-sim, kind="stable")
    r = np.empty(len(sim), dtype=np.int64)
    r[order] = np.arange(1, len(sim) + 1)
    return r


def retrieve(idx: GraphIndex, sent_sim: np.ndarray, dense: np.ndarray, anchors: Iterable[int], cfg: CatConfig):
    """段落の順位(降順の段落番号)と、PPR の全節点の確率を返す。"""
    sent_rank = ranks_from_sim(sent_sim)
    seed, ent_w = build_seed(idx, sent_sim, dense, anchors, cfg)
    Wq = modify_graph(idx, ent_w, sent_rank, cfg)
    v = ppr(Wq, seed, d=cfg.d)
    pscore = v[idx.n_e :]
    return np.argsort(-pscore, kind="stable"), v


# ---------------------------------------------------------------- 活性化の拡散(SA-RAG 風)

def rescale_weight(w: float, c: float) -> float:
    """SA-RAG の再スケール (w-c)/(1-c)。c 以下の辺は 0(無効)にする。"""
    return max(0.0, (w - c) / (1.0 - c)) if c < 1.0 else 0.0


def spreading_activation(
    adj: dict[int, list[tuple[int, float]]],
    init: dict[int, float],
    c: float = 0.4,
    max_hops: int = 3,
    cap: float = 1.0,
) -> dict[int, float]:
    """階層ごとの活性化の拡散。各節点は 1 回だけ展開する(SA-RAG の Algorithm 1 に合わせる)。

    a_target <- min(cap, a_target + w' * a_node)、w' = (w - c)/(1 - c)。初期の節点は init の値から始める。
    """
    act = dict(init)
    visited: set[int] = set()
    frontier = list(init.keys())
    for _ in range(max_hops):
        nxt: list[int] = []
        incoming: dict[int, float] = {}
        for node in frontier:
            if node in visited:
                continue
            visited.add(node)
            for tgt, w in adj.get(node, []):
                w2 = rescale_weight(w, c)
                if w2 > 0:
                    incoming[tgt] = incoming.get(tgt, 0.0) + w2 * act[node]
        for tgt, x in incoming.items():
            act[tgt] = min(cap, act.get(tgt, 0.0) + x)
            if tgt not in visited:
                nxt.append(tgt)
        frontier = nxt
        if not frontier:
            break
    return act


def sa_retrieve(
    idx: GraphIndex, sent_sim: np.ndarray, dense: np.ndarray, c: float = 0.4, gamma: float = 1.0,
    k_fact: int = 10, max_hops: int = 3, tau: float = 0.0,
):
    """SA-RAG 風。語句の辺重み = 質問と、辺の根拠の文の最大類似度。段落の点数は、
    gamma * (拡散で新たに活性化した語句(初期の語句を除く)のうち、段落に含まれるものの活性化の最大値) + 密検索の min-max 点数。"""
    top = np.argsort(-sent_sim)[:k_fact]
    init: dict[int, float] = {}
    for s in top:
        for e in idx.sent_ents[s]:
            init[int(e)] = 1.0
    # 活性化は、初期の語句から max_hops 以内の部分グラフだけで計算する(重みは必要な辺のみ計算)
    adj: dict[int, list[tuple[int, float]]] = {}
    Wc = idx.W
    seen = set(init)
    frontier = list(init)
    for _ in range(max_hops):
        nxt = []
        for u in frontier:
            if u in adj:
                continue
            a, b = Wc.indptr[u], Wc.indptr[u + 1]
            lst = []
            for v in Wc.indices[a:b]:
                v = int(v)
                if v >= idx.n_e:
                    continue
                ss = idx.pair_sents[(u, v) if u < v else (v, u)]
                w = float(max(sent_sim[s] for s in ss))
                lst.append((v, w))
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
            adj[u] = lst
        frontier = nxt
    act = spreading_activation(adj, init, c=c, max_hops=max_hops)
    pscore = np.zeros(idx.n_p)
    # 段落 = 含む語句(その段落の文に現れる語句)の活性化の最大値。語句ごとに段落集合を使う。
    for e, a in act.items():
        if a > tau and e not in init:   # 初期の語句(質問に近い文に出る語句)は、密検索で既に拾われているので数えない
            for p in {int(idx.sent_pass[s]) for s in idx.ent_sents[e]}:
                pscore[p] = max(pscore[p], a)
    dn = (dense - dense.min()) / max(dense.max() - dense.min(), 1e-12)
    total = gamma * pscore + dn
    return np.argsort(-total, kind="stable"), total


# ---------------------------------------------------------------- BM25

class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 0.9, b: float = 0.4):
        self.k1, self.b = k1, b
        self.tok = [[_strip_possessive(t).lower() for t, _, _ in tokenize(d)] for d in docs]
        self.dl = np.array([len(t) for t in self.tok], dtype=np.float64)
        self.avg = self.dl.mean()
        self.inv: dict[str, list[tuple[int, int]]] = {}
        for i, toks in enumerate(self.tok):
            cnt: dict[str, int] = {}
            for t in toks:
                cnt[t] = cnt.get(t, 0) + 1
            for t, c in cnt.items():
                self.inv.setdefault(t, []).append((i, c))
        self.N = len(docs)

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.N)
        for t in set(_strip_possessive(x).lower() for x, _, _ in tokenize(query)):
            post = self.inv.get(t)
            if not post:
                continue
            idf = math.log(1 + (self.N - len(post) + 0.5) / (len(post) + 0.5))
            for i, c in post:
                s[i] += idf * c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg))
        return s


# ---------------------------------------------------------------- ハブの指標(論文 6.1)

def hub_stats(idx: GraphIndex, v: np.ndarray, top_n: int = 10, hub_quantile: float = 0.99) -> dict[str, float]:
    """PPR 後の上位 top_n 語句について、PPR 重み付きの強さ S_ppr と、
    重み付き次数が上位 1% の語句(スーパーハブ)への質量の割合を返す。"""
    strength = idx.weighted_degree()[: idx.n_e]
    ev = v[: idx.n_e]
    top = np.argsort(-ev)[:top_n]
    p = ev[top] / max(ev[top].sum(), 1e-300)
    thr = np.quantile(strength, hub_quantile)
    is_hub = strength >= thr
    return {
        "s_ppr": float((p * strength[top]).sum()),
        "hub_mass_top": float(p[is_hub[top]].sum()),
        "hub_mass_all": float(ev[is_hub].sum() / max(ev.sum(), 1e-300)),
    }
