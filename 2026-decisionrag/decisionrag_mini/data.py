"""縮小再現のデータ: SQuAD v1.1 から「クラスタ → 記事 → 段落」の木と、問い・正解を作る。

論文の Natural Questions と同じく、記事の上の階層が無いコーパスは、記事をクラスタにまとめて階層を作る。
論文はクラスタの説明を Qwen3-8B の 1 文の要約にしたが、ここでは LLM を使わず、クラスタの中心に近い記事のタイトルを並べる。
コーパスは、SQuAD の train と dev の記事を合わせたもの(記事は重ならない)。評価の問いは dev の問いで、
判定モデルの学習は train の問いだけで行う(dev の記事の問いは見ない。コーパスの段落は、学習時にも文脈として現れる)。
"""
from __future__ import annotations

import math
import random
import re
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of in on at to for and or is are was were be been by with as from that this it its which who whom what when where "
            "how why did do does has have had not no but than then so if into about over after before between during their his her he she they".split())


def tokenize(s: str) -> list[str]:
    return [t for t in _TOK.findall(s.lower()) if t not in _STOP]


def truncate_words(text: str, n: int) -> str:
    w = text.split()
    return text if len(w) <= n else " ".join(w[:n]) + " …"


# ---------------------------------------------------------------- BM25
class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        vocab: dict[str, int] = {}
        rows, cols, vals, lens = [], [], [], []
        for i, d in enumerate(docs):
            toks = tokenize(d)
            lens.append(len(toks))
            cnt: dict[int, int] = {}
            for t in toks:
                j = vocab.setdefault(t, len(vocab))
                cnt[j] = cnt.get(j, 0) + 1
            for j, c in cnt.items():
                rows.append(i)
                cols.append(j)
                vals.append(c)
        n = len(docs)
        tf = sp.csr_matrix((vals, (rows, cols)), shape=(n, max(1, len(vocab))), dtype=np.float32)
        lens_a = np.asarray(lens, dtype=np.float32)
        avg = max(1.0, float(lens_a.mean()))
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        t = tf.tocoo()
        w = t.data * (k1 + 1) / (t.data + k1 * (1 - b + b * lens_a[t.row] / avg)) * self.idf[t.col]
        self.mat = sp.csr_matrix((w, (t.row, t.col)), shape=tf.shape, dtype=np.float32).T.tocsr()   # 語 × 文書
        self.vocab, self.n = vocab, n

    def scores(self, query: str) -> np.ndarray:
        idx = [self.vocab[t] for t in set(tokenize(query)) if t in self.vocab]
        if not idx:
            return np.zeros(self.n, dtype=np.float32)
        return np.asarray(self.mat[idx].sum(axis=0)).ravel()

    def top(self, query: str, k: int) -> list[int]:
        s = self.scores(query)
        k = min(k, self.n)
        part = np.argpartition(-s, k - 1)[:k]
        return part[np.argsort(-s[part], kind="stable")].tolist()


# ---------------------------------------------------------------- SQuAD → コーパスと問い
def load_squad(split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split=split)
    return [{"title": r["title"], "context": r["context"], "question": r["question"], "answer": r["answers"]["text"][0] if r["answers"]["text"] else ""}
            for r in ds]


def build_corpus(train_rows: list[dict], dev_rows: list[dict]) -> tuple[dict, list[dict]]:
    """戻り値: (corpus, questions)。

    corpus = {"articles": [{"id", "title", "split", "paras": [{"id", "text"}]}]}。記事と段落の id は決定的に振る。
    questions = [{"qid", "query", "split", "gold_article", "gold_para"}]。
    """
    arts: dict[str, dict] = {}
    order: list[str] = []
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        for r in rows:
            key = r["title"]
            if key not in arts:
                arts[key] = {"id": f"A{len(order)}", "title": key, "split": split, "paras": [], "_idx": {}}
                order.append(key)
            a = arts[key]
            if r["context"] not in a["_idx"]:
                a["_idx"][r["context"]] = len(a["paras"])
                a["paras"].append({"id": f"{a['id']}_p{len(a['paras'])}", "text": r["context"]})
    questions = []
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        for i, r in enumerate(rows):
            a = arts[r["title"]]
            p = a["paras"][a["_idx"][r["context"]]]
            questions.append({"qid": f"{split}{i}", "query": r["question"], "split": split, "gold_article": a["id"], "gold_para": p["id"]})
    articles = [{k: v for k, v in arts[t].items() if k != "_idx"} for t in order]
    return {"articles": articles}, questions


# ---------------------------------------------------------------- 階層(クラスタ)の構築
def _tfidf(texts: list[str]) -> sp.csr_matrix:
    vocab: dict[str, int] = {}
    rows, cols, vals = [], [], []
    for i, t in enumerate(texts):
        cnt: dict[int, int] = defaultdict(int)
        for w in tokenize(t):
            cnt[vocab.setdefault(w, len(vocab))] += 1
        for j, c in cnt.items():
            rows.append(i)
            cols.append(j)
            vals.append(1 + math.log(c))
    m = sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), max(1, len(vocab))), dtype=np.float32)
    df = np.asarray((m > 0).sum(axis=0)).ravel()
    idf = np.log((1 + len(texts)) / (1 + df)) + 1
    m = m.multiply(idf).tocsr()
    norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()) + 1e-9
    return sp.csr_matrix(m.multiply(1 / norm[:, None]))


def _kmeans_once(x: sp.csr_matrix, k: int, iters: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    n = x.shape[0]
    # k-means++ の初期化(コサイン距離 1 - 類似度)
    first = int(rng.integers(n))
    cent = [x[first].toarray().ravel()]
    for _ in range(1, k):
        sim = np.max(np.stack([np.asarray(x @ c).ravel() for c in cent]), axis=0)
        d = np.clip(1.0 - sim, 0, None) ** 2
        pick = int(rng.choice(n, p=d / d.sum())) if d.sum() > 0 else int(rng.integers(n))
        cent.append(x[pick].toarray().ravel())
    cent = np.stack(cent)
    assign = np.full(n, -1)
    for it in range(iters):
        sim = np.asarray(x @ cent.T)
        new = sim.argmax(axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
        for c in range(k):
            members = np.where(assign == c)[0]
            if len(members) == 0:
                big = np.bincount(assign, minlength=k).argmax()
                idx = np.where(assign == big)[0]
                far = idx[np.asarray(x[idx] @ cent[big]).argmin()]
                assign[far] = c
                members = np.array([far])
            v = np.asarray(x[members].sum(axis=0)).ravel()
            cent[c] = v / (np.linalg.norm(v) + 1e-9)
    score = float(np.asarray(x @ cent.T)[np.arange(n), assign].sum())
    return assign, score


def kmeans_cosine(x: sp.csr_matrix, k: int, iters: int = 30, seed: int = 0, n_init: int = 8) -> np.ndarray:
    """球面 k-means(コサイン類似度、k-means++ の初期化)。n_init 回試して、類似度の合計が最大のものを返す。"""
    rng = np.random.default_rng(seed)
    k = min(k, x.shape[0])
    best, best_score = None, -1.0
    for _ in range(n_init):
        a, sc = _kmeans_once(x, k, iters, rng)
        if sc > best_score:
            best, best_score = a, sc
    return best


def cluster_articles(corpus: dict, k: int, seed: int = 0, n_titles: int = 12) -> tuple[list[int], list[str]]:
    """記事をクラスタにまとめる。戻り値: (記事ごとのクラスタ番号, クラスタごとの説明)。説明は、中心に近い記事のタイトルを並べたもの。"""
    texts = [a["title"].replace("_", " ") + " " + " ".join(a["paras"][0]["text"].split()[:60]) for a in corpus["articles"]]
    x = _tfidf(texts)
    assign = kmeans_cosine(x, k, seed=seed)
    descs = []
    for c in range(assign.max() + 1):
        mem = np.where(assign == c)[0]
        v = np.asarray(x[mem].sum(axis=0)).ravel()
        v /= np.linalg.norm(v) + 1e-9
        order = mem[np.argsort(-np.asarray(x[mem] @ v).ravel())][:n_titles]
        descs.append("; ".join(corpus["articles"][i]["title"].replace("_", " ") for i in order))
    return assign.tolist(), descs


def tree_spec(corpus: dict, assign: list[int], descs: list[str], para_words: int = 60) -> dict:
    """decisionrag_core.build_tree に渡す仕様。根 → クラスタ → 記事 → 段落。"""
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for a, c in zip(corpus["articles"], assign):
        by_cluster[c].append({"id": a["id"], "text": a["title"].replace("_", " "),
                              "children": [{"id": p["id"], "text": truncate_words(p["text"], para_words)} for p in a["paras"]]})
    return {"id": "root", "text": "", "children": [{"id": f"C{c}", "text": descs[c], "children": by_cluster[c]} for c in sorted(by_cluster)]}


def article_to_cluster(corpus: dict, assign: list[int]) -> dict[str, str]:
    return {a["id"]: f"C{c}" for a, c in zip(corpus["articles"], assign)}


# ---------------------------------------------------------------- 学習用の難しい負例
class Retrievers:
    """BM25 の索引を、段落とタイトル(記事の先頭)に作る。負例の選択と、比較用の検索に使う。"""

    def __init__(self, corpus: dict):
        self.corpus = corpus
        self.paras = [(a["id"], p["id"], p["text"]) for a in corpus["articles"] for p in a["paras"]]
        self.para_bm25 = BM25([t for _, _, t in self.paras])
        self.art_ids = [a["id"] for a in corpus["articles"]]
        self.art_bm25 = BM25([a["title"].replace("_", " ") + " " + " ".join(a["paras"][0]["text"].split()[:80]) for a in corpus["articles"]])

    def top_paras(self, query: str, k: int) -> list[str]:
        return [self.paras[i][1] for i in self.para_bm25.top(query, k)]

    def top_articles(self, query: str, k: int) -> list[str]:
        return [self.art_ids[i] for i in self.art_bm25.top(query, k)]


def sample_questions(questions: list[dict], split: str, n: int, seed: int = 0) -> list[dict]:
    qs = [q for q in questions if q["split"] == split]
    random.Random(seed).shuffle(qs)
    return qs[:n]
