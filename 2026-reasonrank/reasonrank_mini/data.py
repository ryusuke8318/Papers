"""HotpotQA(distractor)から、順位付けの学習・評価データを作る。

- 質問に対し、正解の 2 段落(supporting facts)を、BM25 の上位候補の中から見つける課題にする。
  2 つ目の正解は質問文との語の重なりが小さいことが多く(橋渡しの推論が必要)、BM25 だけでは取りこぼす。
- 学習: 正解 2 段落 + BM25 上位の非正解 18 段落 = 20 段落をシャッフル(1 窓)。
- 評価 A: BM25 上位 20 件をそのまま 1 窓で並べ替える。
- 評価 B: BM25 上位 100 件を、窓幅 20・刻み 10 のスライディングウィンドウで並べ替える(論文の設定)。
BM25 の索引は、そのデータ分割の全段落(学習は先頭 train_pool 件の質問、評価は dev 全体)から作る。
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of in on at to for and or is are was were be been by with as from that this it its which who whom what when where "
            "how why did do does has have had not no but than then so if into about over after before between during their his her he she they".split())


def tokenize(s: str) -> list[str]:
    return [t for t in _TOK.findall(s.lower()) if t not in _STOP]


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
        n, V = len(docs), len(vocab)
        tf = sp.csr_matrix((vals, (rows, cols)), shape=(n, V), dtype=np.float32)
        lens_a = np.asarray(lens, dtype=np.float32)
        avg = max(1.0, float(lens_a.mean()))
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * lens_a[tf.row] / avg)
        w = tf.data * (k1 + 1) / denom * self.idf[tf.col]
        self.mat = sp.csr_matrix((w, (tf.row, tf.col)), shape=(n, V), dtype=np.float32).T.tocsr()   # V × n
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


def load_hotpot(split: str, limit: int | None = None) -> list[dict]:
    """[{qid, question, paras: {title: text}, gold: [title, …]}, …]。"""
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    out = []
    for r in ds:
        titles, sents = r["context"]["title"], r["context"]["sentences"]
        paras = {t: t + ". " + " ".join(s).strip() for t, s in zip(titles, sents)}
        gold = list(dict.fromkeys(r["supporting_facts"]["title"]))
        if len(gold) != 2 or any(g not in paras for g in gold):
            continue   # 正解が 2 段落のものだけ使う(指標を揃える)
        out.append({"qid": r["id"], "question": r["question"], "paras": paras, "gold": gold})
        if limit and len(out) >= limit:
            break
    return out


def build_pool(rows: list[dict]) -> tuple[list[str], list[str], dict[str, int]]:
    """全質問の段落を、本文で重複を除いて 1 つの集合にする。戻り値: (識別子, 本文, 本文→番号)。"""
    texts, idx = [], {}
    for r in rows:
        for t in r["paras"].values():
            if t not in idx:
                idx[t] = len(texts)
                texts.append(t)
    ids = [f"p{i}" for i in range(len(texts))]
    return ids, texts, idx


def make_lists(rows: list[dict], pool_rows: list[dict], mode: str, n_cand: int = 20, seed: int = 0, n_queries: int | None = None, hard: bool = True) -> list[dict]:
    """mode: 'train'(正解を必ず含む 20 件をシャッフル) / 'eval20'(BM25 上位 20 件) / 'eval100'(BM25 上位 100 件)。評価は hard=True で、並べ替えの余地がある質問だけにする。"""
    ids, texts, idx = build_pool(pool_rows)
    bm = BM25(texts)
    rng = random.Random(seed)
    sel = list(rows)
    rng.shuffle(sel)
    out = []
    for r in sel:
        gold_ids = [f"p{idx[r['paras'][g]]}" for g in r["gold"]]
        depth = {"train": 40, "eval20": 20, "eval100": 100}[mode]
        top = [ids[i] for i in bm.top(r["question"], depth)]
        if mode == "train":
            neg = [x for x in top if x not in gold_ids][:n_cand - len(gold_ids)]
            cands = gold_ids + neg
            rng.shuffle(cands)
            gold_list = gold_ids + neg
        else:
            cands = top
            present = [g for g in gold_ids if g in cands]
            if not present:
                continue    # 1 つも拾えていない質問は、並べ替えでは救えないので除く
            if hard and [cands.index(g) for g in present] == [0, 1]:
                continue    # BM25 がすでに正解 2 つを 1・2 位に置いている質問は、並べ替えの余地がないので除く
            gold_list = present + [x for x in cands if x not in present]
        rels = {g: 1.0 for g in gold_ids}   # 候補に入っていない正解も、NDCG と Recall の分母に数える
        out.append({"qid": r["qid"], "query": r["question"], "cands": [{"id": c, "text": texts[int(c[1:])]} for c in cands],
                    "rels": rels, "gold_list": gold_list})
        if n_queries and len(out) >= n_queries:
            break
    return out


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]
