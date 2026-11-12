"""条文・通達の検索(RAG の R)。chunk の作り方、BM25(文字 bigram)、dense(BAAI/bge-m3)。

chunk の作り方
  unit      条文の単位(項・号・通達番号)をそのまま 1 chunk にする(論文の「文書構造に沿った chunking」に相当)
  fixed<N>  全文をつなげて N 文字ごとに切る(論文の固定長 chunking。重なりなし)
検索の成否は「正解の unit に重なる chunk が上位 k 件に入ったか」(hit)で測る。
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


def build_chunks(units: list[dict], mode: str) -> list[dict]:
    if mode == "unit":
        return [{"text": f"【{u['cite']}】{u['text']}", "units": {u["id"]}} for u in units]
    size = int(mode.removeprefix("fixed"))
    doc, spans, pos = [], [], 0
    for u in units:
        s = f"【{u['cite']}】{u['text']}\n"
        doc.append(s)
        spans.append((u["id"], pos, pos + len(s)))
        pos += len(s)
    doc = "".join(doc)
    chunks = []
    for a in range(0, len(doc), size):
        b = min(len(doc), a + size)
        ids = {i for i, s, e in spans if min(b, e) - max(a, s) >= min(40, e - s)}
        chunks.append({"text": doc[a:b].strip(), "units": ids})
    return chunks


def bigrams(s: str) -> list[str]:
    s = "".join(ch for ch in s if not ch.isspace())
    return [s[i:i + 2] for i in range(len(s) - 1)]


class BM25:
    def __init__(self, docs: list[str], k1=1.5, b=0.75):
        self.tf = [Counter(bigrams(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avg = self.len.mean() if len(docs) else 1.0
        df = Counter(g for t in self.tf for g in t)
        n = len(docs)
        self.idf = {g: math.log(1 + (n - c + 0.5) / (c + 0.5)) for g, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, q: str) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for g in set(bigrams(q)):
            if g not in self.idf:
                continue
            for i, t in enumerate(self.tf):
                f = t.get(g, 0)
                if f:
                    out[i] += self.idf[g] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


class Dense:
    def __init__(self, name="BAAI/bge-m3"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.dev).eval()
        if self.dev == "cuda":
            self.model.half()

    def encode(self, texts: list[str], bs=16) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.dev)
                h = self.model(**enc).last_hidden_state[:, 0]
                out.append(self.torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)


class Retriever:
    def __init__(self, units: list[dict], mode: str, method: str, dense: Dense | None = None):
        self.chunks = build_chunks(units, mode)
        self.method = method
        texts = [c["text"] for c in self.chunks]
        if method == "bm25":
            self.bm = BM25(texts)
        else:
            self.dense, self.emb = dense, dense.encode(texts)

    def rank(self, query: str) -> list[int]:
        if self.method == "bm25":
            s = self.bm.scores(query)
        else:
            s = self.emb @ self.dense.encode([query])[0]
        return [int(i) for i in np.argsort(-s, kind="stable")]

    def topk(self, query: str, k: int) -> list[dict]:
        return [self.chunks[i] for i in self.rank(query)[:k]]


def hit(chunks: list[dict], gold: list[str], primary_only: bool = False) -> bool:
    """正解の unit(複数あるときは先頭だけ、または全部)に重なる chunk が含まれるか。全 gold が含まれたら True。"""
    got = set().union(*[c["units"] for c in chunks]) if chunks else set()
    need = gold[:1] if primary_only else gold
    return all(g in got for g in need)


def sweep(units, cases, dense=None, modes=("unit", "fixed150", "fixed300", "fixed600", "fixed1200"), ks=(1, 3, 5, 8),
          methods=("bm25", "dense")) -> list[dict]:
    rows = []
    for mode in modes:
        for method in methods:
            if method == "dense" and dense is None:
                continue
            r = Retriever(units, mode, method, dense)
            ranks = [r.rank(c["text"]) for c in cases]
            for k in ks:
                hs = [hit([r.chunks[i] for i in rk[:k]], c["gold"]) for rk, c in zip(ranks, cases)]
                hp = [hit([r.chunks[i] for i in rk[:k]], c["gold"], True) for rk, c in zip(ranks, cases)]
                ctx = float(np.mean([sum(len(r.chunks[i]["text"]) for i in rk[:k]) for rk in ranks]))
                rows.append({"mode": mode, "method": method, "k": k, "hit_all": float(np.mean(hs)), "hit_primary": float(np.mean(hp)),
                             "n_chunks": len(r.chunks), "ctx_chars": ctx})
    return rows


def pick_best(rows: list[dict], max_ctx_chars: float = 3000) -> dict:
    """dev の hit_all が最大の設定(挿入する文字数が上限以下のものだけ)。"""
    ok = [r for r in rows if r["ctx_chars"] <= max_ctx_chars]
    return max(ok, key=lambda r: (r["hit_all"], -r["ctx_chars"]))


# ---------------------------------------------------------------- small-to-big
def parent_ids(units: list[dict], uid: str, level: str) -> list[str]:
    """small-to-big の「big」側。当たった unit(small)から、一緒に読ませる unit の id を返す(文書順)。

    level
      para     法・令は同じ項に属する unit(例: 令-6-1-3 → 令-6-1 と 令-6-1-1〜10)。通達は前後 1 つ
      article  法・令は同じ条の全 unit。通達は前後 2 つ
    """
    ids = [u["id"] for u in units]
    i = ids.index(uid)
    kind = uid.split("-")[0]
    if kind == "通達":
        w = 1 if level == "para" else 2
        return ids[max(0, i - w):i + w + 1]
    parts = uid.split("-")  # 法-4-3-2 → ["法","4","3","2"]
    keep = 3 if level == "para" else 2  # para: 項まで、article: 条まで
    key = "-".join(parts[:keep])
    return [x for x in ids if x == key or x.startswith(key + "-")]


def expand(units: list[dict], chunks: list[dict], level: str, max_chars: int | None = None) -> list[dict]:
    """検索で当たった chunk(unit 単位)を、親の範囲まで広げる。順位の高い chunk の親から先に入れ、重複は除く。"""
    by_id = {u["id"]: u for u in units}
    seen, out, total = set(), [], 0
    for c in chunks:
        for uid in c["units"]:
            for pid in parent_ids(units, uid, level):
                if pid in seen:
                    continue
                u = by_id[pid]
                t = f"【{u['cite']}】{u['text']}"
                if max_chars is not None and total + len(t) > max_chars:
                    continue
                seen.add(pid)
                out.append({"text": t, "units": {pid}})
                total += len(t)
    return out


def sweep_s2b(units, cases, dense=None, ks=(1, 2, 3, 4, 5), levels=("para", "article"), methods=("bm25", "dense")) -> list[dict]:
    rows = []
    for method in methods:
        if method == "dense" and dense is None:
            continue
        r = Retriever(units, "unit", method, dense)
        ranks = [r.rank(c["text"]) for c in cases]
        for level in levels:
            for k in ks:
                ctxs = [expand(units, [r.chunks[i] for i in rk[:k]], level) for rk in ranks]
                hs = [hit(cx, c["gold"]) for cx, c in zip(ctxs, cases)]
                hp = [hit(cx, c["gold"], True) for cx, c in zip(ctxs, cases)]
                rows.append({"mode": "s2b", "level": level, "method": method, "k": k, "hit_all": float(np.mean(hs)),
                             "hit_primary": float(np.mean(hp)), "ctx_chars": float(np.mean([sum(len(x["text"]) for x in cx) for cx in ctxs]))})
    return rows
