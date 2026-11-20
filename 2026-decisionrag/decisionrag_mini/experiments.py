"""論文の分析に対応する実験(検証は GPU の判定モデルで行う。LexicalJudge は、配線の確認と、学習なしの下限の比較に使う)。

  main            Tab. 2 / 8: BM25、BM25 上位 100 件を判定モデルで並べ替え(Jev Rerank)、DecisionRAG(木をたどる)、+ BM25 の候補
  pool_scaling    Fig. 5: 候補数 N を増やしたときの Hit@1(二値を 1 件ずつ / 二値を全部 1 請求 / 複数選択 / 二値の上位を複数選択)
  request_size    Fig. 6: 候補 200 件の池を、1 請求あたり m 件に分けたときの Hit@1
  choice          Tab. 7: 複数選択による並べ替えの有無
  budget          Fig. 7 / Tab. 9: 残すクラスタ・記事の数と、再現率・判定数・時間
  latency         式 4: 請求の長さ(候補数)と時間、段の数と遅延
各関数は、質問ごとの指標(per_query)と、平均(summary)を返す。信頼区間は make_figure.py でブートストラップする。
"""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import Retrievers, tokenize  # noqa: E402
from decisionrag_core import (NavConfig, Node, build_tree, flat_rerank, hit_at_k, judge_pooled, latency_estimate, navigate, ndcg_at_k,  # noqa: E402
                              recall_at_k, top_indices)


class Ctx:
    """コーパス・木・検索器・問いをまとめたもの。"""

    def __init__(self, corpus: dict, questions: list[dict], spec: dict):
        self.corpus, self.questions = corpus, questions
        self.root = build_tree(spec)
        self.nodes: dict[str, Node] = {}
        stack = [self.root]
        while stack:
            n = stack.pop()
            self.nodes[n.id] = n
            stack += n.children
        self.retr = Retrievers(corpus)
        self.n_clusters = len(self.root.children)
        self.n_pages = sum(len(c.children) for c in self.root.children)
        self.n_paras = sum(len(p.children) for c in self.root.children for p in c.children)

    def dev(self, n: int, seed: int = 0) -> list[dict]:
        qs = [q for q in self.questions if q["split"] == "dev"]
        random.Random(seed).shuffle(qs)
        return qs[:n]


# ---------------------------------------------------------------- 配線の確認用と、学習なしの下限用の判定器
class LexicalJudge:
    """語彙の重なりで採点する。請求の中の候補で idf を計算するので、確率は、同じ請求の候補に左右される(共有文脈の簡易版)。"""

    def __init__(self, sharpness: float = 8.0):
        self.k = sharpness
        self.stats = {"requests": 0, "tokens": 0, "judgments": 0}

    def _scores(self, query: str, cands) -> list[float]:
        q = set(tokenize(query))
        docs = [set(tokenize(c.text)) for c in cands]
        n = len(docs)
        idf = {t: math.log(1 + (n + 1) / (1 + sum(t in d for d in docs))) for t in q}
        tot = sum(idf.values()) or 1.0
        self.stats["requests"] += 1
        self.stats["judgments"] += n
        self.stats["tokens"] += sum(len(c.text.split()) for c in cands)
        return [sum(idf[t] for t in q & d) / tot for d in docs]

    def binary(self, query: str, candidates) -> list[float]:
        s = self._scores(query, list(candidates))
        return [1 / (1 + math.exp(-self.k * (x - 0.4))) for x in s]

    def choice(self, query: str, candidates) -> list[float]:
        s = self._scores(query, list(candidates))
        e = [math.exp(self.k * x) for x in s]
        z = sum(e)
        return [x / z for x in e]


# ---------------------------------------------------------------- 共通
def _delta(judge, before: dict) -> dict:
    st = getattr(judge, "stats", None)
    return {k: st[k] - before[k] for k in st} if st else {}


def _snap(judge) -> dict:
    return dict(getattr(judge, "stats", {}) or {})


def _metrics(ranked_ids: list[str], gold: set[str]) -> dict:
    return {"hit1": hit_at_k(ranked_ids, gold, 1), "r10": recall_at_k(ranked_ids, gold, 10), "ndcg10": ndcg_at_k(ranked_ids, gold, 10),
            "r20": recall_at_k(ranked_ids, gold, 20)}


def _summary(per_query: dict[str, list[float]]) -> dict:
    return {k: sum(v) / len(v) for k, v in per_query.items() if v}


def _collect(rows: list[dict]) -> dict[str, list[float]]:
    keys = rows[0].keys() if rows else []
    return {k: [r[k] for r in rows] for k in keys}


def _bm25_nodes(ctx: Ctx, query: str, k: int) -> list[Node]:
    return [ctx.nodes[p] for p in ctx.retr.top_paras(query, k)]


# ---------------------------------------------------------------- 実験
def exp_main(ctx: Ctx, judge, qs: list[dict], k1: int = 3, k2: int = 2, r: int = 20, m: int = 5, pool: int = 100) -> dict:
    methods = ["bm25", "bm25_top100_judge", "decisionrag", "decisionrag_bm25"]
    rows: dict[str, list[dict]] = {x: [] for x in methods}
    for q in qs:
        gold = {q["gold_para"]}
        query = q["query"]
        t0 = time.perf_counter()
        bm = _bm25_nodes(ctx, query, max(pool, 20))
        t_bm = time.perf_counter() - t0
        rows["bm25"].append({**_metrics([n.id for n in bm[:20]], gold), "sec": t_bm, "judgments": 0, "tokens": 0, "stages": 0})
        for name, fn in (
            ("bm25_top100_judge", lambda: flat_rerank(judge, query, bm[:pool], r=r, m=m)),
            ("decisionrag", lambda: navigate(ctx.root, query, judge, NavConfig(budgets=[k1, k2], r=r, m=m))),
            ("decisionrag_bm25", lambda: navigate(ctx.root, query, judge, NavConfig(budgets=[k1, k2], r=r, m=m), extra_candidates=bm[:20])),
        ):
            snap, t0 = _snap(judge), time.perf_counter()
            ranked, tr = fn()
            sec = time.perf_counter() - t0
            d = _delta(judge, snap)
            rows[name].append({**_metrics([n.id for n in ranked], gold), "sec": sec + (t_bm if "bm25" in name and name != "decisionrag" else 0),
                               "judgments": d.get("judgments", tr.judgments), "tokens": d.get("tokens", 0), "stages": tr.sequential_stages})
    return {n: {"per_query": _collect(rows[n]), "summary": _summary(_collect(rows[n]))} for n in methods}


def _pool(ctx: Ctx, q: dict, n: int, rng: random.Random) -> list[Node]:
    """正解の段落 + BM25 の難しい負例 n-1 件(正解を除く)。順序は乱数でそろえる。"""
    negs = [p for p in ctx.retr.top_paras(q["query"], n + 5) if p != q["gold_para"]][:n - 1]
    ids = [q["gold_para"]] + negs
    rng.shuffle(ids)
    return [ctx.nodes[i] for i in ids]


def _hit_from_scores(nodes: list[Node], scores: list[float], gold_id: str) -> float:
    return float(nodes[top_indices(scores, 1)[0]].id == gold_id)


def exp_pool_scaling(ctx: Ctx, judge, qs: list[dict], ns=(5, 25, 100, 200), seed: int = 0, choice_limit: int = 255, m_after: int = 255) -> dict:
    out: dict[str, dict] = {}
    rng = random.Random(seed)
    for n in ns:
        modes: dict[str, list[float]] = {"binary_separate": [], "binary_joint": [], "choice": [], "binary_then_choice": []}
        for q in qs:
            pool = _pool(ctx, q, n, rng)
            gid = q["gold_para"]
            sep, _ = judge_pooled(judge, q["query"], pool, 1)
            modes["binary_separate"].append(_hit_from_scores(pool, sep, gid))
            joint, _ = judge_pooled(judge, q["query"], pool, None)
            modes["binary_joint"].append(_hit_from_scores(pool, joint, gid))
            if n <= choice_limit:
                modes["choice"].append(_hit_from_scores(pool, list(judge.choice(q["query"], pool)), gid))
            top = [pool[i] for i in top_indices(joint, min(m_after, n))]
            ch = list(judge.choice(q["query"], top))
            modes["binary_then_choice"].append(float(top[top_indices(ch, 1)[0]].id == gid))
        out[str(n)] = {"per_query": {k: v for k, v in modes.items() if v}, "summary": {k: sum(v) / len(v) for k, v in modes.items() if v}}
    return out


def exp_request_size(ctx: Ctx, judge, qs: list[dict], pool_n: int = 200, sizes=(1, 2, 5, 10, 25, 50, 100, 200), seed: int = 0) -> dict:
    rng = random.Random(seed)
    pools = [_pool(ctx, q, pool_n, rng) for q in qs]
    out: dict[str, dict] = {}
    for m in sizes:
        hits, hit10 = [], []
        for q, pool in zip(qs, pools):
            sc, _ = judge_pooled(judge, q["query"], pool, m)
            hits.append(_hit_from_scores(pool, sc, q["gold_para"]))
            hit10.append(float(q["gold_para"] in {pool[i].id for i in top_indices(sc, 10)}))
        out[str(m)] = {"per_query": {"hit1": hits, "r10": hit10}, "summary": {"hit1": sum(hits) / len(hits), "r10": sum(hit10) / len(hit10)}}
    ch = [float(pool[top_indices(list(judge.choice(q["query"], pool)), 1)[0]].id == q["gold_para"]) for q, pool in zip(qs, pools)]
    out["choice_all"] = {"per_query": {"hit1": ch}, "summary": {"hit1": sum(ch) / len(ch)}}
    return out


def exp_choice(ctx: Ctx, judge, qs: list[dict], ms=(0, 5, 20), k1: int = 3, k2: int = 2, r: int = 20) -> dict:
    out = {}
    for m in ms:
        rows = []
        for q in qs:
            ranked, _ = navigate(ctx.root, q["query"], judge, NavConfig(budgets=[k1, k2], r=r, m=m))
            rows.append(_metrics([n.id for n in ranked], {q["gold_para"]}))
        out[str(m)] = {"per_query": _collect(rows), "summary": _summary(_collect(rows))}
    return out


def exp_budget(ctx: Ctx, judge, qs: list[dict], k1s=(1, 2, 3, 5, 10, None), k2s=(1, 2, 3), r: int = 20, m: int = 5) -> dict:
    """k1(残すクラスタ数。None は全部 = 記事のタイトルを平らに全部採点)と、k2(残す記事数)の組み合わせ。"""
    out = {}
    for k1 in k1s:
        for k2 in k2s:
            kk1 = ctx.n_clusters if k1 is None else k1
            rows = []
            for q in qs:
                snap, t0 = _snap(judge), time.perf_counter()
                ranked, tr = navigate(ctx.root, q["query"], judge, NavConfig(budgets=[kk1, k2], r=r, m=m))
                d = _delta(judge, snap)
                rows.append({**_metrics([n.id for n in ranked], {q["gold_para"]}), "sec": time.perf_counter() - t0,
                             "judgments": d.get("judgments", tr.judgments), "tokens": d.get("tokens", 0)})
            out[f"k1={'all' if k1 is None else k1},k2={k2}"] = {"per_query": _collect(rows), "summary": _summary(_collect(rows))}
    return out


def exp_latency(ctx: Ctx, judge, sizes=(10, 25, 50, 100, 200, 300), repeat: int = 5, seed: int = 0, t_request: float = 0.2) -> dict:
    """1 請求の時間と、候補数の関係。式 4 の見積もり(t_req · ⌈log_b N⌉)の表も付ける。"""
    rng = random.Random(seed)
    qs = ctx.dev(repeat, seed)
    req = {}
    for n in sizes:
        ts = []
        for q in qs:
            pool = _pool(ctx, q, n, rng)
            t0 = time.perf_counter()
            judge.binary(q["query"], pool)
            ts.append(time.perf_counter() - t0)
        req[str(n)] = sorted(ts)[len(ts) // 2]
    analytic = {f"1e{e}": latency_estimate(10 ** e, 100, t_request) for e in range(2, 10)}
    return {"request_sec_median": req, "analytic_sec_b100": analytic, "t_request": t_request}


EXPERIMENTS = {"main": exp_main, "pool_scaling": exp_pool_scaling, "request_size": exp_request_size, "choice": exp_choice,
               "budget": exp_budget, "latency": exp_latency}
