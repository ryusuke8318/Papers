"""検索の改善案を同じ土俵で比べる(dev で設定を選び、test と real で報告)。

手法
  dense / bm25      元の設定(unit 単位)
  hybrid            BM25 と dense の順位を RRF で統合
  graph             当たった unit が参照する unit(令・法・通達の引用)と、参照してくる unit を足す
  rewrite           LLM が事例を「取引の種類・場所・当事者」に書き換えたクエリで検索(rewrite.json を Modal で作る)
  rerank            hybrid の上位 20 件を LLM が並べ替える(rerank.json を Modal で作る)
  ft                bge-m3 を学習用 800 件で追加学習(ft_scores.json を Modal で作る)
採点は「正解の unit が全部、挿入した文脈に入ったか」(hit_all)。挿入する文字数(ctx)も一緒に出す。
  python jp_vat/retrieval_exp.py analyze      # 取りこぼしの内訳
  python jp_vat/retrieval_exp.py base         # dense / bm25 / hybrid / graph
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import retrieval as R  # noqa: E402

EVAL = ("dev", "test_seen", "test_exc_seen", "test_exc_unseen", "real")
CACHE = Path("/tmp/claude-1000/-mnt-e-playground-Papers/bbc9f06c-b1c4-4a39-b674-4c01719cbbd0/scratchpad/vat_emb")


def load():
    units = json.loads((HERE / "sources" / "corpus.json").read_text())
    data = json.loads((HERE / "cases.json").read_text())
    return units, data


def unit_text(u):
    return f"【{u['cite']}】{u['text']}"


# ---------------------------------------------------------------- スコア行列(事例 × unit)
def score_mats(units, cases, queries=None, tag="orig", dense=None):
    """queries: 事例ごとのクエリ(None なら事例の本文)。返り値は (dense, bm25) の行列。"""
    CACHE.mkdir(exist_ok=True, parents=True)
    qs = queries or [c["text"] for c in cases]
    texts = [unit_text(u) for u in units]
    bm = R.BM25(texts)
    sb = np.stack([bm.scores(q) for q in qs])
    f = CACHE / f"{tag}_{len(qs)}.npy"
    fu = CACHE / "units.npy"
    if f.exists() and fu.exists():
        sd = np.load(f) @ np.load(fu).T
    else:
        dense = dense or R.Dense()
        if not fu.exists():
            np.save(fu, dense.encode(texts))
        qe = dense.encode(qs)
        np.save(f, qe)
        sd = qe @ np.load(fu).T
    return sd, sb


def rrf(*mats, k=60):
    out = np.zeros_like(mats[0])
    for m in mats:
        rank = np.argsort(np.argsort(-m, axis=1), axis=1)
        out += 1.0 / (k + rank + 1)
    return out


# ---------------------------------------------------------------- 参照グラフ
def build_graph(units):
    ids = {u["id"] for u in units}
    edges = {i: set() for i in ids}

    def target(kind, art, para, goh=None):
        for cand in ([f"{kind}-{art}-{para}-{goh}"] if goh else []) + [f"{kind}-{art}-{para}"]:
            if cand in ids:
                return cand
        return None

    for u in units:
        t = u["text"].replace("－", "-")
        found = set()
        for m in re.finditer(r"(法|令)第(\d+)条第(\d+)項(?:第(\d+)号(?:から第(\d+)号まで)?)?", t):
            kind, art, para, a, b = m.groups()
            if a and b:
                for g in range(int(a), int(b) + 1):
                    x = target(kind, art, para, g)
                    if x:
                        found.add(x)
            else:
                x = target(kind, art, para, a)
                if x:
                    found.add(x)
        for m in re.finditer(r"5-7-(\d+)(?:の(\d+))?", t):
            x = f"通達-5-7-{m.group(1)}" + (f"の{m.group(2)}" if m.group(2) else "")
            if x in ids:
                found.add(x)
        found.discard(u["id"])
        for x in found:
            edges[u["id"]].add(x)
            edges[x].add(u["id"])  # 逆向き(参照してくる unit)も辿る
    return edges


# ---------------------------------------------------------------- 文脈の作り方と採点
def context(units, ranked_ids, k, graph=None, hop_cap=None, max_chars=None):
    by_id = {u["id"]: u for u in units}
    out = list(ranked_ids[:k])
    if graph is not None:
        extra = []
        for i in ranked_ids[:k]:
            for j in sorted(graph.get(i, ())):
                if j not in out and j not in extra:
                    extra.append(j)
        out += extra[:hop_cap] if hop_cap is not None else extra
    if max_chars is not None:
        keep, tot = [], 0
        for i in out:
            n = len(unit_text(by_id[i]))
            if tot + n <= max_chars:
                keep.append(i)
                tot += n
        out = keep
    return out


def evaluate(units, data, rankings, k, **kw):
    """rankings: {split: [ranked unit id list per case]}。split ごとの hit_all、挿入文字数。"""
    by_id = {u["id"]: u for u in units}
    res = {}
    for sp in EVAL:
        hs, ns = [], []
        for c, rk in zip(data[sp], rankings[sp]):
            cx = context(units, rk, k, **kw)
            hs.append(all(g in cx for g in c["gold"]))
            ns.append(sum(len(unit_text(by_id[i])) for i in cx))
        res[sp] = {"hit": float(np.mean(hs)), "ctx": float(np.mean(ns))}
    return res


def rankings_from(units, mats_by_split, f):
    ids = [u["id"] for u in units]
    return {sp: [[ids[j] for j in np.argsort(-f(m)[i], kind="stable")] for i in range(f(m).shape[0])] for sp, m in mats_by_split.items()}


def split_mats(units, data, queries=None, tag="orig"):
    allc = [c for sp in EVAL for c in data[sp]]
    qs = None if queries is None else [queries[c["id"]] for c in allc]
    sd, sb = score_mats(units, allc, qs, tag)
    out, pos = {}, 0
    for sp in EVAL:
        n = len(data[sp])
        out[sp] = (sd[pos:pos + n], sb[pos:pos + n])
        pos += n
    return out


def fmt(res):
    return "  ".join(f"{sp[:9]} {res[sp]['hit']:.2f}" for sp in EVAL) + f"  | ctx dev {res['dev']['ctx']:.0f}"


def analyze():
    units, data = load()
    sm = split_mats(units, data)
    ids = [u["id"] for u in units]
    rk = rankings_from(units, {s: m[0] for s, m in sm.items()}, lambda m: m)
    miss = {}
    for sp in EVAL:
        tot = {}
        for c, r in zip(data[sp], rk[sp]):
            top = r[:8]
            for g in c["gold"]:
                a = tot.setdefault(g, [0, 0])
                a[1] += 1
                a[0] += g in top
        miss[sp] = tot
    for sp in EVAL:
        bad = sorted(((v[1] - v[0], g, v[1]) for g, v in miss[sp].items() if v[1] > v[0]), reverse=True)[:8]
        print(sp, "取りこぼし(件数, unit, 出現):", bad)
    kinds = {}
    for sp in EVAL:
        for c, r in zip(data[sp], rk[sp]):
            for g in c["gold"]:
                k = g.split("-")[0]
                a = kinds.setdefault((sp, k), [0, 0])
                a[1] += 1
                a[0] += g in r[:8]
    for key in sorted(kinds):
        print(key, f"{kinds[key][0]}/{kinds[key][1]}")
    g = build_graph(units)
    print("グラフ: 辺", sum(len(v) for v in g.values()) // 2, "孤立 unit", sum(1 for v in g.values() if not v))


def base():
    units, data = load()
    sm = split_mats(units, data)
    g = build_graph(units)
    D = {s: m[0] for s, m in sm.items()}
    B = {s: m[1] for s, m in sm.items()}
    H = {s: rrf(sm[s][0], sm[s][1]) for s in sm}
    rows = {}
    for name, M in (("dense", D), ("bm25", B), ("hybrid", H)):
        rk = rankings_from(units, M, lambda m: m)
        for k in (3, 5, 8):
            rows[f"{name} k={k}"] = evaluate(units, data, rk, k)
        if name != "bm25":
            for k in (3, 5):
                for cap in (3, 6):
                    rows[f"{name}+graph k={k} cap={cap}"] = evaluate(units, data, rk, k, graph=g, hop_cap=cap)
    for n, r in rows.items():
        print(f"{n:28}", fmt(r))
    (HERE / "results" / "retrieval_exp_base.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


def _variants(units, data):
    """クエリの作り方ごとのスコア行列。orig=事例の本文、rw=書き換えのみ、cat=本文+書き換え。"""
    rw = json.loads((HERE / "results" / "rewrite.json").read_text())
    allc = [c for sp in EVAL for c in data[sp]]
    out = {"orig": split_mats(units, data)}
    out["rw"] = split_mats(units, data, rw, "rw")
    out["cat"] = split_mats(units, data, {c["id"]: c["text"] + "\n" + rw[c["id"]] for c in allc}, "cat")
    return out


def rewrite():
    units, data = load()
    g = build_graph(units)
    rows = {}
    for q, sm in _variants(units, data).items():
        H = {s: rrf(m[0], m[1]) for s, m in sm.items()}
        rk = rankings_from(units, H, lambda m: m)
        for k in (5, 8):
            rows[f"hybrid[{q}] k={k}"] = evaluate(units, data, rk, k)
        rows[f"hybrid[{q}]+graph k=5 cap=6"] = evaluate(units, data, rk, 5, graph=g, hop_cap=6)
    for n, r in rows.items():
        print(f"{n:30}", fmt(r))
    (HERE / "results" / "retrieval_exp_rewrite.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


def cands(first="orig", n=20):
    """rerank に渡す候補(ハイブリッドの上位 n 件)を results/rerank_in.json に書く。"""
    units, data = load()
    sm = _variants(units, data)[first]
    rk = rankings_from(units, {s: rrf(m[0], m[1]) for s, m in sm.items()}, lambda m: m)
    out = {c["id"]: r[:n] for sp in EVAL for c, r in zip(data[sp], rk[sp])}
    (HERE / "results" / "rerank_in.json").write_text(json.dumps(out, ensure_ascii=False))
    hits = {sp: float(np.mean([all(g in r[:n] for g in c["gold"]) for c, r in zip(data[sp], rk[sp])])) for sp in EVAL}
    print("候補", n, "件に正解が全部入る率:", {k: round(v, 2) for k, v in hits.items()})


def rerank_eval():
    units, data = load()
    g = build_graph(units)
    rr = json.loads((HERE / "results" / "rerank.json").read_text())
    rk = {sp: [rr[c["id"]] for c in data[sp]] for sp in EVAL}
    rows = {}
    for k in (3, 5, 8):
        rows[f"rerank k={k}"] = evaluate(units, data, rk, k)
    rows["rerank+graph k=3 cap=6"] = evaluate(units, data, rk, 3, graph=g, hop_cap=6)
    rows["rerank+graph k=5 cap=6"] = evaluate(units, data, rk, 5, graph=g, hop_cap=6)
    for n, r in rows.items():
        print(f"{n:30}", fmt(r))
    (HERE / "results" / "retrieval_exp_rerank.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


def ft_eval():
    """追加学習した埋め込み(3 seed)。単独の dense、BM25 との RRF、さらに graph。seed の平均を出す。"""
    units, data = load()
    g = build_graph(units)
    sm = split_mats(units, data)
    B = {s: m[1] for s, m in sm.items()}
    files = sorted((HERE / "results").glob("ft_scores_embft_s*.json"))
    allrows = []
    for f in files:
        sc = json.loads(f.read_text())["scores"]
        D = {sp: np.array(sc[sp]) for sp in EVAL}
        rows = {}
        for name, M in (("ft", D), ("ft+bm25", {s: rrf(D[s], B[s]) for s in D})):
            rk = rankings_from(units, M, lambda m: m)
            for k in (5, 8):
                rows[f"{name} k={k}"] = evaluate(units, data, rk, k)
            rows[f"{name}+graph k=5 cap=6"] = evaluate(units, data, rk, 5, graph=g, hop_cap=6)
        allrows.append(rows)
    mean = {}
    for n in allrows[0]:
        mean[n] = {sp: {"hit": float(np.mean([r[n][sp]["hit"] for r in allrows])), "ctx": allrows[0][n][sp]["ctx"]} for sp in EVAL}
        print(f"{n:24}", fmt(mean[n]))
    print("seed 数:", len(allrows))
    (HERE / "results" / "retrieval_exp_ft.json").write_text(json.dumps(mean, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"analyze": analyze, "base": base, "rewrite": rewrite, "cands": cands, "rerank": rerank_eval, "ft": ft_eval}[sys.argv[1]]()
