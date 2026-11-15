"""縮小再現の実験本体。1 データセット分の全手法を評価し、質問ごとの値を JSON に書く。

使い方: python run.py --dataset hotpot --n 500 --seed 0 --out results/hotpot.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import catrag_core as cc  # noqa: E402
import catrag_metrics as cm  # noqa: E402
import data as D  # noqa: E402

EMB_MODEL = "BAAI/bge-small-en-v1.5"
QPREFIX = "Represent this sentence for searching relevant passages: "


def embed(texts: list[str], cache: Path, query: bool = False) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB_MODEL, device="cuda")
    m.half()
    pre = QPREFIX if query else ""
    e = m.encode([pre + t for t in texts], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    e = e.astype(np.float32)
    np.save(cache, e)
    return e


class Prepared:
    def __init__(self, corpus: dict, tag: str):
        self.corpus = corpus
        P = corpus["passages"]
        key = f"{corpus['name']}_{tag}_{len(corpus['questions'])}"
        t0 = time.time()
        self.idx = cc.build_graph(P)
        self.t_graph = time.time() - t0
        D.WORK.mkdir(exist_ok=True)
        self.P_emb = embed([p["title"] + ": " + p["text"] for p in P], D.WORK / f"pemb_{key}.npy")
        sent_texts = [P[pi]["title"] + ": " + s for pi, p in enumerate(P) for s in p["sents"]]
        assert len(sent_texts) == len(self.idx.sent_pass)
        self.S_emb = embed(sent_texts, D.WORK / f"semb_{key}.npy")
        self.Q_emb = embed([q["question"] for q in corpus["questions"]], D.WORK / f"qemb_{key}.npy", query=True)
        self.bm25 = cc.BM25([p["title"] + " " + p["text"] for p in P])
        self.texts = [p["text"] for p in P]


def methods(base: cc.CatConfig) -> dict:
    base = replace(base, use_anchor=False, use_edge=False, use_key=False)   # 3 要素のスイッチは、ここで手法ごとに決める

    def c(**kw):
        return replace(base, **kw)
    return {
        "static_ppr": c(),
        "anchor": c(use_anchor=True),
        "edge": c(use_edge=True),
        "key": c(use_key=True),
        "anchor+edge": c(use_anchor=True, use_edge=True),
        "anchor+key": c(use_anchor=True, use_key=True),
        "edge+key": c(use_edge=True, use_key=True),
        "catrag": c(use_anchor=True, use_edge=True, use_key=True),
    }


def evaluate(prep: Prepared, base: cc.CatConfig, sa_cfg: dict | None, method_names: list[str] | None = None, limit: int | None = None, with_hub: bool = True, baselines: bool = True):
    qs = prep.corpus["questions"][:limit]
    ms = methods(base)
    if method_names:
        ms = {k: v for k, v in ms.items() if k in method_names}
    names = (["bm25", "dense"] if baselines else []) + (["sa_rag"] if sa_cfg is not None else []) + list(ms)
    out = {n: {k: [] for k in ("recall2", "recall5", "fcr2", "fcr5", "jsr5")} for n in names}
    hub = {n: {"s_ppr": [], "hub_mass_top": [], "hub_mass_all": []} for n in ms}
    times = {n: 0.0 for n in names}

    def record(name, ranked, q):
        g = set(q["gold"])
        o = out[name]
        o["recall2"].append(cm.recall_at_k(ranked, g, 2))
        o["recall5"].append(cm.recall_at_k(ranked, g, 5))
        o["fcr2"].append(cm.full_chain_at_k(ranked, g, 2))
        o["fcr5"].append(cm.full_chain_at_k(ranked, g, 5))
        o["jsr5"].append(cm.joint_proxy_at_k(ranked, g, 5, q["answer"], prep.texts))

    sent_sims = prep.Q_emb[: len(qs)] @ prep.S_emb.T
    dense_all = prep.Q_emb[: len(qs)] @ prep.P_emb.T
    for qi, q in enumerate(qs):
        dense = dense_all[qi]
        if baselines:
            t = time.time()
            record("bm25", np.argsort(-prep.bm25.scores(q["question"]), kind="stable")[:20].tolist(), q)
            times["bm25"] += time.time() - t
            record("dense", np.argsort(-dense, kind="stable")[:20].tolist(), q)
        ss = sent_sims[qi].astype(np.float64)
        if sa_cfg is not None:
            t = time.time()
            r, _ = cc.sa_retrieve(prep.idx, ss, dense.astype(np.float64), **sa_cfg)
            record("sa_rag", r[:20].tolist(), q)
            times["sa_rag"] += time.time() - t
        anchors = prep.idx.match_query_entities(q["question"])
        for name, cfg in ms.items():
            t = time.time()
            r, v = cc.retrieve(prep.idx, ss, dense.astype(np.float64), anchors, cfg)
            record(name, r[:20].tolist(), q)
            if with_hub:
                for k, x in cc.hub_stats(prep.idx, v).items():
                    hub[name][k].append(x)
            times[name] += time.time() - t
    return {"metrics": out, "hub": hub if with_hub else {}, "time_sec": times, "n": len(qs)}


def summarize(res: dict) -> dict:
    return {n: {k: float(np.mean(v)) for k, v in m.items()} for n, m in res["metrics"].items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpot")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--params", default=str(HERE / "results" / "tuned_params.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    params = json.loads(Path(a.params).read_text()) if Path(a.params).exists() else {}
    base = cc.CatConfig(**{k: (tuple(v) if k == "tier_ranks" else v) for k, v in params.get("cat", {}).items()})
    sa_cfg = params.get("sa", {})  # 空の dict は SA-RAG の既定値
    corpus = D.get_corpus(a.dataset, a.n, a.seed)
    t0 = time.time()
    prep = Prepared(corpus, "test")
    t_prep = time.time() - t0
    res = evaluate(prep, base, sa_cfg, limit=a.limit)
    res.update({"dataset": a.dataset, "seed": a.seed, "n_passages": len(corpus["passages"]), "n_entities": prep.idx.n_e,
                "n_edges": int(prep.idx.W.nnz), "t_prepare_sec": t_prep, "cat_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(base).items()},
                "sa_config": sa_cfg, "gold_n": [len(q["gold"]) for q in corpus["questions"][: res["n"]]]})
    print(json.dumps({"summary": summarize(res), "time_sec": res["time_sec"], "t_prepare_sec": t_prep, "n_entities": prep.idx.n_e}, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res))
