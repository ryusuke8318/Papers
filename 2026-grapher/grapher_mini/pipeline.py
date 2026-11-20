"""分割 1 つ分の準備(埋め込み、BM25、実体)と、質問ごとの候補・グラフの作成。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import ents as E  # noqa: E402
import grapher_core as gc  # noqa: E402

EMB_MODEL = "BAAI/bge-small-en-v1.5"
QPREFIX = "Represent this sentence for searching relevant passages: "
N_CAND = 200
RARE_DF = 20
GRID = [round(0.1 * i, 1) for i in range(1, 10)]


def embed(texts, cache: Path, query=False):
    if cache.exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB_MODEL, device="cuda")
    m.half()
    pre = QPREFIX if query else ""
    e = m.encode([pre + t for t in texts], batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.save(cache, e)
    return e


class Split:
    def __init__(self, name: str, tag: str):
        self.name, self.tag = name, tag
        self.corpus = D.get_split(name, tag, D.SPECS[name])
        P = self.corpus["passages"]
        self.Q = self.corpus["questions"]
        self.P_emb = embed([p["title"] + ": " + p["text"] for p in P], D.WORK / f"pemb_{name}_{tag}.npy")
        self.Q_emb = embed([q["question"] for q in self.Q], D.WORK / f"qemb_{name}_{tag}.npy", query=True)
        self.bm25 = gc.BM25([p["title"] + " " + p["text"] for p in P])
        ner = E.get_ents(name, tag, P)
        self.ents = {
            "ner": [frozenset(x) for x in ner],
            "ner+title": [frozenset(x) | {E.norm(p["title"])} for x, p in zip(ner, P)],
        }
        # 感度分析用: コーパス内の出現段落数が RARE_DF を超える実体(american、film など)を辺の根拠から外す
        from collections import Counter
        df = Counter(x for e in self.ents["ner"] for x in e)
        self.ents["ner_rare"] = [frozenset(x for x in e if df[x] <= RARE_DF) for e in self.ents["ner"]]

    def candidates(self, qi: int):
        q = self.Q[qi]
        cos = self.P_emb @ self.Q_emb[qi]
        bm = self.bm25.scores(q["question"])
        h = gc.hybrid_scores(cos, bm)
        cand = gc.top_candidates(h, N_CAND)
        return cand, h[cand], cos[cand], bm[cand]

    def graph(self, cand, mode="ner"):
        return gc.adjacency_from_entities([self.ents[mode][i] for i in cand])
