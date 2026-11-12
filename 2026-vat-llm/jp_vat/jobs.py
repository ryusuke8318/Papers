"""日本の消費税 内外判定の検証で回す条件(GPU)と API モデルの一覧。torch は読み込まない。"""
from __future__ import annotations

BASE = dict(train=False, lr=2e-4, batch=16, epochs=3, rank=32, seed=0)
MODELS = {"q15": "Qwen/Qwen2.5-1.5B-Instruct", "q3": "Qwen/Qwen2.5-3B-Instruct", "q7": "Qwen/Qwen2.5-7B-Instruct"}

JOBS: dict[str, dict] = {}
for tag, m in MODELS.items():
    JOBS[f"{tag}_base"] = {**BASE, "name": f"{tag}_base", "model": m, "conds": ["plain", "rag", "mocked", "long"]}
    for s in (0, 1, 2):
        JOBS[f"{tag}_ft_s{s}"] = {**BASE, "name": f"{tag}_ft_s{s}", "model": m, "train": True, "seed": s,
                                  "conds": ["ft", "rag_ft", "mocked_ft"]}
for t, m in MODELS.items():
    if t != "q7":
        JOBS[f"{t}_s2b"] = {**BASE, "name": f"{t}_s2b", "model": m, "conds": ["rag_s2b"]}
for t in ("q15", "q3"):  # 改善した検索(hg: ハイブリッド+参照グラフ、rrg: LLM rerank+参照グラフ)
    JOBS[f"{t}_ctx"] = {**BASE, "name": f"{t}_ctx", "model": MODELS[t], "conds": ["rag_hg", "rag_rrg"]}
    for s in (0, 1, 2):
        JOBS[f"{t}_ft_s{s}_ctx"] = {**BASE, "name": f"{t}_ft_s{s}_ctx", "model": MODELS[t], "train": True, "seed": s,
                                    "conds": ["rag_ft_hg", "rag_ft_rrg"]}
CTX_ORDER = [f"{t}_ctx" for t in ("q15", "q3")] + [f"{t}_ft_s{s}_ctx" for s in (0, 1, 2) for t in ("q15", "q3")]
JOBS["smoke_base"] = {**BASE, "name": "smoke_base", "model": MODELS["q15"], "conds": ["plain", "rag"], "limit": 4}
JOBS["smoke_ft"] = {**BASE, "name": "smoke_ft", "model": MODELS["q15"], "train": True, "epochs": 1, "limit": 4,
                    "conds": ["ft", "rag_ft"], "n_train": 32}

MAIN_ORDER = ["q15_base", "q3_base"] + [f"{t}_ft_s{s}" for s in (0, 1, 2) for t in ("q15", "q3")]
API_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
API_CONDS = ["plain", "rag", "mocked", "long"]
