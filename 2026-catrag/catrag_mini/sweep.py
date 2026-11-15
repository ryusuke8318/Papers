"""CatRAG の主要なハイパーパラメータ(eps、beta、n_seed/k_edge)を 1 つずつ動かし、静的 PPR との差を見る。
評価用の質問で行う感度分析であり、値の選択には使わない。"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402
import data as D  # noqa: E402
import catrag_core as cc  # noqa: E402
import catrag_metrics as cm  # noqa: E402

ds = sys.argv[1]
params = json.loads((HERE / "results" / f"tuned_params_{ds}.json").read_text())
base = cc.CatConfig(**{k: (tuple(v) if k == "tier_ranks" else v) for k, v in params["cat"].items()})
base = replace(base, use_anchor=False, use_edge=False, use_key=False)
prep = R.Prepared(D.get_corpus(ds, 500, 0), "test")
full = replace(base, use_anchor=True, use_edge=True, use_key=True)
grid = {"static": base}
for e in (0.05, 0.2, 0.5, 1.0):
    grid[f"eps={e}"] = replace(base, use_anchor=True, eps=e)
for b in (0.0, 1.0, 2.5, 5.0, 10.0):
    grid[f"beta={b}"] = replace(base, use_key=True, beta=b)
for ns, ke in ((5, 15), (5, 50), (10, 15), (10, 50)):
    grid[f"n_seed={ns},k_edge={ke}"] = replace(base, use_edge=True, n_seed=ns, k_edge=ke)
grid["catrag(eps=0.2,beta=2.5)"] = full
out = {}
ref = None
for name, cfg in grid.items():
    ms = R.methods(cfg)
    # methods() は 3 要素のスイッチを上書きするので、cfg の組み合わせを直接評価する
    res = {}
    for k in ("recall5", "fcr5"):
        res[k] = []
    qs = prep.corpus["questions"]
    ss_all = prep.Q_emb @ prep.S_emb.T
    d_all = prep.Q_emb @ prep.P_emb.T
    for qi, q in enumerate(qs):
        r, _ = cc.retrieve(prep.idx, ss_all[qi].astype(np.float64), d_all[qi].astype(np.float64), prep.idx.match_query_entities(q["question"]), cfg)
        g = set(q["gold"])
        res["recall5"].append(cm.recall_at_k(r[:20].tolist(), g, 5))
        res["fcr5"].append(cm.full_chain_at_k(r[:20].tolist(), g, 5))
    out[name] = res
    if name == "static":
        ref = res
    print(name, round(float(np.mean(res["recall5"])), 4), round(float(np.mean(res["fcr5"])), 4), flush=True)
(HERE / "results" / f"sweep_{ds}.json").write_text(json.dumps(out))
