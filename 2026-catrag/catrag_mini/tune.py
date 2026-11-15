"""開発用の質問(評価用とは別の抽出)で、少数のハイパーパラメータを選ぶ。選んだ値は results/tuned_params_<dataset>.json に保存する。

選ぶ対象は、静的 PPR の (lam, k_fact)、CatRAG の (tier_ranks, r_key)、SA-RAG の (c, gamma) のみ。
論文の値をそのまま使うもの(eps=0.2、beta=2.5、n_seed=5、k_edge=15、d=0.5)は動かさない。
選択基準は dev の Recall@5 + FCR@5(質問ごとの値の平均)。
"""
import argparse
import itertools
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402
import data as D  # noqa: E402
import catrag_core as cc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="hotpot")
ap.add_argument("--n_dev", type=int, default=200)
ap.add_argument("--n_test", type=int, default=500)
a = ap.parse_args()

test = D.get_corpus(a.dataset, a.n_test, 0)
corpus = D.get_corpus(a.dataset, a.n_dev, 1, tag="dev", skip={q["id"] for q in test["questions"]})
prep = R.Prepared(corpus, "dev")
log = []


def score(cfg, name):
    res = R.evaluate(prep, cfg, None, [name], with_hub=False, baselines=False)
    m = res["metrics"][name]
    return float(np.mean(m["recall5"]) + np.mean(m["fcr5"])), {k: float(np.mean(v)) for k, v in m.items()}


base = cc.CatConfig()
best = (-1, None)
for lam, kf in itertools.product([0.1, 0.3, 1.0, 3.0], [5, 10, 20]):
    cfg = replace(base, lam=lam, k_fact=kf)
    s, m = score(cfg, "static_ppr")
    log.append({"stage": "static", "lam": lam, "k_fact": kf, "score": s, **m})
    print("static", lam, kf, round(s, 4), flush=True)
    if s > best[0]:
        best = (s, cfg)
base = best[1]

best2 = (-1, None)
for tr, rk in itertools.product([(5, 50, 500), (10, 100, 1000), (20, 200, 2000), (50, 500, 5000)], [10, 20, 50]):
    cfg = replace(base, use_anchor=True, use_edge=True, use_key=True, tier_ranks=tr, r_key=rk)
    s, m = score(cfg, "catrag")
    log.append({"stage": "catrag", "tier_ranks": list(tr), "r_key": rk, "score": s, **m})
    print("catrag", tr, rk, round(s, 4), flush=True)
    if s > best2[0]:
        best2 = (s, cfg)
cat = best2[1]

best3 = (-1, None)
for c, g in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7], [0.5, 1.0, 2.0]):
    sa = {"c": c, "gamma": g}
    res = R.evaluate(prep, base, sa, [], with_hub=False, baselines=False, limit=100)
    m = res["metrics"]["sa_rag"]
    s = float(np.mean(m["recall5"]) + np.mean(m["fcr5"]))
    log.append({"stage": "sa", "c": c, "gamma": g, "score": s, **{k: float(np.mean(v)) for k, v in m.items()}})
    print("sa", c, g, round(s, 4), flush=True)
    if s > best3[0]:
        best3 = (s, sa)

out = {"cat": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cat).items()}, "sa": best3[1], "dev_n": len(corpus["questions"]), "log": log}
(HERE / "results" / f"tuned_params_{a.dataset}.json").write_text(json.dumps(out, indent=1))
print("selected", out["cat"], out["sa"])
