"""SA-RAG の再調整と再評価。

最初の調整(tune.py)では c を 0.3〜0.7、閾値 tau を 0 に固定していたため、ほぼ全段落が活性化して(拡散が飽和して)
結果が密検索と同じになった。SA-RAG の論文は活性化の閾値 tau_a = 0.5 を使っているので、(c, tau) の範囲を広げて
開発用の質問で選び直し、評価用の質問で SA-RAG だけ再評価して results/<dataset>.json の sa_rag を差し替える。
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402
import data as D  # noqa: E402
import catrag_core as cc  # noqa: E402

ds = sys.argv[1]
n_dev = 60
pp = HERE / "results" / f"tuned_params_{ds}.json"
params = json.loads(pp.read_text())
base = cc.CatConfig(**{k: (tuple(v) if k == "tier_ranks" else v) for k, v in params["cat"].items()})
test = D.get_corpus(ds, 500, 0)
dev = D.get_corpus(ds, 200, 1, tag="dev", skip={q["id"] for q in test["questions"]})
prep = R.Prepared(dev, "dev")
log, best = [], (-1, None)
for c, tau, g in itertools.product([0.5, 0.65, 0.8], [0.05, 0.3], [0.5, 1.0]):
    sa = {"c": c, "tau": tau, "gamma": g}
    res = R.evaluate(prep, base, sa, ["__none__"], with_hub=False, baselines=False, limit=n_dev)
    m = res["metrics"]["sa_rag"]
    s = float(np.mean(m["recall5"]) + np.mean(m["fcr5"]))
    log.append({**sa, "score": s, **{k: float(np.mean(v)) for k, v in m.items()}})
    print(sa, round(s, 4), flush=True)
    if s > best[0]:
        best = (s, sa)
params["sa_first"] = params["sa"]
params["sa"] = best[1]
params["sa_log"] = log
pp.write_text(json.dumps(params, indent=1))
prep = R.Prepared(test, "test")
res = R.evaluate(prep, base, best[1], ["__none__"], with_hub=False, baselines=False)
op = HERE / "results" / f"{ds}.json"
old = json.loads(op.read_text())
old["metrics"]["sa_rag"] = res["metrics"]["sa_rag"]
old["time_sec"]["sa_rag"] = res["time_sec"]["sa_rag"]
old["sa_config"] = best[1]
op.write_text(json.dumps(old))
print("done", best[1], {k: float(np.mean(v)) for k, v in res["metrics"]["sa_rag"].items()})
