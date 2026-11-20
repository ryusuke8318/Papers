"""全 alpha の PR@5/10 と診断量を、質問ごとに保存する。使い方: python run.py --dataset hotpot --tag eval"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import pipeline as pl
import grapher_core as gc
import grapher_metrics as gm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    S = pl.Split(a.dataset, a.tag)
    n = len(S.Q)
    res = {"dataset": a.dataset, "tag": a.tag, "n_queries": n, "n_passages": len(S.corpus["passages"]), "grid": pl.GRID, "modes": {}}
    base = {"pr5": [], "pr10": [], "cand_pr": [], "hub10": []}
    modes = {}
    for mode in S.ents:
        modes[mode] = {"deg_mean": [], "isolated_frac": [], "gold_connected": [], "gold_isolated": []}
        for meth in ("ppr", "gcs"):
            for al in pl.GRID:
                modes[mode][f"{meth}_{al}"] = {"pr5": [], "pr10": [], "hub10": [], "rank_gold_mean": []}
    for qi in range(n):
        gold = set(S.Q[qi]["gold"])
        cand, s, _, _ = S.candidates(qi)
        loc = {int(c): i for i, c in enumerate(cand)}
        base_rank = [int(c) for c in cand]
        base["pr5"].append(gm.perfect_recall_at_k(base_rank, gold, 5))
        base["pr10"].append(gm.perfect_recall_at_k(base_rank, gold, 10))
        base["cand_pr"].append(float(gold <= set(base_rank)))
        gl = [loc[g] for g in gold if g in loc]
        for mode in S.ents:
            A = S.graph(cand, mode)
            deg = gc.degrees(A)
            m = modes[mode]
            m["deg_mean"].append(float(deg.mean()))
            m["isolated_frac"].append(float((deg == 0).mean()))
            m["gold_isolated"].append(float(np.mean([deg[i] == 0 for i in gl])) if gl else 0.0)
            # 正解どうしが(候補内で)辺でつながっているか(2 件以上のとき)
            m["gold_connected"].append(float(all(((A[i] > 0) | (A[:, i] > 0))[[j for j in gl if j != i]].any() for i in gl)) if len(gl) == len(gold) and len(gl) > 1 else -1.0)
            for meth, fn in (("ppr", gc.ppr), ("gcs", gc.gcs)):
                for al in pl.GRID:
                    sc = fn(A, s, al)
                    order = np.argsort(-sc, kind="stable")
                    r = [int(cand[i]) for i in order]
                    d = m[f"{meth}_{al}"]
                    d["pr5"].append(gm.perfect_recall_at_k(r, gold, 5))
                    d["pr10"].append(gm.perfect_recall_at_k(r, gold, 10))
                    d["hub10"].append(gm.hub_share(order.tolist(), deg, 10))
                    pos = {int(c): k for k, c in enumerate(r)}
                    d["rank_gold_mean"].append(float(np.mean([pos[g] for g in gold if g in pos])) if any(g in pos for g in gold) else -1.0)
            if mode == "ner":
                base["hub10"].append(gm.hub_share(list(range(len(cand))), deg, 10))
        if qi % 500 == 0:
            print(a.dataset, a.tag, qi, f"{time.time()-t0:.0f}s", flush=True)
    res["baseline"] = base
    res["modes"] = modes
    res["seconds"] = time.time() - t0
    out = a.out or str(pl.HERE / "results" / f"{a.dataset}_{a.tag}.json")
    json.dump(res, open(out, "w"))
    print("done", out, f"{res['seconds']:.0f}s", "base pr10", np.mean(base["pr10"]), "cand_pr", np.mean(base["cand_pr"]))


if __name__ == "__main__":
    main()
