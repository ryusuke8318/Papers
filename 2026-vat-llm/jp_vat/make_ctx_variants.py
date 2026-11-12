"""改善した検索(ハイブリッド + 参照グラフ、LLM rerank + 参照グラフ)の文脈を sources/rag_contexts.json に追加する。
キーは "hg::<事例 id>" と "rrg::<事例 id>"。条件名は rag_hg / rag_ft_hg / rag_rrg / rag_ft_rrg。hit 率は results/ctx_variants_hit.json。
  python jp_vat/retrieval_exp.py cands && modal run jp_vat/modal_app.py --stage rerank   # rrg に必要
  python jp_vat/make_ctx_variants.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import retrieval_exp as X  # noqa: E402


def main():
    units, data = X.load()
    by_id = {u["id"]: u for u in units}
    g = X.build_graph(units)
    sm = X.split_mats(units, data)
    hy = X.rankings_from(units, {s: X.rrf(m[0], m[1]) for s, m in sm.items()}, lambda m: m)
    rr = json.loads((HERE / "results" / "rerank.json").read_text())
    rag = json.loads((HERE / "sources" / "rag_contexts.json").read_text())
    rep = {}
    for key, k in (("hg", 5), ("rrg", 5)):
        for sp in X.EVAL:
            h, n = [], []
            for i, c in enumerate(data[sp]):
                ranked = hy[sp][i] if key == "hg" else rr[c["id"]]
                cx = X.context(units, ranked, k, graph=g, hop_cap=6)
                rag[f"{key}::{c['id']}"] = [X.unit_text(by_id[u]) for u in cx]
                h.append(all(x in cx for x in c["gold"]))
                n.append(sum(len(X.unit_text(by_id[u])) for u in cx))
            rep.setdefault(key, {})[sp] = {"hit": float(np.mean(h)), "ctx": float(np.mean(n))}
        print(key, {sp: (round(v["hit"], 2), round(v["ctx"])) for sp, v in rep[key].items()})
    (HERE / "sources" / "rag_contexts.json").write_text(json.dumps(rag, ensure_ascii=False))
    (HERE / "results" / "ctx_variants_hit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
