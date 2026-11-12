"""small-to-big の検索結果を sources/rag_contexts.json に "s2b::<事例 id>" として追加し、hit 率を results/s2b_hit.json に保存する。
dev で選んだ設定(results/retrieval_s2b.json の best)を使う。条件名は rag_s2b / rag_ft_s2b。
  python jp_vat/make_s2b_contexts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import retrieval as R  # noqa: E402


def main():
    best = json.loads((HERE / "results" / "retrieval_s2b.json").read_text())["best"]
    units = json.loads((HERE / "sources" / "corpus.json").read_text())
    data = json.loads((HERE / "cases.json").read_text())
    rag = json.loads((HERE / "sources" / "rag_contexts.json").read_text())
    r = R.Retriever(units, "unit", best["method"], R.Dense() if best["method"] == "dense" else None)
    hits, ctx = {}, {}
    for sp in ("test_seen", "test_exc_seen", "test_exc_unseen", "real", "dev"):
        h, n = [], []
        for c in data[sp]:
            cx = R.expand(units, r.topk(c["text"], best["k"]), best["level"])
            rag["s2b::" + c["id"]] = [x["text"] for x in cx]
            h.append(R.hit(cx, c["gold"]))
            n.append(sum(len(x["text"]) for x in cx))
        hits[sp], ctx[sp] = float(np.mean(h)), float(np.mean(n))
    (HERE / "sources" / "rag_contexts.json").write_text(json.dumps(rag, ensure_ascii=False))
    (HERE / "results" / "s2b_hit.json").write_text(json.dumps({"best": best, "hit_all": hits, "ctx_chars": ctx}, ensure_ascii=False, indent=1))
    print("設定:", {k: best[k] for k in ("level", "method", "k")})
    for sp in hits:
        print(f"  {sp:16} hit {hits[sp]:.3f}  ctx {ctx[sp]:.0f} 字")


if __name__ == "__main__":
    main()
