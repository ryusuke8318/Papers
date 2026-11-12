"""dev で選んだ検索設定(results/retrieval_sweep.json の best)で、全事例に挿入する chunk を固定し sources/rag_contexts.json に保存する。

GPU 評価と API 評価は同じ検索結果を使う。
  python jp_vat/make_rag_contexts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import retrieval as R  # noqa: E402


def main():
    best = json.loads((HERE / "results" / "retrieval_sweep.json").read_text())["best"]
    units = json.loads((HERE / "sources" / "corpus.json").read_text())
    data = json.loads((HERE / "cases.json").read_text())
    dense = R.Dense() if best["method"] == "dense" else None
    r = R.Retriever(units, best["mode"], best["method"], dense)
    out, hits = {}, {}
    for sp in ("test_seen", "test_exc_seen", "test_exc_unseen", "real", "dev"):
        h = []
        for c in data[sp]:
            ch = r.topk(c["text"], best["k"])
            out[c["id"]] = [x["text"] for x in ch]
            h.append(R.hit(ch, c["gold"]))
        hits[sp] = sum(h) / len(h)
    (HERE / "sources" / "rag_contexts.json").write_text(json.dumps(out, ensure_ascii=False))
    print("設定:", {k: best[k] for k in ("mode", "method", "k")}, "検索の hit 率:", {k: round(v, 3) for k, v in hits.items()})
    (HERE / "results" / "rag_hit.json").write_text(json.dumps({"best": best, "hit_all": hits}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
