"""検索設定の探索(論文 Tab. 1〜3 に相当)。dev 200 件で hit 率を測り、最良の設定を results/retrieval_sweep.json に保存する。

  python jp_vat/retrieval_sweep.py [--no-dense]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import retrieval as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dense", action="store_true")
    a = ap.parse_args()
    units = json.loads((HERE / "sources" / "corpus.json").read_text())
    dev = json.loads((HERE / "cases.json").read_text())["dev"]
    dense = None if a.no_dense else R.Dense()
    rows = R.sweep(units, dev, dense)
    best = R.pick_best(rows)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "retrieval_sweep.json").write_text(json.dumps({"rows": rows, "best": best}, ensure_ascii=False, indent=1))
    print("best:", best)
    for r in rows:
        print(f'{r["mode"]:9s}{r["method"]:6s}k={r["k"]} all={r["hit_all"]:.3f} primary={r["hit_primary"]:.3f} ctx={r["ctx_chars"]:.0f}')


if __name__ == "__main__":
    main()
