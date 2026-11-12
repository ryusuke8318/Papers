"""small-to-big 検索の設定を dev で選ぶ(results/retrieval_s2b.json)。
  python jp_vat/retrieval_s2b.py
検索は unit 単位の小さな chunk(small)で行い、当たった unit の親の範囲(項・条、通達は前後)まで広げて挿入する(big)。
比較のため、元の設定(unit のまま、上位 k 件)と同じ文字数になる点も出す。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import retrieval as R  # noqa: E402


def main():
    units = json.loads((HERE / "sources" / "corpus.json").read_text())
    dev = json.loads((HERE / "cases.json").read_text())["dev"]
    rows = R.sweep_s2b(units, dev, R.Dense())
    ok = [r for r in rows if r["ctx_chars"] <= 3000]
    best = max(ok, key=lambda r: (r["hit_all"], -r["ctx_chars"]))
    for r in sorted(rows, key=lambda r: (r["method"], r["level"], r["k"])):
        print(f'{r["method"]:6} {r["level"]:8} k={r["k"]} hit_all {r["hit_all"]:.3f} primary {r["hit_primary"]:.3f} ctx {r["ctx_chars"]:.0f}')
    print("best:", best)
    (HERE / "results" / "retrieval_s2b.json").write_text(json.dumps({"rows": rows, "best": best}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
