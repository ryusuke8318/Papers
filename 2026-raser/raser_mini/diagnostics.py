"""結果の補助的な集計(1 回目の答えが「分からない」の割合、PRUNE の枝が残る割合、IRCOT* が 2 回以上回る割合、答えの長さ)。

  python raser_mini/diagnostics.py   # docs/diagnostics.md に書く
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from raser_core import f1_max, is_abstain, normalize_answer  # noqa: E402

lines = ["# 補助的な集計(diagnostics.py が生成)\n",
         "| LLM | データセット | 1 回目が「分からない」 | 1 回目の答えの語数(平均 / 中央値) | 正解を含む答え(one-shot / PRUNE / IRCOT*) | PRUNE で検証を通った枝がある | 提案された実体がある | IRCOT* が 2 回以上回る |",
         "|---|---|---|---|---|---|---|---|"]
for tag, name in (("q3b", "Qwen2.5-3B"), ("q7b", "Qwen2.5-7B")):
    for ds in ("musique", "2wiki", "hotpotqa"):
        rows = [json.loads(l) for l in open(HERE / "results" / f"{tag}_{ds}.jsonl", encoding="utf-8")]
        n = len(rows)
        L = [len(r["one_shot"]["answer"].split()) for r in rows]
        cont = [sum(normalize_answer(r["gold"]) in normalize_answer(r[k]["answer"]) for r in rows) / n for k in ("one_shot", "prune", "ircot")]
        lines.append(f"| {name} | {ds} | {sum(is_abstain(r['one_shot']['answer']) for r in rows) / n:.2f} | {st.mean(L):.1f} / {st.median(L):.0f} | "
                     + " / ".join(f"{c:.2f}" for c in cont)
                     + f" | {sum(bool(r['prune']['kept']) for r in rows) / n:.2f} | {sum(bool(r['prune']['bridges']) for r in rows) / n:.2f} | {sum(r['ircot']['rounds'] >= 2 for r in rows) / n:.2f} |")
(HERE.parent / "docs" / "diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
