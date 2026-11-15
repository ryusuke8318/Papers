"""results/eval_<dataset>.json から、表(docs/tables.md)と図(docs/*.svg)を作る。数値はすべて、質問単位のブートストラップ(10,000 回)の 95% 区間つき。

python make_report.py [--name eval]
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from efficientrag_core import bootstrap_ci, paired_bootstrap  # noqa: E402

DOCS = HERE.parent / "docs"
LABEL = {"direct": "Direct", "cot": "CoT", "direct_r10": "Direct-R@10", "direct_r20": "Direct-R@20", "direct_r30": "Direct-R@30",
         "decompose_r": "Decompose-R", "iter_retgen2": "Iter-RetGen iter2", "iter_retgen3": "Iter-RetGen iter3", "selfask": "Self-Ask",
         "efficientrag": "EfficientRAG", "efficientrag_no_filter": "EfficientRAG(Filter なし)", "efficientrag_no_tag": "EfficientRAG(tag なし)",
         "efficientrag_k3": "EfficientRAG(k=3)", "efficientrag_k10": "EfficientRAG(k=10)",
         "efficientrag_thr03": "EfficientRAG(閾値 0.3)", "efficientrag_thr07": "EfficientRAG(閾値 0.7)"}
LABEL_EN = {"direct_r10": "Direct-R@10", "direct_r20": "R@20", "direct_r30": "R@30", "decompose_r": "Decompose-R", "iter_retgen2": "Iter-RetGen2",
            "iter_retgen3": "Iter-RetGen3", "selfask": "Self-Ask", "efficientrag": "EfficientRAG"}
DS_NAME = {"hotpotqa": "HotpotQA", "2wiki": "2WikiMultihopQA", "musique": "MuSiQue"}


def name_of(m: str) -> str:
    if m in LABEL:
        return LABEL[m]
    if re.fullmatch(r"efficientrag_n\d+", m):
        return f"EfficientRAG(学習 {m[len('efficientrag_n'):]} 問)"
    return m


def col(res: list[dict], key: str) -> list[float]:
    return [r[key] for r in res]


def fmt(vals, scale=100.0, nd=1, seed=0):
    m, lo, hi = bootstrap_ci(vals, 10000, seed)
    return f"{m * scale:.{nd}f} [{lo * scale:.{nd}f}, {hi * scale:.{nd}f}]"


def fmt_d(a, b, scale=100.0, nd=1):
    m, lo, hi = paired_bootstrap(a, b, 10000, 0)
    return f"{m * scale:+.{nd}f} [{lo * scale:+.{nd}f}, {hi * scale:+.{nd}f}]", lo, hi


def verdict_diff(lo, hi, better_if_positive=True):
    """区間が 0 をまたがなければ、差あり。K のように小さいほうがよい指標は、better_if_positive=False にして、「少ない / 多い」と書く。"""
    if lo > 0:
        return "上回る" if better_if_positive else "多い"
    if hi < 0:
        return "下回る" if better_if_positive else "少ない"
    return "差なし"


def load(ds: str, name: str):
    p = HERE / "results" / f"{name}_{ds}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def tables(ds: str, d: dict) -> str:
    M = d["methods"]
    out = [f"### {DS_NAME[ds]}(評価 {d['n']} 問、検索コーパス {d['corpus']:,} 段落)\n"]
    out.append("検索(Recall@K と K = 候補プールの大きさ)\n")
    out.append("| 手法 | Recall (%) | K | LLM 呼び出し/問 |\n|---|---|---|---|")
    for m, res in M.items():
        if m in ("direct", "cot"):
            continue
        out.append(f"| {name_of(m)} | {fmt(col(res, 'recall'))} | {fmt(col(res, 'K'), 1, 2)} | {sum(col(res, 'llm_calls')) / len(res):.2f} |")
    out.append("\nQA(Qwen2.5-7B-Instruct が回答。Acc は同じ LLM による判定)\n")
    out.append("| 手法 | EM | F1 | Acc |\n|---|---|---|---|")
    for m, res in M.items():
        out.append(f"| {name_of(m)} | {fmt(col(res, 'em'))} | {fmt(col(res, 'f1'))} | {fmt(col(res, 'acc'))} |")
    return "\n".join(out) + "\n"


def claims(ds: str, d: dict) -> str:
    M = d["methods"]
    E = M.get("efficientrag")
    if not E:
        return ""
    rows = []
    for base in ("direct_r10", "direct_r20", "direct_r30", "iter_retgen2", "iter_retgen3", "selfask", "decompose_r"):
        if base not in M:
            continue
        b = M[base]
        dr, lo, hi = fmt_d(col(E, "recall"), col(b, "recall"))
        dk, klo, khi = fmt_d(col(E, "K"), col(b, "K"), 1, 2)
        rows.append(f"| {name_of(base)} | {dr} ({verdict_diff(lo, hi)}) | {dk} ({verdict_diff(klo, khi, False)}) |")
    t1 = "| 比較対象 | Recall の差 (pt) | K の差 | \n|---|---|---|\n" + "\n".join(rows)
    rows = []
    for base in ("direct", "cot", "direct_r10", "decompose_r", "iter_retgen2", "iter_retgen3", "selfask"):
        if base not in M:
            continue
        b = M[base]
        cells = []
        for k in ("em", "f1", "acc"):
            s, lo, hi = fmt_d(col(E, k), col(b, k))
            cells.append(f"{s} ({verdict_diff(lo, hi)})")
        rows.append(f"| {name_of(base)} | " + " | ".join(cells) + " |")
    t2 = "| 比較対象 | EM の差 (pt) | F1 の差 (pt) | Acc の差 (pt) |\n|---|---|---|---|\n" + "\n".join(rows)
    return (f"#### {DS_NAME[ds]}: EfficientRAG − 比較対象(同じ質問での差、95% 区間)\n\n{t1}\n\n{t2}\n")


def ablation(ds: str, d: dict) -> str:
    M = d["methods"]
    keys = [k for k in M if k.startswith("efficientrag")]
    if len(keys) < 2:
        return ""
    E = M["efficientrag"]
    rows = []
    for k in keys:
        if k == "efficientrag":
            continue
        b = M[k]
        r, rlo, rhi = fmt_d(col(b, "recall"), col(E, "recall"))
        kk, klo, khi = fmt_d(col(b, "K"), col(E, "K"), 1, 2)
        a, alo, ahi = fmt_d(col(b, "acc"), col(E, "acc"))
        rows.append(f"| {name_of(k)} | {r} ({verdict_diff(rlo, rhi)}) | {kk} | {a} ({verdict_diff(alo, ahi)}) |")
    return (f"#### {DS_NAME[ds]}: 各条件 − EfficientRAG(標準設定)\n\n| 条件 | Recall の差 (pt) | K の差 | Acc の差 (pt) |\n|---|---|---|---|\n" + "\n".join(rows) + "\n")


def latency(d: dict) -> str:
    L = d.get("latency")
    if not L:
        return ""
    import statistics
    rows = ["| 手法 | LLM 呼び出し/問 | 反復数 | 遅延 平均 (s) | 95% 区間 | Direct-R@10 比 |", "|---|---|---|---|---|---|"]
    base = L.get("direct_r10", {}).get("sec")
    for m, v in L.items():
        mean, lo, hi = bootstrap_ci(v["sec"], 10000, 0)
        ratio = f"{mean / (sum(base) / len(base)):.2f}" if base else ""
        it = statistics.mean(v["iters"]) if any(v["iters"]) else None
        rows.append(f"| {LABEL.get(m, m) if m != 'direct' else 'Direct'} | {statistics.mean(v['llm_calls']):.2f} | {'' if it is None else f'{it:.2f}'} | {mean:.2f} | [{lo:.2f}, {hi:.2f}] | {ratio} |")
    if "efficientrag" in L:
        e = L["efficientrag"]["sec"]
        for m in ("iter_retgen3", "selfask"):
            if m in L:
                d_, lo, hi = paired_bootstrap([x for x in L[m]["sec"]], e, 10000, 0)
                rows.append(f"\n{LABEL[m]} − EfficientRAG の遅延差: {d_:+.2f} s [{lo:+.2f}, {hi:+.2f}](EfficientRAG は {100 * (1 - sum(e) / sum(L[m]['sec'])):.0f}% 短い)")
    return "\n".join(rows) + "\n"


def svg_recall_k(datasets: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(datasets)
    fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
    for ax, (ds, d) in zip(axs[0], datasets.items()):
        for m, res in d["methods"].items():
            if m in ("direct", "cot") or m.startswith("efficientrag_"):
                continue
            km, klo, khi = bootstrap_ci(col(res, "K"), 2000)
            rm, rlo, rhi = bootstrap_ci(col(res, "recall"), 2000)
            er = m.startswith("efficientrag")
            ax.errorbar(km, rm * 100, xerr=[[km - klo], [khi - km]], yerr=[[(rm - rlo) * 100], [(rhi - rm) * 100]], fmt="o" if not er else "*",
                        ms=6 if not er else 11, color="#c0392b" if er else "#34495e", capsize=2, lw=1)
            ax.annotate(LABEL_EN.get(m, m), (km, rm * 100), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_title(DS_NAME[ds])
        ax.set_xlabel("K (chunks in candidate pool)")
        ax.set_ylabel("Recall (%)")
        ax.grid(alpha=.3)
    fig.tight_layout()
    p = DOCS / "recall_vs_k.svg"
    fig.savefig(p, format="svg")
    return str(p)


def svg_scale(d: dict) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = []
    for m, res in d["methods"].items():
        if re.fullmatch(r"efficientrag_n\d+", m):
            pts.append((int(m[len("efficientrag_n"):]), res))
    base = d["methods"].get("efficientrag")
    if base and pts:
        fit = HERE / "results" / "fit_hotpotqa.json"
        full = [int(k[1:]) for k in json.loads(fit.read_text()) if re.fullmatch(r"n\d+", k)] if fit.exists() else []
        if full:
            pts.append((max(full), base))
    if not pts:
        return None
    pts.sort(key=lambda x: x[0])
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, key, ttl in zip(axs, ("recall", "K", "acc"), ("Recall (%)", "K", "Acc (%)")):
        xs, ys, lo, hi = [], [], [], []
        for n, res in pts:
            m, l, h = bootstrap_ci(col(res, key), 2000)
            s = 1 if key == "K" else 100
            xs.append(n); ys.append(m * s); lo.append((m - l) * s); hi.append((h - m) * s)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", capsize=3, color="#2c3e50")
        ax.set_xscale("log")
        ax.set_xlabel("training questions for Labeler / Filter")
        ax.set_ylabel(ttl)
        ax.grid(alpha=.3)
    fig.tight_layout()
    p = DOCS / "data_scale.svg"
    fig.savefig(p, format="svg")
    return str(p)


def main(name: str = "eval"):
    ds_data = {ds: load(ds, name) for ds in DS_NAME}
    ds_data = {k: v for k, v in ds_data.items() if v}
    parts = ["# 結果の表(自動生成。`python efficientrag_mini/make_report.py`)\n", "[a, b] は質問単位のブートストラップ(10,000 回)の 95% 区間。差は同じ質問での対応のある差。\n"]
    for ds, d in ds_data.items():
        parts += [tables(ds, d), claims(ds, d), ablation(ds, d)]
        if ds == "hotpotqa":
            parts += ["#### 効率(質問 1 件ずつ処理したときの遅延)\n", latency(d)]
    (DOCS / "tables.md").write_text("\n".join(p for p in parts if p), encoding="utf-8")
    svg_recall_k(ds_data)
    if "hotpotqa" in ds_data:
        svg_scale(ds_data["hotpotqa"])
    print("done", list(ds_data))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else "eval")
