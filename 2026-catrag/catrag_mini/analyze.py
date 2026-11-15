"""results/<dataset>.json から、docs/tables.md(表)と docs/*.svg(図)を作る。数値はすべて結果ファイルから引く。"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import catrag_metrics as cm  # noqa: E402

DS = [("hotpot", "HotpotQA"), ("2wiki", "2WikiMultiHopQA"), ("musique", "MuSiQue")]
METRICS = [("recall2", "R@2"), ("recall5", "R@5"), ("fcr2", "FCR@2"), ("fcr5", "FCR@5"), ("jsr5", "JSR*@5")]
MAIN = ["bm25", "dense", "sa_rag", "static_ppr", "catrag"]
ABL = ["static_ppr", "anchor", "edge", "key", "anchor+edge", "anchor+key", "edge+key", "catrag"]


def load():
    out = {}
    for k, _ in DS:
        p = HERE / "results" / f"{k}.json"
        if p.exists():
            out[k] = json.loads(p.read_text())
    return out


def fmt(x, lo=None, hi=None):
    s = f"{100 * x:.1f}"
    return s if lo is None else f"{s} [{100 * lo:.1f}, {100 * hi:.1f}]"


def main():
    R = load()
    L = []
    for k, name in DS:
        if k not in R:
            continue
        r = R[k]
        m = r["metrics"]
        L.append(f"## {name}(質問 {r['n']} 件、段落 {r['n_passages']} 件)\n")
        L.append("### 手法の比較(平均 [95% ブートストラップ区間]、単位 %)\n")
        L.append("| 手法 | " + " | ".join(n for _, n in METRICS) + " |")
        L.append("|---|" + "---|" * len(METRICS))
        for meth in MAIN:
            cells = []
            for mk, _ in METRICS:
                mean, lo, hi = cm.bootstrap_ci(np.array(m[meth][mk]), 5000, 0)
                cells.append(fmt(mean, lo, hi))
            L.append(f"| {meth} | " + " | ".join(cells) + " |")
        L.append("\n### アブレーション(静的 PPR からの差、対応ありブートストラップ、単位 % ポイント。*は区間が 0 をまたがない)\n")
        L.append("| 構成 | " + " | ".join(n for _, n in METRICS) + " |")
        L.append("|---|" + "---|" * len(METRICS))
        for meth in ABL:
            cells = []
            for mk, _ in METRICS:
                if meth == "static_ppr":
                    cells.append(fmt(np.mean(m[meth][mk])))
                else:
                    d = cm.paired_diff_ci(np.array(m[meth][mk]), np.array(m["static_ppr"][mk]), 5000, 0)
                    cells.append(f"{100*d['diff']:+.1f} [{100*d['lo']:+.1f}, {100*d['hi']:+.1f}]" + ("*" if d["significant"] else ""))
            L.append(f"| {meth} | " + " | ".join(cells) + " |")
        L.append("\n(static_ppr の行は絶対値、他の行は static_ppr との差)\n")
        L.append("### 1 つ外したときの差(catrag − 外した構成、% ポイント。論文の表 5 に対応。*は区間が 0 をまたがない)\n")
        L.append("| 外す部品 | R@5 | FCR@5 |")
        L.append("|---|---|---|")
        for lab, other in [("Symbolic Anchoring", "edge+key"), ("Dynamic Edge Weighting", "anchor+key"), ("Key-Fact Enhancement", "anchor+edge")]:
            cells = []
            for mk in ("recall5", "fcr5"):
                d = cm.paired_diff_ci(np.array(m["catrag"][mk]), np.array(m[other][mk]), 5000, 0)
                cells.append(f"{100*d['diff']:+.1f} [{100*d['lo']:+.1f}, {100*d['hi']:+.1f}]" + ("*" if d["significant"] else ""))
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("### ハブへの質量(上位 10 語句。平均)\n")
        L.append("| 手法 | S_ppr | スーパーハブの質量(上位10語句内) | スーパーハブの質量(全語句) |")
        L.append("|---|---|---|---|")
        for meth in ["static_ppr", "anchor", "edge", "key", "catrag"]:
            h = r["hub"][meth]
            L.append(f"| {meth} | {np.mean(h['s_ppr']):.1f} | {100*np.mean(h['hub_mass_top']):.1f}% | {100*np.mean(h['hub_mass_all']):.1f}% |")
        for kk in ["s_ppr", "hub_mass_top", "hub_mass_all"]:
            d = cm.paired_diff_ci(np.array(r["hub"]["catrag"][kk]), np.array(r["hub"]["static_ppr"][kk]), 5000, 0)
            L.append(f"\n- catrag - static_ppr の {kk}: {d['diff']:.4g} [{d['lo']:.4g}, {d['hi']:.4g}]")
        L.append("\n### 実行時間(全質問の合計、秒)\n")
        L.append(", ".join(f"{a}: {b:.0f}" for a, b in r["time_sec"].items()) + f"、前処理: {r['t_prepare_sec']:.0f}\n")
        g = np.array(r["gold_n"])
        L.append("### 支持段落の数ごとの FCR@5(catrag / static_ppr / dense)\n")
        for n_g in sorted(set(g.tolist())):
            sel = g == n_g
            L.append(f"- 支持段落 {n_g} 件(質問 {int(sel.sum())} 件): " + " / ".join(f"{100*np.mean(np.array(m[x]['fcr5'])[sel]):.1f}" for x in ["catrag", "static_ppr", "dense"]))
        L.append("")
    L.append("## ハイパーパラメータの感度(評価用の質問。値の選択には使っていない)\n")
    L.append("静的 PPR との差(% ポイント、対応ありブートストラップ 95% 区間。*は 0 をまたがない)。\n")
    for k, name in DS:
        p = HERE / "results" / f"sweep_{k}.json"
        if not p.exists():
            continue
        sw = json.loads(p.read_text())
        L.append(f"### {name}\n")
        L.append("| 設定 | R@5 の差 | FCR@5 の差 |")
        L.append("|---|---|---|")
        for cfg, v in sw.items():
            if cfg == "static":
                L.append(f"| static(絶対値) | {100*np.mean(v['recall5']):.1f} | {100*np.mean(v['fcr5']):.1f} |")
                continue
            cells = []
            for mk in ("recall5", "fcr5"):
                d = cm.paired_diff_ci(np.array(v[mk]), np.array(sw["static"][mk]), 5000, 0)
                cells.append(f"{100*d['diff']:+.1f} [{100*d['lo']:+.1f}, {100*d['hi']:+.1f}]" + ("*" if d["significant"] else ""))
            L.append(f"| {cfg} | " + " | ".join(cells) + " |")
        L.append("")
    (HERE.parent / "docs" / "tables.md").write_text("\n".join(L))
    figure(R)


def figure(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [k for k, _ in DS if k in R]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    names = ["dense", "sa_rag", "static_ppr", "catrag"]
    cols = ["#8a8a8a", "#c9a227", "#2a6f97", "#c1440e"]
    for ax, mk, title in zip(axes, ["recall5", "fcr5"], ["Recall@5", "Full Chain Retrieval@5"]):
        w = 0.2
        for i, (nm, c) in enumerate(zip(names, cols)):
            vals, err = [], []
            for k in ks:
                mean, lo, hi = cm.bootstrap_ci(np.array(R[k]["metrics"][nm][mk]), 3000, 0)
                vals.append(100 * mean)
                err.append([100 * (mean - lo), 100 * (hi - mean)])
            ax.bar(np.arange(len(ks)) + (i - 1.5) * w, vals, w, yerr=np.array(err).T, color=c, label=nm, capsize=2)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([dict(DS)[k] for k in ks])
        ax.set_title(title + " (%)")
        ax.set_ylim(0, 100)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE.parent / "docs" / "main.svg")


if __name__ == "__main__":
    main()
