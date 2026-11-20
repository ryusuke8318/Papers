"""results/*.json から docs/tables.md、docs/main.svg、docs/alpha.svg、results/summary.json を作る。README の数値はすべて summary.json と一致させる。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import grapher_metrics as gm  # noqa: E402

R = HERE / "results"
DOCS = HERE.parent / "docs"
DS = [("hotpot", "HotpotQA"), ("2wiki", "2WikiMultiHopQA"), ("musique", "MuSiQue")]
MODES = ["ner", "ner+title", "ner_rare"]
PAPER = {  # 論文 Table 3 (Llama-Embed-Nemotron-8B+BM25 / Cohere-Embed-V4+BM25) の PR@10: baseline, PPR, GCS
    "hotpot": ((88.2, 84.5, 88.7), (85.5, 85.8, 86.4)),
    "2wiki": ((49.4, 49.6, 51.1), (48.3, 52.8, 50.1)),
    "musique": ((35.5, 27.7, 36.9), (34.1, 34.6, 35.5)),
}


def pct(x):
    return 100 * x


def fmt_ci(x):
    m, lo, hi = gm.bootstrap_ci(x)
    return f"{pct(m):.1f} [{pct(lo):.1f}, {pct(hi):.1f}]", (pct(m), pct(lo), pct(hi))


def fmt_d(a, b):
    d = gm.paired_diff_ci(a, b)
    star = "*" if d["significant"] else ""
    return f"{pct(d['diff']):+.1f} [{pct(d['lo']):+.1f}, {pct(d['hi']):+.1f}]{star}", d


def select_alpha(tune, mode):
    out = {}
    for meth in ("ppr", "gcs"):
        sc = {a: float(np.mean(tune["modes"][mode][f"{meth}_{a}"]["pr10"])) for a in tune["grid"]}
        out[meth] = max(sc, key=lambda a: (sc[a], -a))   # 同点は小さい alpha
        out[meth + "_scores"] = sc
    return out


def main():
    tune = json.load(open(R / "hotpot_tune.json"))
    alphas = {m: select_alpha(tune, m) for m in MODES}
    summ = {"alpha": {m: {k: v for k, v in a.items() if not k.endswith("_scores")} for m, a in alphas.items()},
            "tune_scores": {m: {k: v for k, v in a.items() if k.endswith("_scores")} for m, a in alphas.items()},
            "tune_n": tune["n_queries"], "tune_base_pr10": float(np.mean(tune["baseline"]["pr10"])), "ds": {}}
    L = ["# 表(自動生成。`grapher_mini/analyze.py`)", ""]
    L += [f"alpha の選択(HotpotQA の別の質問 {tune['n_queries']} 問、PR@10 最大): " + ", ".join(f"{m}: PPR={alphas[m]['ppr']}, GCS={alphas[m]['gcs']}" for m in MODES), ""]
    res = {d: json.load(open(R / f"{d}_eval.json")) for d, _ in DS}
    for d, nm in DS:
        r = res[d]
        b = r["baseline"]
        S = summ["ds"][d] = {"n_queries": r["n_queries"], "n_passages": r["n_passages"], "cand_pr": float(np.mean(b["cand_pr"])), "modes": {}}
        L += [f"## {nm}(質問 {r['n_queries']} 件、段落 {r['n_passages']} 件)", ""]
        L += [f"候補 200 件に正解が全部入る割合(PR@200、リランクの上限): {pct(np.mean(b['cand_pr'])):.1f}%", ""]
        for mode in MODES:
            m = r["modes"][mode]
            ap, ag = alphas[mode]["ppr"], alphas[mode]["gcs"]
            M = S["modes"][mode] = {"deg_mean": float(np.mean(m["deg_mean"])), "isolated_frac": float(np.mean(m["isolated_frac"])),
                                    "gold_isolated": float(np.mean(m["gold_isolated"]))}
            gcn = np.array(m["gold_connected"]); gcn = gcn[gcn >= 0]
            M["gold_connected"] = float(gcn.mean()) if len(gcn) else None
            L += [f"### 実体の作り方: {mode}(PPR の alpha={ap}、GCS の alpha={ag})", ""]
            L += [f"候補グラフの平均次数 {M['deg_mean']:.1f}(候補 200 件中)、辺のない候補の割合 {pct(M['isolated_frac']):.1f}%、正解が 2 件以上の質問で正解どうしがつながる割合 {pct(M['gold_connected']):.1f}%", ""]
            L += ["| 手法 | PR@5 | PR@10 | PR@5 の差(基準比) | PR@10 の差(基準比) | PR@10 悪化/改善した質問数 |", "|---|---|---|---|---|---|"]
            meths = {"baseline": (b["pr5"], b["pr10"]), "GraphER-PPR": (m[f"ppr_{ap}"]["pr5"], m[f"ppr_{ap}"]["pr10"]), "GraphER-GCS": (m[f"gcs_{ag}"]["pr5"], m[f"gcs_{ag}"]["pr10"])}
            for k, (p5, p10) in meths.items():
                s5, v5 = fmt_ci(p5); s10, v10 = fmt_ci(p10)
                if k == "baseline":
                    L.append(f"| {k} | {s5} | {s10} | | | |")
                    M[k] = {"pr5": v5, "pr10": v10}
                else:
                    d5s, d5 = fmt_d(p5, b["pr5"]); d10s, d10 = fmt_d(p10, b["pr10"])
                    rc = gm.regression_counts(p10, b["pr10"])
                    L.append(f"| {k} | {s5} | {s10} | {d5s} | {d10s} | {rc['worse']} / {rc['better']} |")
                    M[k] = {"pr5": v5, "pr10": v10, "d5": d5, "d10": d10, "reg10": rc, "reg5": gm.regression_counts(p5, b["pr5"])}
            L.append("")
            if mode == "ner":
                # alpha 感度(評価データでの値。選択には使っていない)
                L += ["alpha 感度(評価データ上の PR@10 の基準との差、% ポイント。選択には使っていない。*は 95% 区間が 0 をまたがない)", "", "| alpha | " + " | ".join(str(a) for a in r["grid"]) + " |", "|---" * (len(r["grid"]) + 1) + "|"]
                M["sens"] = {}
                for meth in ("ppr", "gcs"):
                    row, M["sens"][meth] = [], {}
                    for a in r["grid"]:
                        x = m[f"{meth}_{a}"]["pr10"]
                        dd = gm.paired_diff_ci(x, b["pr10"], n_boot=2000)
                        M["sens"][meth][str(a)] = {"diff": pct(dd["diff"]), "lo": pct(dd["lo"]), "hi": pct(dd["hi"])}
                        row.append(f"{pct(dd['diff']):+.1f}" + ("*" if dd["significant"] else ""))
                    L.append(f"| {meth.upper()} | " + " | ".join(row) + " |")
                L.append("")
                # GCS が基準を下回る alpha の数(平均で)、質問ごとの悪化
                M["gcs_alpha_below_base"] = {k: sum(1 for a in r["grid"] if np.mean(m[f"gcs_{a}"][k]) < np.mean(b[k]) - 1e-12) for k in ("pr5", "pr10")}
                M["gcs_alpha_ci_below"] = {k: sum(1 for a in r["grid"] if gm.paired_diff_ci(m[f"gcs_{a}"][k], b[k], n_boot=1000)["hi"] < 0) for k in ("pr5", "pr10")}
                M["ppr_alpha_ci_below"] = {k: sum(1 for a in r["grid"] if gm.paired_diff_ci(m[f"ppr_{a}"][k], b[k], n_boot=1000)["hi"] < 0) for k in ("pr5", "pr10")}
                # ハブ分析
                L += ["ハブ分析: 上位 10 件のうち、候補内で次数が上位 10% の節点の割合(平均。基準は seed 順位の上位 10 件)", "", "| 手法 | 上位 10 件中のハブの割合 | 正解の平均順位(0 始まり) |", "|---|---|---|"]
                M["hub"] = {"baseline": float(np.mean(b["hub10"]))}
                L.append(f"| baseline | {pct(M['hub']['baseline']):.1f}% | |")
                for k, key in (("GraphER-PPR", f"ppr_{ap}"), ("GraphER-GCS", f"gcs_{ag}")):
                    h = float(np.mean(m[key]["hub10"]))
                    gr = np.array(m[key]["rank_gold_mean"]); gr = float(gr[gr >= 0].mean())
                    M["hub"][k] = h; M.setdefault("gold_rank", {})[k] = gr
                    L.append(f"| {k} | {pct(h):.1f}% | {gr:.1f} |")
                L.append("")
                # PPR の各 alpha でのハブ割合
                M["hub_ppr_by_alpha"] = {str(a): float(np.mean(m[f"ppr_{a}"]["hub10"])) for a in r["grid"]}
                L.append("PPR の alpha ごとの上位 10 件中のハブの割合(%): " + ", ".join(f"{a}: {pct(v):.1f}" for a, v in zip(r["grid"], M["hub_ppr_by_alpha"].values())))
                L.append("")
    # GAT
    gp = R / "gat.json"
    if gp.exists():
        g = json.load(open(gp))
        summ["gat"] = {"alpha": g["alpha"], "ds": {}, "best_val": {k: max(e["val_pr10"] for e in v) for k, v in g["logs"].items()}}
        L += ["## GAT ランカー(HotpotQA の別の質問で学習。PR@K、%)", "", f"GCS の特徴の alpha={g['alpha']}。検証での最良の PR@10: " + ", ".join(f"{k}={pct(v):.1f}" for k, v in summ["gat"]["best_val"].items()), "",
              "| データ | 手法 | PR@5 | PR@10 | PR@10 の差(基準比) | PR@10 の差(GCS 比) |", "|---|---|---|---|---|---|"]
        for d, nm in DS:
            b = res[d]["baseline"]; m = res[d]["modes"]["ner"]; ag = alphas["ner"]["gcs"]
            summ["gat"]["ds"][d] = {}
            for name in ("gat", "mlp"):
                x = g["eval"][d][name]
                s5, v5 = fmt_ci(x["pr5"]); s10, v10 = fmt_ci(x["pr10"])
                d1s, d1 = fmt_d(x["pr10"], b["pr10"]); d2s, d2 = fmt_d(x["pr10"], m[f"gcs_{ag}"]["pr10"])
                summ["gat"]["ds"][d][name] = {"pr5": v5, "pr10": v10, "d_base": d1, "d_gcs": d2}
                L.append(f"| {nm} | {name.upper()} | {s5} | {s10} | {d1s} | {d2s} |")
        L.append("")
    json.dump(summ, open(R / "summary.json", "w"), indent=1)
    (DOCS / "tables.md").write_text("\n".join(L) + "\n")
    figures(res, alphas, summ)
    print("ok")


def figures(res, alphas, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    col = {"GraphER-PPR": "#c4552d", "GraphER-GCS": "#2a6f97", "GAT": "#5b8c5a"}
    for j, (K, kk) in enumerate((("PR@5", "d5"), ("PR@10", "d10"))):
        for i, k in enumerate(("GraphER-PPR", "GraphER-GCS")):
            for t, (d, nm) in enumerate(DS):
                v = summ["ds"][d]["modes"]["ner"][k][kk]
                x = t + (i - 0.5) * 0.3
                ax[j].bar(x, pct(v["diff"]), 0.28, color=col[k], label=k if t == 0 else None)
                ax[j].errorbar(x, pct(v["diff"]), yerr=[[pct(v["diff"] - v["lo"])], [pct(v["hi"] - v["diff"])]], color="k", capsize=3, lw=1)
        ax[j].axhline(0, color="gray", lw=0.8)
        ax[j].set_xticks(range(3)); ax[j].set_xticklabels([n for _, n in DS], fontsize=8)
        ax[j].set_title(f"{K}: difference from baseline (points, 95% CI)", fontsize=9)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(DOCS / "main.svg"); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for t, (d, nm) in enumerate(DS):
        for meth, k in (("ppr", "GraphER-PPR"), ("gcs", "GraphER-GCS")):
            s = summ["ds"][d]["modes"]["ner"]["sens"][meth]
            xs = [float(a) for a in s]
            ax[t].plot(xs, [s[a]["diff"] for a in s], "o-", color=col[k], label=k, ms=3)
            ax[t].fill_between(xs, [s[a]["lo"] for a in s], [s[a]["hi"] for a in s], color=col[k], alpha=0.15)
        ax[t].axhline(0, color="gray", lw=0.8); ax[t].set_title(nm, fontsize=9); ax[t].set_xlabel("alpha (weight on seed / restart prob.)", fontsize=8)
    ax[0].set_ylabel("PR@10 difference from baseline (points)", fontsize=8); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(DOCS / "alpha.svg"); plt.close(fig)


if __name__ == "__main__":
    main()
