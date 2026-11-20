"""結果の集計: 表(docs/tables.md)、図(docs/em.svg、docs/curves.svg)、results/summary.json を作る。"""
from __future__ import annotations

import json

import numpy as np

import common  # noqa: F401
from david_stats import bootstrap_ci, paired_diff_ci, spans_zero

RES = common.RESULTS
DOCS = common.HERE.parent / "docs"
DS = ["hotpot", "2wiki", "musique"]
DS_NAME = {"hotpot": "HotpotQA", "2wiki": "2WikiMultiHopQA", "musique": "MuSiQue"}
RUNS = [("untrained", "未学習(同じ形式のプロンプトのみ)"), ("baseline", "baseline(outcome のみの GRPO、cold start)"),
        ("full", "DAVID-GRPO(3 部品すべて)"), ("no_warm", "warm-start なし"),
        ("no_reward", "grounded retrieval reward なし (λ=0)"), ("no_expansion", "grounded expansion なし")]


def load(run, ds):
    p = RES / run / f"eval_{ds}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(m, lo, hi):
    return f"{100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def main():
    runs = [(r, n) for r, n in RUNS if load(r, "hotpot")]
    summ = {}
    out = ["# 結果の表(`david_mini/analyze.py` が `david_mini/results/*` から生成)\n",
           "EM は質問ごとの 0/1 の平均(%)で、[ ] は 5,000 回のブートストラップによる 95% 区間です。\n"]
    out.append("## EM(各 300 問)\n")
    out.append("| 構成 | " + " | ".join(DS_NAME[d] for d in DS) + " | 3 データの平均 |")
    out.append("|---|---|---|---|---|")
    for r, n in runs:
        cells, allv = [], []
        for d in DS:
            v = [x["em"] for x in load(r, d)]
            allv += v
            m, lo, hi = bootstrap_ci(v)
            cells.append(pct(m, lo, hi))
            summ.setdefault(r, {})[d] = {"em": m, "lo": lo, "hi": hi, "n": len(v)}
        m, lo, hi = bootstrap_ci(allv)
        summ[r]["all"] = {"em": m, "lo": lo, "hi": hi, "n": len(allv)}
        out.append(f"| {n} | " + " | ".join(cells) + f" | {pct(m, lo, hi)} |")
    for ref, title in (("baseline", "baseline との差(同じ質問での対応あり、% ポイント。* は区間が 0 をまたがない)"),
                       ("full", "DAVID-GRPO(full)との差(各構成 − full。* は区間が 0 をまたがない)")):
        if not load(ref, "hotpot"):
            continue
        out.append(f"\n## {title}\n")
        out.append("| 構成 | " + " | ".join(DS_NAME[d] for d in DS) + " | 3 データをまとめた差 |")
        out.append("|---|---|---|---|---|")
        for r, n in runs:
            if r == ref:
                continue
            cells, allA, allB = [], [], []
            for d in DS:
                a = [x["em"] for x in load(r, d)]
                b = [x["em"] for x in load(ref, d)]
                allA += a; allB += b
                m, lo, hi = paired_diff_ci(a, b)
                summ[r].setdefault(f"diff_vs_{ref}", {})[d] = {"d": m, "lo": lo, "hi": hi}
                cells.append(f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]" + ("" if spans_zero(lo, hi) else "*"))
            m, lo, hi = paired_diff_ci(allA, allB)
            summ[r].setdefault(f"diff_vs_{ref}", {})["all"] = {"d": m, "lo": lo, "hi": hi}
            cells.append(f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]" + ("" if spans_zero(lo, hi) else "*"))
            out.append(f"| {n} | " + " | ".join(cells) + " |")
    out.append("\n## 行動の統計と証拠の再現率(評価 300 問ずつ)\n")
    out.append("| 構成 | データ | 検索回数(平均) | 答えを出した割合 % | 証拠の再現率(平均)% | 証拠が全部そろった割合 % |")
    out.append("|---|---|---|---|---|---|")
    for r, n in runs:
        for d in DS:
            rows = load(r, d)
            ns = np.mean([x["n_search"] for x in rows])
            an = 100 * np.mean([x["answered"] for x in rows])
            rc = 100 * np.mean([x["recall"] for x in rows])
            fr = 100 * np.mean([x["recall"] >= 1 - 1e-9 for x in rows])
            summ[r][d].update({"n_search": float(ns), "answered": an, "recall": rc, "full_recall": fr})
            out.append(f"| {r} | {DS_NAME[d]} | {ns:.2f} | {an:.1f} | {rc:.1f} | {fr:.1f} |")
    # 学習ログ
    out.append("\n## 学習ログの要約\n")
    out.append("| 構成 | ステップ数 | 1 ステップの時間(秒、平均) | 生成のピーク(GB) | 学習のピーク(GB) | 全フェーズのピーク(GB) | 生成トークン/秒(平均) | 拡張の試行/採用(合計) |")
    out.append("|---|---|---|---|---|---|---|---|")
    logs = {}
    for r, n in runs:
        p = RES / r / "log.jsonl"
        if not p.exists():
            continue
        raw = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        L = list({x['step']: x for x in raw}.values())  # 再開で重複したステップは後のものを使う
        L.sort(key=lambda x: x['step'])
        logs[r] = L
        pk = json.loads((RES / r / "peak_mem.json").read_text()) if (RES / r / "peak_mem.json").exists() else {}
        s = {"steps": len(L), "t_step": float(np.mean([x["t_step"] for x in L])),
             "gen_peak": max(x["gen_peak_gb"] for x in L), "train_peak": max(x["train_peak_gb"] for x in L),
             "overall_peak": max(pk.values()) if pk else None, "tok_per_s": float(np.mean([x["tok_per_s"] for x in L])),
             "n_plan": sum(x["n_plan"] for x in L), "n_accept": sum(x["n_accept"] for x in L),
             "n_extra": sum(x["n_extra_rollouts"] for x in L), "phase_peaks": pk}
        summ[r]["train"] = s
        out.append(f"| {r} | {s['steps']} | {s['t_step']:.1f} | {s['gen_peak']:.2f} | {s['train_peak']:.2f} | "
                   f"{s['overall_peak']} | {s['tok_per_s']:.0f} | {s['n_plan']}/{s['n_accept']} |")
    # 2 つ目の seed(baseline と full だけ)
    if load("baseline_s1", "hotpot") and load("full_s1", "hotpot"):
        out.append("\n## 2 つ目の seed (seed 1)での baseline と DAVID-GRPO\n")
        out.append("| 構成 | HotpotQA | 2WikiMultiHopQA | MuSiQue | 3 データの平均 |")
        out.append("|---|---|---|---|---|")
        for r in ("baseline_s1", "full_s1"):
            cells, allv = [], []
            for d in DS:
                v = [x["em"] for x in load(r, d)]
                allv += v
                m, lo, hi = bootstrap_ci(v)
                cells.append(pct(m, lo, hi))
                summ.setdefault(r, {})[d] = {"em": m, "lo": lo, "hi": hi, "n": len(v)}
            m, lo, hi = bootstrap_ci(allv)
            summ[r]["all"] = {"em": m, "lo": lo, "hi": hi}
            out.append(f"| {r} | " + " | ".join(cells) + f" | {pct(m, lo, hi)} |")
        out.append("\n**full − baseline(同じ seed どうしの対応あり差、% ポイント。* は区間が 0 をまたがない)**\n")
        out.append("| seed | HotpotQA | 2WikiMultiHopQA | MuSiQue | 3 データをまとめた差 |")
        out.append("|---|---|---|---|---|")
        for sd, (a_, b_) in (("0", ("full", "baseline")), ("1", ("full_s1", "baseline_s1"))):
            cells, A, B = [], [], []
            for d in DS:
                a = [x["em"] for x in load(a_, d)]
                b = [x["em"] for x in load(b_, d)]
                A += a; B += b
                m, lo, hi = paired_diff_ci(a, b)
                summ.setdefault(f"seed{sd}_full_minus_baseline", {})[d] = {"d": m, "lo": lo, "hi": hi}
                cells.append(f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]" + ("" if spans_zero(lo, hi) else "*"))
            m, lo, hi = paired_diff_ci(A, B)
            summ[f"seed{sd}_full_minus_baseline"]["all"] = {"d": m, "lo": lo, "hi": hi}
            cells.append(f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]" + ("" if spans_zero(lo, hi) else "*"))
            out.append(f"| {sd} | " + " | ".join(cells) + " |")
        for r in ("baseline_s1", "full_s1"):
            for d in DS:
                rows = load(r, d)
                summ[r][d].update({"n_search": float(np.mean([x["n_search"] for x in rows])),
                                   "answered": 100 * float(np.mean([x["answered"] for x in rows])),
                                   "recall": 100 * float(np.mean([x["recall"] for x in rows]))})
        out.append("\n| 構成 | データ | 検索回数(平均) | 答えを出した割合 % | 証拠の再現率(平均)% |")
        out.append("|---|---|---|---|---|")
        for r in ("baseline_s1", "full_s1"):
            for d in DS:
                u = summ[r][d]
                out.append(f"| {r} | {DS_NAME[d]} | {u['n_search']:.2f} | {u['answered']:.1f} | {u['recall']:.1f} |")
    (DOCS).mkdir(exist_ok=True)
    (DOCS / "tables.md").write_text("\n".join(out) + "\n")
    (RES / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    figs(summ, logs, runs)
    print("\n".join(out))


def figs(summ, logs, runs):
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "DejaVu Sans"
    cols = {"untrained": "#999999", "baseline": "#d95f02", "full": "#1b9e77", "no_warm": "#7570b3", "no_reward": "#e7298a", "no_expansion": "#66a61e"}
    labels = {"untrained": "untrained", "baseline": "baseline", "full": "full", "no_warm": "w/o warm-start", "no_reward": "w/o retrieval reward", "no_expansion": "w/o expansion"}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.8 / len(runs)
    for i, (r, _) in enumerate(runs):
        xs = np.arange(3) + i * w
        m = [100 * summ[r][d]["em"] for d in DS]
        lo = [100 * (summ[r][d]["em"] - summ[r][d]["lo"]) for d in DS]
        hi = [100 * (summ[r][d]["hi"] - summ[r][d]["em"]) for d in DS]
        ax.bar(xs, m, w, yerr=[lo, hi], color=cols[r], label=labels[r], capsize=2)
    ax.set_xticks(np.arange(3) + 0.4 - w / 2)
    ax.set_xticklabels([DS_NAME[d] for d in DS])
    ax.set_ylabel("EM (%)  [95% CI]")
    ax.legend(fontsize=7, ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(DOCS / "em.svg")
    plt.close(fig)
    if not logs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for r, L in logs.items():
        for ax, key, t in zip(axes, ["em", "recall", "searches"], ["training EM (on-policy rollouts)", "training evidence recall", "searches per rollout"]):
            y = np.array([x[key] for x in L])
            k = 5
            ys = np.convolve(y, np.ones(k) / k, mode="valid") if len(y) >= k else y
            ax.plot(np.arange(len(ys)) + (k if len(y) >= k else 1), ys, color=cols[r], label=labels[r])
            ax.set_title(t, fontsize=9)
            ax.set_xlabel("step (5-step moving average)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(DOCS / "curves.svg")


if __name__ == "__main__":
    main()
