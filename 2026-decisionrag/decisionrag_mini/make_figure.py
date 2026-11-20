"""results/<名前>.json から、図(SVG)と表(markdown)を作る。標準ライブラリのみ。

  python decisionrag_mini/make_figure.py --results decisionrag_mini/results/judge.json --out docs

図 1(main.svg): 手法ごとの Hit@1 と R@10(95% ブートストラップ区間)
図 2(pool_scaling.svg): 候補数 N と Hit@1(二値を 1 件ずつ / 全部 1 請求 / 複数選択 / 二値の上位を複数選択)
図 3(request_size.svg): 候補 200 件の池を 1 請求 m 件に分けたときの Hit@1
図 4(budget.svg): 判定数(費用の代わり)と R@10。設定ごとに 1 点
表(tables.md): 全数値と、対応ありの差の 95% 区間
信頼区間は、評価の質問を復元抽出して平均を取り直す(2000 回)。差の区間は、同じ質問での差を使う(対応あり)。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

INK, MUTE, GRID = "#1f2933", "#6b7785", "#e3e8ee"
COL = ["#8a97a6", "#1b6ca8", "#c2410c", "#2f855a", "#6b46c1", "#b7791f"]
LABEL = {"bm25": "BM25", "bm25_top100_judge": "BM25 上位 100 + 判定(rerank)", "decisionrag": "DecisionRAG(木をたどる)",
         "decisionrag_bm25": "DecisionRAG + BM25 の候補",
         "binary_separate": "二値、1 件ずつ", "binary_joint": "二値、全部 1 請求", "choice": "複数選択(全候補)", "binary_then_choice": "二値の上位 → 複数選択"}


def boot(vals, n=2000, seed=0):
    rng = random.Random(seed)
    k = len(vals)
    ms = sorted(st.mean(vals[rng.randrange(k)] for _ in range(k)) for _ in range(n))
    return st.mean(vals), ms[int(0.025 * n)], ms[int(0.975 * n)]


def paired(a, b, n=2000, seed=0):
    return boot([x - y for x, y in zip(a, b)], n, seed)


class Svg:
    def __init__(self, w, h, title):
        self.w, self.h, self.p = w, h, [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif" font-size="12">',
                                         f'<rect width="{w}" height="{h}" fill="#ffffff"/>']
        self.text(w / 2, 22, title, 14, "middle", INK, "bold")

    def text(self, x, y, s, size=12, anchor="start", color=INK, weight="normal"):
        s = str(s).replace("&", "&amp;").replace("<", "&lt;")
        self.p.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{s}</text>')

    def line(self, x1, y1, x2, y2, color=INK, width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{d}/>')

    def rect(self, x, y, w, h, color):
        self.p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0, w):.1f}" height="{max(0, h):.1f}" fill="{color}"/>')

    def circle(self, x, y, r, color):
        self.p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"/>')

    def end(self):
        return "\n".join(self.p + ["</svg>"])


def _axes(s, L, R, T, B, xmap, ymap, xticks, yticks, xlabel, ylabel, fmt_x=str):
    for t in yticks:
        s.line(L, ymap(t), s.w - R, ymap(t), GRID)
        s.text(L - 8, ymap(t) + 4, f"{t:.1f}", 11, "end", MUTE)
    for t in xticks:
        s.line(xmap(t), T, xmap(t), s.h - B, GRID)
        s.text(xmap(t), s.h - B + 16, fmt_x(t), 11, "middle", MUTE)
    s.text((L + s.w - R) / 2, s.h - 12, xlabel, 11, "middle", MUTE)
    s.text(14, (T + s.h - B) / 2, ylabel, 11, "middle", MUTE)


def fig_main(res, path):
    main = res.get("main")
    if not main:
        return
    names = [n for n in ("bm25", "bm25_top100_judge", "decisionrag", "decisionrag_bm25") if n in main]
    s = Svg(820, 120 + 34 * len(names), "手法ごとの Hit@1 と R@10(95% 区間)")
    left, top = 250, 50
    for pi, (key, lab) in enumerate((("hit1", "Hit@1"), ("r10", "R@10"))):
        x0, x1 = left + pi * 290, left + pi * 290 + 250
        X = lambda v, x0=x0, x1=x1: x0 + v * (x1 - x0)
        s.text((x0 + x1) / 2, top - 6, lab, 12, "middle", INK, "bold")
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            s.line(X(t), top, X(t), top + 34 * len(names), GRID)
            s.text(X(t), top + 34 * len(names) + 14, f"{t:.2f}", 10, "middle", MUTE)
        for i, n in enumerate(names):
            y = top + i * 34 + 17
            m, a, b = boot(main[n]["per_query"][key])
            if pi == 0:
                s.text(left - 10, y + 4, LABEL[n], 12, "end")
            s.rect(x0, y - 9, X(m) - x0, 18, COL[i % len(COL)])
            s.line(X(a), y, X(b), y, INK, 1.5)
            s.text(X(b) + 5, y + 4, f"{m:.3f}", 10, "start", MUTE)
    path.write_text(s.end(), encoding="utf-8")


def _lines(res_part, modes, xs, path, title, xlabel, log_x=True, hline=None):
    import math
    L, R, T, B = 70, 190, 46, 54
    s = Svg(820, 380, title)
    xv = [math.log(x) for x in xs] if log_x else xs
    lo, hi = min(xv), max(xv)
    X = lambda x: L + ((math.log(x) if log_x else x) - lo) / max(1e-9, hi - lo) * (s.w - L - R)
    Y = lambda v: s.h - B - v * (s.h - T - B)
    _axes(s, L, R, T, B, X, Y, xs, (0, 0.2, 0.4, 0.6, 0.8, 1.0), xlabel, "Hit@1")
    for mi, mode in enumerate(modes):
        pts = [(x, res_part[str(x)]["summary"].get(mode)) for x in xs if res_part.get(str(x)) and mode in res_part[str(x)]["summary"]]
        if not pts:
            continue
        c = COL[(mi + 1) % len(COL)]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            s.line(X(x1), Y(y1), X(x2), Y(y2), c, 2)
        for x, y in pts:
            s.circle(X(x), Y(y), 4, c)
        s.text(s.w - R + 10, 70 + mi * 20, LABEL.get(mode, mode), 11, "start", c)
    if hline is not None:
        s.line(L, Y(hline), s.w - R, Y(hline), COL[3], 1.5, "5,4")
        s.text(s.w - R + 10, 70 + len(modes) * 20, "複数選択(全候補)", 11, "start", COL[3])
    path.write_text(s.end(), encoding="utf-8")


def fig_pool(res, path):
    ps = res.get("pool_scaling")
    if ps:
        xs = sorted(int(k) for k in ps)
        _lines(ps, ["binary_separate", "binary_joint", "choice", "binary_then_choice"], xs, path, "候補数 N と Hit@1(正解 1 件 + BM25 の難しい負例)", "候補数 N(すべて同じ文脈で採点)")


def fig_request(res, path):
    rs = res.get("request_size")
    if rs:
        xs = sorted(int(k) for k in rs if k != "choice_all")
        _lines(rs, ["hit1"], xs, path, "1 請求あたりの候補数 m と Hit@1(候補 200 件の池)", "1 請求あたりの候補数 m", hline=rs.get("choice_all", {}).get("summary", {}).get("hit1"))


def fig_budget(res, path):
    b = res.get("budget")
    if not b:
        return
    pts = [(k, v["summary"]["judgments"], v["summary"]["r10"]) for k, v in b.items()]
    L, R, T, B = 70, 40, 46, 54
    s = Svg(820, 400, "予算と再現率(1 点 = 1 設定。左上ほど良い)")
    xmax = max(p[1] for p in pts) * 1.1
    X = lambda v: L + v / xmax * (s.w - L - R)
    Y = lambda v: s.h - B - v * (s.h - T - B)
    xt = [round(xmax * i / 5) for i in range(6)]
    _axes(s, L, R, T, B, X, Y, xt, (0, 0.2, 0.4, 0.6, 0.8, 1.0), "1 問あたりの判定数(費用の目安)", "R@10")
    for i, (k, j, r) in enumerate(pts):
        s.circle(X(j), Y(r), 5, COL[1])
        s.text(X(j) + 7, Y(r) - 5, k.replace("k1=", "k1 ").replace(",k2=", " / k2 "), 9, "start", MUTE)
    path.write_text(s.end(), encoding="utf-8")


def tables(res) -> str:
    out = ["# 集計表", "", f"評価の質問数: {res.get('setup', {}).get('n_eval', '?')}。区間は 95% ブートストラップ(質問を復元抽出)。", ""]
    main = res.get("main")
    if main:
        out += ["## 手法の比較", "", "| 手法 | Hit@1 | R@10 | nDCG@10 | 判定数/問 | トークン/問 | 秒/問 | 逐次の段数 |", "|---|---|---|---|---|---|---|---|"]
        for n in ("bm25", "bm25_top100_judge", "decisionrag", "decisionrag_bm25"):
            if n in main:
                p, sm = main[n]["per_query"], main[n]["summary"]
                m1, a1, b1 = boot(p["hit1"])
                m2, a2, b2 = boot(p["r10"])
                out.append(f"| {LABEL[n]} | {m1:.3f} [{a1:.3f}, {b1:.3f}] | {m2:.3f} [{a2:.3f}, {b2:.3f}] | {sm['ndcg10']:.3f} | {sm['judgments']:.0f} | {sm['tokens']:.0f} | {sm['sec']:.2f} | {sm['stages']:.0f} |")
        out += ["", "対応ありの差(同じ質問での差の 95% 区間):", "", "| A | B | 指標 | A − B |", "|---|---|---|---|"]
        for a, b in (("decisionrag", "bm25_top100_judge"), ("decisionrag_bm25", "decisionrag"), ("decisionrag", "bm25")):
            if a in main and b in main:
                for key in ("hit1", "r10"):
                    m, lo, hi = paired(main[a]["per_query"][key], main[b]["per_query"][key])
                    out.append(f"| {LABEL[a]} | {LABEL[b]} | {key} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        out.append("")
    ps = res.get("pool_scaling")
    if ps:
        out += ["## 候補数 N と Hit@1", "", "| N | " + " | ".join(LABEL[m] for m in ("binary_separate", "binary_joint", "choice", "binary_then_choice")) + " |", "|---|---|---|---|---|"]
        for n in sorted(ps, key=int):
            sm = ps[n]["summary"]
            out.append(f"| {n} | " + " | ".join(f"{sm[m]:.3f}" if m in sm else "-" for m in ("binary_separate", "binary_joint", "choice", "binary_then_choice")) + " |")
        out += ["", "共有文脈の効果(全部 1 請求 − 1 件ずつ、対応ありの差):", "", "| N | 差 |", "|---|---|"]
        for n in sorted(ps, key=int):
            p = ps[n]["per_query"]
            m, lo, hi = paired(p["binary_joint"], p["binary_separate"])
            out.append(f"| {n} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        out.append("")
    rs = res.get("request_size")
    if rs:
        out += ["## 1 請求あたりの候補数 m(候補 200 件)", "", "| m | Hit@1 | R@10 |", "|---|---|---|"]
        for k in sorted((k for k in rs if k != "choice_all"), key=int):
            out.append(f"| {k} | {rs[k]['summary']['hit1']:.3f} | {rs[k]['summary']['r10']:.3f} |")
        if "choice_all" in rs:
            out.append(f"| 複数選択(全部) | {rs['choice_all']['summary']['hit1']:.3f} | - |")
        out.append("")
    ch = res.get("choice")
    if ch:
        out += ["## 複数選択による並べ替え(上位 m 件)", "", "| m | Hit@1 | R@10 |", "|---|---|---|"]
        for k in sorted(ch, key=int):
            out.append(f"| {k} | {ch[k]['summary']['hit1']:.3f} | {ch[k]['summary']['r10']:.3f} |")
        out.append("")
    b = res.get("budget")
    if b:
        out += ["## 予算(k1: 残すクラスタ、k2: 残す記事)", "", "| 設定 | R@10 | Hit@1 | 判定数/問 | トークン/問 | 秒/問 |", "|---|---|---|---|---|---|"]
        for k, v in b.items():
            sm = v["summary"]
            out.append(f"| {k} | {sm['r10']:.3f} | {sm['hit1']:.3f} | {sm['judgments']:.0f} | {sm['tokens']:.0f} | {sm['sec']:.2f} |")
        out.append("")
    lat = res.get("latency")
    if lat:
        out += ["## 1 請求の時間と、式 4 の見積もり", "", "| 1 請求の候補数 | 中央値(秒) |", "|---|---|"]
        for k, v in lat["request_sec_median"].items():
            out.append(f"| {k} | {v:.3f} |")
        out += ["", f"式 4(b=100、t_req={lat['t_request']} 秒): " + "、".join(f"N={k}: {v:.1f} 秒" for k, v in lat["analytic_sec_b100"].items()), ""]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    res = json.loads(Path(a.results).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_main(res, out / "main.svg")
    fig_pool(res, out / "pool_scaling.svg")
    fig_request(res, out / "request_size.svg")
    fig_budget(res, out / "budget.svg")
    (out / "tables.md").write_text(tables(res), encoding="utf-8")
    print("wrote", out)
