"""results/*.json から図(SVG)と集計表(markdown)を作る。標準ライブラリのみ。

  python rankrag_mini/make_figure.py       # docs/ranking_size.svg, docs/topk.svg, docs/tables.md を書く

図 1: ranking データの行数 × 並べ替え後の R@5(第 1 版と第 2 版、SQuAD と HotpotQA)
図 2: 文脈数 k × EM(検索順と RankRAG の rerank、第 2 版の SQuAD と HotpotQA)
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "results"
OUT = HERE.parent / "docs"
INK, MUTE, GRID = "#1f2933", "#6b7785", "#e3e8ee"
C = {"v1": "#8a97a6", "v2": "#1b6ca8", "ret": "#8a97a6", "rr": "#c2410c", "chat": "#2f855a"}


def load(name: str):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def seeds(name: str) -> list[dict]:
    return [r for r in (load(name), load(f"{name}_s1"), load(f"{name}_s2")) if r]


def r5(res: dict, ds: str) -> float:
    return res["eval"][ds]["ranking"]["dense"]["reranked"]["R@5"]


def mean_sd(vals: list[float]) -> tuple[float, float]:
    return (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0)


class Plot:
    def __init__(self, w, h, title):
        self.w, self.h, self.parts = w, h, []
        self.parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif" font-size="12">')
        self.parts.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
        self.text(w / 2, 20, title, size=14, anchor="middle", weight="bold")

    def text(self, x, y, s, size=12, anchor="start", color=INK, weight="normal"):
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{s}</text>')

    def line(self, x1, y1, x2, y2, color=INK, width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{d}/>')

    def circle(self, x, y, r, color):
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"/>')

    def poly(self, pts, color, width=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{color}" stroke-width="{width}"{d}/>')

    def save(self, path: Path):
        path.write_text("\n".join(self.parts + ["</svg>"]))


def panel(p: Plot, x0, y0, w, h, title, xs, series, ylim, xlabel, ylabel, xlog=False, ref=None, xticks=None):
    """series: [(ラベル, 色, [(x, mean, sd) ...], 点線か)]"""
    import math

    fx = (lambda v: math.log10(v)) if xlog else (lambda v: v)
    lo, hi = fx(min(xs)), fx(max(xs))
    X = lambda v: x0 + (fx(v) - lo) / (hi - lo) * w  # noqa: E731
    Y = lambda v: y0 + h - (v - ylim[0]) / (ylim[1] - ylim[0]) * h  # noqa: E731
    p.text(x0, y0 - 10, title, size=12, weight="bold")
    n = 5
    for i in range(n + 1):
        v = ylim[0] + (ylim[1] - ylim[0]) * i / n
        p.line(x0, Y(v), x0 + w, Y(v), GRID)
        p.text(x0 - 6, Y(v) + 4, f"{v:.2f}", size=10, anchor="end", color=MUTE)
    for v in xticks or xs:
        p.line(X(v), y0 + h, X(v), y0 + h + 4, MUTE)
        p.text(X(v), y0 + h + 16, f"{v:g}", size=10, anchor="middle", color=MUTE)
    p.line(x0, y0 + h, x0 + w, y0 + h, MUTE)
    p.text(x0 + w / 2, y0 + h + 32, xlabel, size=11, anchor="middle", color=MUTE)
    p.text(x0 - 40, y0 - 10, ylabel, size=10, color=MUTE)
    if ref is not None:
        p.line(x0, Y(ref[0]), x0 + w, Y(ref[0]), MUTE, 1, "4 3")
        p.text(x0 + w - 2, Y(ref[0]) - 4, ref[1], size=10, anchor="end", color=MUTE)
    for label, color, pts, dash in series:
        if not pts:
            continue
        p.poly([(X(x), Y(m)) for x, m, _ in pts], color, 2, dash)
        for x, m, sd in pts:
            if sd:
                p.line(X(x), Y(m - sd), X(x), Y(m + sd), color, 1)
            p.circle(X(x), Y(m), 3.5, color)


def legend(p: Plot, x, y, items):
    for i, (label, color, dash) in enumerate(items):
        yy = y + i * 16
        p.line(x, yy - 4, x + 22, yy - 4, color, 2, dash)
        p.text(x + 28, yy, label, size=11)


SIZES_V1 = [500, 2000, 8000]
SIZES_V2 = [500, 1000, 2000, 4000, 8000, 16000]


def size_series(prefix: str, sizes, ds):
    pts = []
    for s in sizes:
        runs = seeds(f"{prefix}_rr_r{s}")
        if runs:
            m, sd = mean_sd([r5(r, ds) for r in runs])
            pts.append((s, m, sd))
    return pts


def fig_size():
    p = Plot(900, 360, "ranking データの行数と、並べ替え後の R@5(dense 検索の上位 30 件を並べ替え)")
    for j, (ds, name) in enumerate((("squad", "SQuAD 開発セット"), ("hotpot", "HotpotQA(zero-shot)"))):
        x0 = 70 + j * 430
        ref1 = load("05b_rr_r2000")["eval"][ds]["ranking"]["dense"]["retriever"]["R@5"]
        v2ref = load("v2_base")
        ref2 = v2ref["eval"][ds]["ranking"]["dense"]["retriever"]["R@5"] if v2ref else None
        series = [("第 1 版", C["v1"], size_series("05b", SIZES_V1, ds), None),
                  ("第 2 版", C["v2"], size_series("v2", SIZES_V2, ds), None)]
        panel(p, x0, 60, 330, 220, name, SIZES_V2, series, (0.3, 1.0), "ranking の行数(対数軸)", "R@5", xlog=True,
              ref=(ref2 if ref2 is not None else ref1, "検索のみ(第 2 版)" if ref2 is not None else "検索のみ"), xticks=SIZES_V2)
    legend(p, 300, 345, [("第 1 版(コーパス小・ランダム負例)", C["v1"], None), ("第 2 版(コーパス大・BM25 hard negative)", C["v2"], None)])
    p.save(OUT / "ranking_size.svg")


def fig_topk():
    p = Plot(900, 360, "文脈数 k と EM(第 2 版、ranking 2,000 行の RankRAG と ChatQA 相当)")
    for j, (ds, name) in enumerate((("squad", "SQuAD 開発セット"), ("hotpot", "HotpotQA(zero-shot)"))):
        x0 = 70 + j * 430
        rr, ch = seeds("v2_rr_r2000"), seeds("v2_chatqa")
        def pts(runs, kind):
            return [(k, *mean_sd([r["eval"][ds]["generation"][f"{kind}_k{k}"]["EM"] for r in runs])) for k in (1, 3, 5)] if runs else []
        series = [("検索順(ChatQA 相当)", C["chat"], pts(ch, "retriever"), "5 3"),
                  ("検索順(RankRAG)", C["ret"], pts(rr, "retriever"), None),
                  ("rerank(RankRAG)", C["rr"], pts(rr, "rerank"), None)]
        panel(p, x0, 60, 330, 220, name, [1, 3, 5], series, (0.0, 0.8), "文脈数 k", "EM", xticks=[1, 3, 5])
    legend(p, 250, 345, [("ChatQA 相当(検索順)", C["chat"], "5 3"), ("RankRAG(検索順)", C["ret"], None), ("RankRAG(rerank 順)", C["rr"], None)])
    p.save(OUT / "topk.svg")


def rk(res, ds, key, m="R@5"):
    return res["eval"][ds]["ranking"]["dense"][key].get(m)


def cell(runs, ds, key, m):
    vals = [rk(r, ds, key, m) for r in runs]
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    mu, sd = mean_sd(vals)
    return f"{mu:.3f}" + (f" ± {sd:.3f}" if len(vals) > 1 else "")


def table(title, names, note=""):
    out = [f"### {title}\n"]
    if note:
        out += [note, ""]
    out.append("| 条件 | seed 数 | SQuAD R@5(検索 → rerank) | SQuAD AUROC | HotpotQA R@5(検索 → rerank) | HotpotQA AUROC |")
    out.append("|---|---|---|---|---|---|")
    for n, label in names:
        runs = seeds(n)
        if not runs:
            continue
        cs = []
        for ds in ("squad", "hotpot"):
            base = rk(runs[0], ds, "retriever")
            cs.append(f"{base:.3f} → {cell(runs, ds, 'reranked', 'R@5')}")
            cs.append(cell(runs, ds, "reranked", "AUROC"))
        out.append(f"| {label} | {len(runs)} | " + " | ".join(cs) + " |")
    out.append("")
    return "\n".join(out)


def gen_table(title, names):
    out = [f"### {title}\n", "| 条件 | seed 数 | SQuAD EM k=1(検索 → rerank) | SQuAD EM k=5(検索 → rerank) | HotpotQA EM k=1 | HotpotQA EM k=5 |", "|---|---|---|---|---|---|"]
    for n, label in names:
        runs = seeds(n)
        if not runs or not runs[0]["eval"]["squad"].get("generation"):
            continue
        cs = []
        for ds in ("squad", "hotpot"):
            for k in (1, 5):
                a = mean_sd([r["eval"][ds]["generation"][f"retriever_k{k}"]["EM"] for r in runs])[0]
                b = mean_sd([r["eval"][ds]["generation"][f"rerank_k{k}"]["EM"] for r in runs])[0]
                cs.append(f"{a:.3f} → {b:.3f}")
        out.append(f"| {label} | {len(runs)} | " + " | ".join(cs) + " |")
    out.append("")
    return "\n".join(out)


MAIN = [("v2_base", "学習なし"), ("v2_chatqa", "ChatQA 相当(ranking なし)")] + [(f"v2_rr_r{s}", f"RankRAG, ranking {s:,} 行") for s in SIZES_V2] + [
    ("v2_rr_r2000_noRAR", "RankRAG 2,000 行から RAR を除く"), ("v2_rr_r2000_noRQA", "RankRAG 2,000 行から RQA を除く"),
    ("v2_specialist", "専門 ranker(20,000 行のみ)")]
CTRL = [("v2_rr_r8000", "RankRAG 8,000 行(総数 18,000 行)"), ("v2_rr_matched8k", "総数を揃えた対照(8,000 行、ranking 2,000 行)"),
        ("v2_rankonly_r2000", "ranking 2,000 行のみ")]
DIAG = [("v2_rr_r2000", "2,000 行・元の負例"), ("v2_rr_r2000_rand", "2,000 行・ランダム負例"), ("v2_rr_r2000_bm25x", "2,000 行・自分の候補を除く BM25 負例"),
        ("v2_rr_r8000", "8,000 行・元の負例"), ("v2_rr_r8000_rand", "8,000 行・ランダム負例"), ("v2_rr_r8000_bm25x", "8,000 行・自分の候補を除く BM25 負例"),
        ("v2_specialist", "専門 ranker・元の負例"), ("v2_specialist_rand", "専門 ranker・ランダム負例"), ("v2_specialist_bm25x", "専門 ranker・自分の候補を除く BM25 負例")]
B15 = [("v2_15b_base", "学習なし"), ("v2_15b_chatqa", "ChatQA 相当"), ("v2_15b_rr_r2000", "RankRAG 2,000 行"), ("v2_15b_specialist", "専門 ranker")]
V1 = [("05b_base", "学習なし"), ("05b_chatqa", "ChatQA 相当"), ("05b_rr_r500", "RankRAG 500 行"), ("05b_rr_r2000", "RankRAG 2,000 行"),
      ("05b_rr_r8000", "RankRAG 8,000 行"), ("05b_rr_r2000_noRAR", "RAR なし"), ("05b_rr_r2000_noRQA", "RQA なし"), ("05b_specialist", "専門 ranker")]


def fig_diag():
    p = Plot(900, 360, "ranking の負例の取り方と、並べ替え後の R@5(0.5B、1 seed)")
    xs = [2000, 8000, 20000]
    for j, (ds, name) in enumerate((("squad", "SQuAD 開発セット"), ("hotpot", "HotpotQA(zero-shot)"))):
        x0 = 70 + j * 430
        def pts(suffix):
            names = [f"v2_rr_r2000{suffix}", f"v2_rr_r8000{suffix}", f"v2_specialist{suffix}"]
            return [(x, r5(load(n), ds), 0) for x, n in zip(xs, names) if load(n)]
        series = [("元の BM25 負例", C["v1"], pts(""), None), ("ランダム負例", C["chat"], pts("_rand"), None),
                  ("自分の候補を除く BM25 負例", C["rr"], pts("_bm25x"), None)]
        panel(p, x0, 60, 330, 220, name, xs, series, (0.0, 1.0), "ranking の行数(対数軸、20,000 行は専門 ranker)", "R@5", xlog=True,
              ref=(rk(load("v2_base"), ds, "retriever"), "検索のみ"), xticks=xs)
    legend(p, 250, 345, [("元の BM25 負例", C["v1"], None), ("ランダム負例", C["chat"], None), ("自分の候補を除く BM25 負例", C["rr"], None)])
    p.save(OUT / "negatives.svg")


def tables() -> str:
    parts = [table("第 2 版・主な条件(0.5B)", MAIN, "seed の複数ある条件は平均 ± 標準偏差。"),
             table("対照(総数を揃える、ranking のみ)", CTRL),
             table("診断: ranking の負例の取り方(0.5B)", DIAG),
             table("1.5B", B15),
             table("第 1 版(0.5B、コーパス小・ランダム負例)", V1),
             gen_table("生成(EM、第 2 版 0.5B)", [(n, l) for n, l in MAIN + CTRL if load(n)]),
             gen_table("生成(EM、1.5B)", B15)]
    return "\n".join(parts)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_size()
    fig_topk()
    fig_diag()
    (OUT / "tables.md").write_text(tables())
    print("書き出し:", OUT)
