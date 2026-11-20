"""results/*.json から図(SVG)と集計表(markdown)を作る。標準ライブラリのみ。

  python reasonrank_mini/make_figure.py    # docs/ndcg.svg, docs/tokens.svg, docs/tables.md を書く

図 1: 100 件のスライディングウィンドウでの NDCG@10(95% ブートストラップ信頼区間つき)
図 2: 質問あたりの出力トークン数 × NDCG@10(推論の費用と効果)
信頼区間は、評価の質問を復元抽出して平均を取り直す(2000 回)。差の区間は、同じ質問での差を使う(対応あり)。
"""
from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "results"
OUT = HERE.parent / "docs"
INK, MUTE, GRID = "#1f2933", "#6b7785", "#e3e8ee"
COL = {"init": "#8a97a6", "direct": "#2f855a", "reason": "#1b6ca8", "rl": "#c2410c", "teacher": "#6b46c1"}

# (表示名, 結果ファイル, 色)。推論つきのモデルは温度 0.6、順位のみは貪欲デコード
ROWS = [
    ("BM25(並べ替えなし)", None, "init"),
    ("ゼロショット 1.5B(順位のみ)", "zs_direct_pq", "direct"),
    ("ゼロショット 1.5B(推論)", "zs_cot_t06", "reason"),
    ("ゼロショット 教師 8B(思考)", "zs_teacher_t06", "teacher"),
    ("SFT 順位のみ", "sft_direct_pq", "direct"),
    ("SFT 推論(フィルタなし)", "sft_reason_nofilter_t06", "reason"),
    ("SFT 推論(フィルタあり)", "sft_reason_t06", "reason"),
    ("SFT 推論 + GRPO(NDCG のみ)", "grpo_ndcg_t06", "rl"),
    ("SFT 推論 + GRPO(マルチビュー)", "grpo_multi_t06", "rl"),
    ("RL のみ(SFT なし)", "rl_only_t06", "rl"),
    ("3B SFT 順位のみ(2 エポック)", "sft_direct_3b", "direct"),
    ("3B SFT 推論(2 エポック分で停止)", "sft_reason_3b_ep2_ev", "reason"),
]


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ev(name, split="eval100"):
    r = load(name)
    return r["eval"][split] if r else None


def boot_ci(vals, n=2000, seed=0):
    rng = random.Random(seed)
    k = len(vals)
    ms = sorted(st.mean(vals[rng.randrange(k)] for _ in range(k)) for _ in range(n))
    return st.mean(vals), ms[int(0.025 * n)], ms[int(0.975 * n)]


def paired_ci(a, b, n=2000, seed=0):
    d = [x - y for x, y in zip(a, b)]
    return boot_ci(d, n, seed)


def per_query(name, split="eval100"):
    e = ev(name, split)
    return e.get("per_query_ndcg10") if e else None


def rows_table(split):
    out = []
    for label, name, c in ROWS:
        if name is None:
            e = next((ev(n, split) for _, n, _ in ROWS if n and ev(n, split)), None)
            pq = e["init_per_query_ndcg10"] if e else None
            if pq:
                out.append((label, c, boot_ci(pq), None, None, None))
            continue
        e = ev(name, split)
        if not e:
            continue
        pq = e.get("per_query_ndcg10")
        ci = boot_ci(pq) if pq else (e["ndcg10"], e["ndcg10"], e["ndcg10"])
        out.append((label, c, ci, e["recall10"], e["invalid_rate"], e["out_tokens_per_query"]))
    return out


class Svg:
    def __init__(self, w, h, title):
        self.w, self.h, self.p = w, h, []
        self.p.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif" font-size="12">')
        self.p.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
        self.text(w / 2, 22, title, 14, "middle", INK, "bold")

    def text(self, x, y, s, size=12, anchor="start", color=INK, weight="normal"):
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        self.p.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{s}</text>')

    def line(self, x1, y1, x2, y2, color=INK, width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{d}/>')

    def rect(self, x, y, w, h, color):
        self.p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{color}"/>')

    def circle(self, x, y, r, color):
        self.p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"/>')

    def end(self):
        return "\n".join(self.p + ["</svg>"])


def fig_ndcg(rows, path):
    left, right, top, rowh = 250, 40, 48, 30
    h = top + rowh * len(rows) + 40
    s = Svg(760, h, "NDCG@10(HotpotQA、100 件のスライディングウィンドウ)")
    lo, hi = 0.6, 0.9
    X = lambda v: left + (v - lo) / (hi - lo) * (s.w - left - right)
    for t in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        s.line(X(t), top - 6, X(t), top + rowh * len(rows), GRID)
        s.text(X(t), top + rowh * len(rows) + 16, f"{t:.2f}", 11, "middle", MUTE)
    for i, (label, c, (m, a, b), *_rest) in enumerate(rows):
        y = top + i * rowh + rowh / 2
        s.text(left - 10, y + 4, label, 12, "end")
        s.rect(left, y - 8, max(0, X(m) - left), 16, COL[c])
        s.line(X(a), y, X(b), y, INK, 1.5)
        s.line(X(a), y - 5, X(a), y + 5, INK, 1.5)
        s.line(X(b), y - 5, X(b), y + 5, INK, 1.5)
        s.text(X(b) + 6, y + 4, f"{m:.3f}", 11, "start", MUTE)
    s.text(s.w / 2, h - 6, "棒 = 平均、線 = 95% ブートストラップ信頼区間(質問を復元抽出)", 11, "middle", MUTE)
    path.write_text(s.end(), encoding="utf-8")


def fig_tokens(rows, path):
    pts = [(l, c, m[0], r[2]) for l, c, m, *r in rows if r and r[0] is not None]
    s = Svg(760, 400, "出力トークン数と NDCG@10(100 件のスライディング)")
    L, R, T, B = 70, 30, 50, 60
    xs = [p[3] for p in pts]
    xmax = max(xs) * 1.08
    ymin, ymax = 0.6, 0.9
    X = lambda v: L + (v / xmax) * (s.w - L - R)
    Y = lambda v: s.h - B - (v - ymin) / (ymax - ymin) * (s.h - T - B)
    for t in (0.6, 0.7, 0.8, 0.9):
        s.line(L, Y(t), s.w - R, Y(t), GRID)
        s.text(L - 8, Y(t) + 4, f"{t:.1f}", 11, "end", MUTE)
    for t in range(0, int(xmax), 5000):
        s.line(X(t), T - 6, X(t), s.h - B, GRID)
        s.text(X(t), s.h - B + 16, f"{t:,}", 11, "middle", MUTE)
    for l, c, m, tok in pts:
        s.circle(X(tok), Y(m), 6, COL[c])
        s.text(X(tok) + 9, Y(m) + 4, l, 11, "start", INK)
    s.text(s.w / 2, s.h - 14, "質問あたりの出力トークン数(9 回の窓の合計)", 11, "middle", MUTE)
    path.write_text(s.end(), encoding="utf-8")


def tables(rows20, rows100):
    lines = ["# 集計表", "", "NDCG@10 は 95% ブートストラップ信頼区間つき。評価は HotpotQA の dev で、BM25 が正解 2 段落を 1・2 位に置けていない質問だけ。", ""]
    for title, rows in (("100 件のスライディングウィンドウ(窓 20、刻み 10)", rows100), ("20 件の 1 窓", rows20)):
        lines += [f"## {title}", "", "| 条件 | NDCG@10 | 95% 区間 | Recall@10 | 無効な出力(窓あたり) | 出力トークン/質問 |", "|---|---|---|---|---|---|"]
        for label, _c, (m, a, b), rec, inv, tok in rows:
            lines.append(f"| {label} | {m:.3f} | [{a:.3f}, {b:.3f}] | " + (f"{rec:.3f} | {inv:.2f} | {tok:,.0f} |" if rec is not None else "- | - | - |"))
        lines.append("")
    pairs = [("SFT 順位のみ", "sft_direct_pq", "SFT 推論(フィルタあり)", "sft_reason_t06"),
             ("SFT 推論(フィルタあり)", "sft_reason_t06", "SFT 推論(フィルタなし)", "sft_reason_nofilter_t06"),
             ("GRPO マルチビュー", "grpo_multi_t06", "GRPO NDCG のみ", "grpo_ndcg_t06"),
             ("GRPO マルチビュー", "grpo_multi_t06", "SFT 推論(フィルタあり)", "sft_reason_t06"),
             ("GRPO NDCG のみ", "grpo_ndcg_t06", "SFT 推論(フィルタあり)", "sft_reason_t06"),
             ("ゼロショット 教師 8B", "zs_teacher_t06", "SFT 順位のみ", "sft_direct_pq"),
             ("3B SFT 順位のみ", "sft_direct_3b", "3B SFT 推論(2 エポック分)", "sft_reason_3b_ep2_ev"),
             ("3B SFT 順位のみ", "sft_direct_3b", "1.5B SFT 順位のみ", "sft_direct_pq"),
             ("3B SFT 推論(2 エポック分)", "sft_reason_3b_ep2_ev", "1.5B SFT 推論(フィルタあり)", "sft_reason_t06")]
    lines += ["## 対応ありの差(100 件、同じ質問での NDCG@10 の差と 95% 区間)", "", "| A | B | A − B | 95% 区間 |", "|---|---|---|---|"]
    for la, a, lb, b in pairs:
        pa, pb = per_query(a), per_query(b)
        if pa and pb:
            m, lo, hi = paired_ci(pa, pb)
            lines.append(f"| {la} | {lb} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    # GRPO の 2 つ目のシード
    extra = [("grpo_multi_s1", "GRPO マルチビュー(seed 1)"), ("grpo_ndcg_s1", "GRPO NDCG のみ(seed 1)")]
    got = [(l, ev(n)) for n, l in extra if ev(n)]
    if got:
        lines += ["", "## GRPO の 2 つ目のシード(100 件)", "", "| 条件 | NDCG@10 | 無効な出力 |", "|---|---|---|"]
        for l, e in got:
            lines.append(f"| {l} | {e['ndcg10']:.3f} | {e['invalid_rate']:.2f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    r100, r20 = rows_table("eval100"), rows_table("eval20")
    fig_ndcg(r100, OUT / "ndcg.svg")
    fig_tokens(r100, OUT / "tokens.svg")
    (OUT / "tables.md").write_text(tables(r20, r100), encoding="utf-8")
    print("wrote", OUT)
