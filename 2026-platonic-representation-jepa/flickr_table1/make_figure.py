"""COCO のスケーリング結果(results/coco_*.json)から、サンプル数 × mutual 10-NN の図を SVG で書き出す。

標準ライブラリのみ。実行: python flickr_table1/make_figure.py
"""
import glob
import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "results" / "scaling_coco.svg"
COLORS = {"good": "#1f77b4", "bad": "#d62728", "ntp": "#2ca02c"}
LABELS = {"good": "good view", "bad": "bad view", "ntp": "Next Token Prediction"}


def load():
    runs = defaultdict(list)  # (variant, samples) -> [mutual_knn, ...]
    for f in glob.glob(str(HERE / "results" / "coco_*_seed[0-9]_s*.json")):
        m = re.search(r"coco_(\w+)_seed(\d)_s(\d+)\.json", f)
        if m and m.group(1) in COLORS:
            runs[(m.group(1), int(m.group(3)))].append(json.load(open(f))["mutual_knn"])
    return runs


def main():
    runs = load()
    W, H, L, R, T, B = 640, 400, 64, 24, 24, 56
    xmin, xmax, ymin, ymax = math.log10(3e4), math.log10(1e7), 0.0, 0.22
    X = lambda n: L + (math.log10(n) - xmin) / (xmax - xmin) * (W - L - R)
    Y = lambda v: H - B - (v - ymin) / (ymax - ymin) * (H - T - B)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>']
    for v in [0, .05, .10, .15, .20]:
        s.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#ddd"/>')
        s.append(f'<text x="{L-8}" y="{Y(v)+4:.1f}" text-anchor="end">{v:.2f}</text>')
    for e, n in [(5, 1e5), (5.48, 3e5), (6, 1e6), (6.48, 3e6), (7, 1e7)]:
        s.append(f'<line x1="{X(n):.1f}" x2="{X(n):.1f}" y1="{T}" y2="{H-B}" stroke="#eee"/>')
        s.append(f'<text x="{X(n):.1f}" y="{H-B+18}" text-anchor="middle">{n:,.0f}</text>')
    s.append(f'<text x="{(L+W-R)/2}" y="{H-12}" text-anchor="middle">学習で見たサンプル数 (対数軸)</text>')
    s.append(f'<text transform="translate(16,{(T+H-B)/2}) rotate(-90)" text-anchor="middle">mutual 10-NN (テスト 5,000 組)</text>')
    for v, c in COLORS.items():
        pts = sorted((n, vals) for (vv, n), vals in runs.items() if vv == v)
        if not pts:
            continue
        path = " ".join(f"{X(n):.1f},{Y(st.mean(vals)):.1f}" for n, vals in pts)
        s.append(f'<polyline points="{path}" fill="none" stroke="{c}" stroke-width="2"/>')
        for n, vals in pts:
            m = st.mean(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0
            if sd:
                s.append(f'<line x1="{X(n):.1f}" x2="{X(n):.1f}" y1="{Y(m-sd):.1f}" y2="{Y(m+sd):.1f}" stroke="{c}"/>')
            s.append(f'<circle cx="{X(n):.1f}" cy="{Y(m):.1f}" r="3.5" fill="{c}"/>')
    # 凡例
    for i, (v, c) in enumerate(COLORS.items()):
        y = T + 14 + i * 18
        s.append(f'<line x1="{L+12}" x2="{L+36}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2"/>')
        s.append(f'<text x="{L+42}" y="{y+4}">{LABELS[v]}</text>')
    s.append("</svg>")
    OUT.write_text("\n".join(s), encoding="utf-8")
    print("保存:", OUT)


if __name__ == "__main__":
    main()
