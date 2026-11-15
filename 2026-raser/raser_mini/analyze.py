"""質問ごとの結果(results/*.jsonl)から、ルーターの学習・評価と、論文の主張ごとの検証を行う(CPU だけ)。

  python raser_mini/analyze.py            # docs/tables.md、results/summary.json、docs/*.svg を作る

ルーターは、LLM ごとに、3 つのデータセットを混ぜた全質問に対して、質問単位の 5 分割交差検証で評価する(評価側の予測だけを使う)。
区間は、質問の復元抽出(2000 回)による 95% 区間。経路の比較は、同じ再標本の上で差を取る(対応あり)。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
import raser_core as rc  # noqa: E402

RES = Path(os.environ.get("RASER_RESULTS", HERE / "results"))      # 試験用に、環境変数で出力先を変えられる
DOCS = Path(os.environ.get("RASER_DOCS", HERE.parent / "docs"))
DATASETS = ["musique", "2wiki", "hotpotqa"]
MODELS = {"q3b": "Qwen2.5-3B", "q7b": "Qwen2.5-7B"}
THETA, BUDGET, TAU, B = 0.20, 0.60, 0.1, 2000
SEEDS = range(5)


def load(tag):
    rows = []
    for ds in DATASETS:
        p = RES / f"{tag}_{ds}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    F1 = np.array([[r[k]["f1"] for k in rc.ROUTES] for r in rows])
    T = np.array([[r[k]["tokens"] for k in rc.ROUTES] for r in rows], float)
    return rows, rc.feature_matrix(rows), F1, T, np.array([r["dataset"] for r in rows]), np.array([f"{r['dataset']}/{r['id']}" for r in rows])


def fmt(t, d=3):
    p, lo, hi = t
    return f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def ci(fn, n, d=3, B=B):
    return fmt(rc.bootstrap(fn, n, B=B), d)


def raser3_route(o, budget=BUDGET):
    return rc.cost_aware_argmax(o["pred3"], o["cbar"], o["lam"][budget])


def raser2_route(o, theta=THETA):
    return (o["p2"] >= theta).astype(int)          # 0=one-shot、1=PRUNE


def sk_clf(kind):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {"logreg": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "mlp32": lambda s: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=s)),
            "histgb": lambda s: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, max_iter=100, random_state=s)}[kind]


def sk_reg(kind):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {"ridge": lambda s: make_pipeline(StandardScaler(), Ridge()),
            "mlp32": lambda s: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,), max_iter=800, random_state=s)),
            "histgb": lambda s: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.1, max_iter=100, random_state=s)}[kind]


def main():
    out, md = {}, []
    P = md.append
    data = {t: load(t) for t in MODELS if any((RES / f"{t}_{d}.jsonl").exists() for d in DATASETS)}
    for tag, (rows, X, F1, T, ds, gid) in data.items():
        name = MODELS[tag]
        n = len(rows)
        o = out[tag] = {}
        P(f"\n## {name}(質問 {n} 件: " + ", ".join(f"{d} {int((ds == d).sum())}" for d in DATASETS) + ")\n")

        # ---- 経路ごとの F1 とトークン
        P("### 経路ごとの平均 F1 と平均トークン(区間は 95% ブートストラップ)\n")
        P("| データセット | 経路 | F1 | トークン | トークン比(対 one-shot) |\n|---|---|---|---|---|")
        for d in DATASETS + ["全体"]:
            m = (ds == d) if d != "全体" else np.ones(n, bool)
            f, t = F1[m], T[m]
            for j, r in enumerate(rc.ROUTES):
                ratio = ci(lambda i: t[i, j].sum() / t[i, 0].sum(), len(f), 2) if j else "1"
                P(f"| {d} | {r} | {ci(lambda i: f[i, j].mean(), len(f))} | {t[:, j].mean():.0f} | {ratio} |")
            o.setdefault("routes", {})[d] = {"f1": f.mean(0).tolist(), "tokens": t.mean(0).tolist()}

        # ---- 回復可能性(Table 1 に相当)
        P("\n### 回復可能性(論文 Table 1 に相当。Self-Ask* は実装していない)\n")
        P("| データセット | one-shot が正解 | どの経路でも直らない | わずかな改善 | 追加検索が効く(>=0.1) |\n|---|---|---|---|---|")
        for d in DATASETS + ["全体"]:
            m = (ds == d) if d != "全体" else np.ones(n, bool)
            f = F1[m]
            cells = []
            for key in ("one_shot_correct", "no_method_helps", "small_gain", "bridge_helps"):
                cells.append(ci(lambda i: rc.recoverability(f[i])[key], len(f), 3, 1000))
            P(f"| {d} | " + " | ".join(cells) + " |")
            o.setdefault("recov", {})[d] = rc.recoverability(f)
        lab = rc.bridgeable_label(F1[:, 1], F1[:, 0], TAU)
        o["bridgeable_rate"] = float(lab.mean())
        P(f"\nPRUNE が one-shot を τ={TAU} より改善した質問(RASER-2 のラベル y=1)の割合: {ci(lambda i: lab[i].mean(), n)}\n")

        # ---- ルーター(主設定)
        cv = rc.cv_router(X, F1, T, gid, tau=TAU, seed=0, budgets=(BUDGET,))
        r2, r3 = raser2_route(cv), raser3_route(cv)
        orc = rc.oracle_route(F1, T)
        P(f"### ルーター(θ={THETA}、予算={BUDGET}、5 分割交差検証)\n")
        P("| 範囲 | 方式 | F1 | トークン | 対 always-PRUNE の F1 差 | 対 always-PRUNE のトークン比 | 経路の比率(one/prune/ircot) |\n|---|---|---|---|---|---|---|")
        for d in DATASETS + ["全体"]:
            m = np.where(ds == d)[0] if d != "全体" else np.arange(n)
            for nm, rt in (("one-shot", np.zeros(n, int)), ("always-PRUNE", np.ones(n, int)), ("always-IRCOT*", np.full(n, 2)), ("RASER-2", r2), ("RASER-3", r3), ("oracle", orc)):
                f, t, r = F1[m], T[m], rt[m]
                a = np.arange(len(m))
                df = ci(lambda i: f[i, :][np.arange(len(i)), r[i]].mean() - f[i, 1].mean(), len(m)) if nm not in ("always-PRUNE",) else "-"
                tr = ci(lambda i: t[i][np.arange(len(i)), r[i]].sum() / t[i, 1].sum(), len(m), 2) if nm not in ("always-PRUNE",) else "1"
                st = rc.route_stats(r, f, t)
                P(f"| {d} | {nm} | {ci(lambda i: f[i][np.arange(len(i)), r[i]].mean(), len(m))} | {st['tokens']:.0f} | {df} | {tr} | "
                  + "/".join(f"{x * 100:.0f}" for x in st["mix"]) + " |")
                o.setdefault("router", {}).setdefault(d, {})[nm] = st
        # 同じ割合を無作為に昇格させる場合(ルーターの価値の確認)
        P("\n### 無作為に昇格する場合との比較(全体。昇格の割合は RASER-2 と同じ)\n")
        p_esc = float(r2.mean())
        rnd_f1 = (1 - p_esc) * F1[:, 0].mean() + p_esc * F1[:, 1].mean()
        rnd_tk = (1 - p_esc) * T[:, 0].mean() + p_esc * T[:, 1].mean()
        diff = rc.bootstrap(lambda i: F1[i][np.arange(len(i)), r2[i]].mean() - ((1 - p_esc) * F1[i, 0].mean() + p_esc * F1[i, 1].mean()), n, B=B)
        P(f"昇格割合 {p_esc * 100:.1f}%。無作為の期待 F1 {rnd_f1:.3f}、トークン {rnd_tk:.0f}。RASER-2 の F1 - 無作為の期待 F1 = {fmt(diff)}\n")
        o["random_esc"] = {"p": p_esc, "f1": rnd_f1, "diff": [float(x) for x in diff]}
        o["raser2_vs_random"] = [float(x) for x in diff]

        # ---- 同じトークン(always-PRUNE の 45%)で比べる
        target = 0.45 * T[:, 1].mean()
        rm2, rm3 = rc.match_tokens_r2(cv["p2"], T, target), rc.match_tokens_r3(cv["pred3"], cv["cbar"], T, target)
        t0, t1 = T[:, 0].mean(), T[:, 1].mean()
        P("### 同じトークンでの比較(平均トークンが always-PRUNE の 45% になるよう、RASER-2 は上位から上げ、RASER-3 は lambda を選ぶ。全体)\n")
        P("無作為 = 同じトークンになるよう、無作為に選んだ質問だけを PRUNE に上げる場合の期待 F1。論文の「41〜49%」に対応する比較。\n")
        P("| 方式 | 実際のトークン比(対 PRUNE) | F1 | F1 - always-PRUNE | F1 - 無作為 |\n|---|---|---|---|---|")
        o["matched45"] = {}
        for nm, rt in (("RASER-2", rm2), ("RASER-3", rm3)):
            ptk = (T[np.arange(n), rt].mean() - t0) / (t1 - t0)
            rnd = lambda i: F1[i, 0].mean() + ptk * (F1[i, 1].mean() - F1[i, 0].mean())
            rf = lambda i: F1[i][np.arange(len(i)), rt[i]].mean()
            P(f"| {nm} | {T[np.arange(n), rt].mean() / t1:.2f} | {ci(rf, n)} | {ci(lambda i: rf(i) - F1[i, 1].mean(), n)} | {ci(lambda i: rf(i) - rnd(i), n)} |")
            o["matched45"][nm] = {"f1": float(rf(np.arange(n))), "ratio": float(T[np.arange(n), rt].mean() / t1), "vs_random": [float(x) for x in rc.bootstrap(lambda i: rf(i) - rnd(i), n, B=B)],
                                  "vs_prune": [float(x) for x in rc.bootstrap(lambda i: rf(i) - F1[i, 1].mean(), n, B=B)]}
        P(f"\n参考: one-shot F1 {F1[:, 0].mean():.3f}、always-PRUNE F1 {F1[:, 1].mean():.3f}、oracle F1 {F1[np.arange(n), orc].mean():.3f}。"
          f"RASER-3(主設定、予算 {BUDGET})が one-shot から oracle までの差のうち取った割合: {(F1[np.arange(n), r3].mean() - F1[:, 0].mean()) / (F1[np.arange(n), orc].mean() - F1[:, 0].mean()):.2f}\n")

        # ---- 交差検証の種による揺れ
        s2, s3 = [], []
        for s in SEEDS:
            c = rc.cv_router(X, F1, T, gid, tau=TAU, seed=s, budgets=(BUDGET,))
            s2.append(rc.route_stats(raser2_route(c), F1, T)); s3.append(rc.route_stats(raser3_route(c), F1, T))
        P("### 交差検証の分割の種(0〜4)による揺れ(全体)\n")
        P(f"RASER-2: F1 {np.mean([x['f1'] for x in s2]):.3f} ± {np.std([x['f1'] for x in s2]):.3f}、トークン {np.mean([x['tokens'] for x in s2]):.0f}。"
          f"RASER-3: F1 {np.mean([x['f1'] for x in s3]):.3f} ± {np.std([x['f1'] for x in s3]):.3f}、トークン {np.mean([x['tokens'] for x in s3]):.0f}\n")
        o["seeds"] = {"r2_f1": [x["f1"] for x in s2], "r3_f1": [x["f1"] for x in s3]}

        # ---- データセットごとに学習する場合
        P("### データセットごとに学習した場合(各 500 問で 5 分割。全データを混ぜた主設定との比較用)\n")
        P("| データセット | RASER-2 F1 | RASER-2 トークン | RASER-3 F1 | RASER-3 トークン | always-PRUNE F1 | トークン |\n|---|---|---|---|---|---|---|")
        for d in DATASETS:
            m = np.where(ds == d)[0]
            c = rc.cv_router(X[m], F1[m], T[m], gid[m], tau=TAU, seed=0, budgets=(BUDGET,))
            a, b = rc.route_stats(raser2_route(c), F1[m], T[m]), rc.route_stats(raser3_route(c), F1[m], T[m])
            P(f"| {d} | {a['f1']:.3f} | {a['tokens']:.0f} | {b['f1']:.3f} | {b['tokens']:.0f} | {F1[m, 1].mean():.3f} | {T[m, 1].mean():.0f} |")

        # ---- 特徴量の重要度とアブレーション
        P("\n### 特徴量の重要度(GBM を全質問で学習。論文 Table 5 に相当)\n")
        y = rc.bridgeable_label(F1[:, 1], F1[:, 0], TAU)
        imp2 = rc.make_gbm_clf(0).fit(X, y).feature_importances_
        imp3 = np.mean([rc.make_gbm_reg(0).fit(X, F1[:, j]).feature_importances_ for j in range(3)], axis=0)
        P("| 特徴量 | RASER-2(分類器) | RASER-3(3 回帰器の平均) |\n|---|---|---|")
        for k, f in enumerate(rc.FEATURES):
            P(f"| {f} | {imp2[k]:.2f} | {imp3[k]:.2f} |")
        o["importance"] = {"r2": imp2.tolist(), "r3": imp3.tolist()}
        P(f"\nscore_gap + score_top1 の合計: RASER-2 {imp2[3] + imp2[4]:.2f}、RASER-3 {imp3[3] + imp3[4]:.2f}(論文は 0.73)\n")

        P("### 特徴量のアブレーション(1 つ抜く / 1 つだけ使う。全体、F1 / トークン)\n")
        P("F1@45% = 平均トークンを always-PRUNE の 45% にそろえたときの F1(θ や lambda の違いによる費用の差を除くため)。\n")
        P("| 使う特徴量 | RASER-2 F1 | RASER-2 トークン | RASER-2 F1@45% | RASER-3 F1 | RASER-3 トークン | RASER-3 F1@45% |\n|---|---|---|---|---|---|---|")
        sets = [("全 6 個", list(range(6)))] + [(f"{f} を抜く", [j for j in range(6) if j != k]) for k, f in enumerate(rc.FEATURES)] \
            + [(f"{f} だけ", [k]) for k, f in enumerate(rc.FEATURES)] + [("score_gap + score_top1", [3, 4]), ("confidence + ans_len", [0, 1]), ("bridge_cues + qtype", [2, 5]),
                                                                                                    ("コンペで作れる 4 つ(ans_len, score_gap, score_top1, qtype)", [1, 3, 4, 5])]
        o["ablation"] = {}
        for nm, cols in sets:
            rs2, rs3, m2, m3 = [], [], [], []
            for s in range(3):
                c = rc.cv_router(X, F1, T, gid, tau=TAU, seed=s, budgets=(BUDGET,), cols=cols)
                rs2.append(rc.route_stats(raser2_route(c), F1, T)); rs3.append(rc.route_stats(raser3_route(c), F1, T))
                m2.append(rc.route_stats(rc.match_tokens_r2(c["p2"], T, target), F1, T)["f1"])
                m3.append(rc.route_stats(rc.match_tokens_r3(c["pred3"], c["cbar"], T, target), F1, T)["f1"])
            a = [np.mean([x[k] for x in rs2]) for k in ("f1", "tokens")] + [np.mean(m2)]
            b = [np.mean([x[k] for x in rs3]) for k in ("f1", "tokens")] + [np.mean(m3)]
            o["ablation"][nm] = {"r2": a, "r3": b}
            P(f"| {nm} | {a[0]:.3f} | {a[1]:.0f} | {a[2]:.3f} | {b[0]:.3f} | {b[1]:.0f} | {b[2]:.3f} |")
        P("\n(分割の種 0〜2 の平均)\n")

        # ---- 分類器のアブレーション
        P("### 分類器のアブレーション(全体。論文 Table 6/7 に相当。XGBoost などの代わりに HistGradientBoosting を使った)\n")
        P("| ルーター | モデル | F1 | トークン | 経路の比率 |\n|---|---|---|---|---|")
        o["clf_ablation"] = {}
        for kind in ("gbm", "logreg", "mlp32", "histgb"):
            mk = rc.make_gbm_clf if kind == "gbm" else sk_clf(kind)
            c = rc.cv_router(X, F1, T, gid, tau=TAU, seed=0, make_clf=mk)
            st = rc.route_stats(raser2_route(c), F1, T)
            o["clf_ablation"]["r2_" + kind] = st
            P(f"| RASER-2 | {kind} | {st['f1']:.3f} | {st['tokens']:.0f} | " + "/".join(f"{x * 100:.0f}" for x in st["mix"][:2]) + " |")
        for kind in ("gbm", "ridge", "mlp32", "histgb"):
            mk = rc.make_gbm_reg if kind == "gbm" else sk_reg(kind)
            c = rc.cv_router(X, F1, T, gid, tau=TAU, seed=0, budgets=(BUDGET,), make_reg=mk)
            st = rc.route_stats(raser3_route(c), F1, T)
            o["clf_ablation"]["r3_" + kind] = st
            P(f"| RASER-3 | {kind} | {st['f1']:.3f} | {st['tokens']:.0f} | " + "/".join(f"{x * 100:.0f}" for x in st["mix"]) + " |")

        # ---- 閾値・予算・lambda の曲線
        P("\n### 閾値 θ の掃引(RASER-2、全体。論文 Table 8 に相当)\n")
        P("| θ | F1 | トークン | 昇格 % |\n|---|---|---|---|")
        o["theta"] = {}
        for th in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            st = rc.route_stats(raser2_route(cv, th), F1, T)
            o["theta"][th] = st
            P(f"| {th:.2f} | {st['f1']:.3f} | {st['tokens']:.0f} | {st['mix'][1] * 100:.1f} |")
        buds = (0.33, 0.5, 0.6, 0.75, 1.0)
        cvb = rc.cv_router(X, F1, T, gid, tau=TAU, seed=0, budgets=buds)
        P("\n### 予算の掃引(RASER-3、全体。論文 Table 9 に相当。予算は、訓練分割の always-IRCOT* の平均トークンに対する割合)\n")
        P("| 予算 | F1 | トークン | 経路の比率(one/prune/ircot %) |\n|---|---|---|---|")
        o["budget"] = {}
        for b in buds:
            st = rc.route_stats(raser3_route(cvb, b), F1, T)
            o["budget"][b] = st
            P(f"| {b:.2f} | {st['f1']:.3f} | {st['tokens']:.0f} | " + "/".join(f"{x * 100:.0f}" for x in st["mix"]) + " |")
        curve = []
        for lam in rc.default_lam_grid():
            st = rc.route_stats(rc.cost_aware_argmax(cv["pred3"], cv["cbar"], lam), F1, T)
            curve.append((float(lam), st["f1"], st["tokens"], st["mix"]))
        o["lam_curve"] = curve
        P("\n### lambda の掃引(RASER-3、全体。交差検証の予測を使い、lambda は全分割で共通)\n")
        P("| lambda | F1 | トークン | 経路の比率 |\n|---|---|---|---|")
        for lam, f, tk, mx in curve[::6]:
            P(f"| {lam:.1e} | {f:.3f} | {tk:.0f} | " + "/".join(f"{x * 100:.0f}" for x in mx) + " |")

    # ---- 小さい LLM ほど反復検索の利益が大きいか(同じ質問に対する 2 つの LLM の比較)
    if "q3b" in data and "q7b" in data:
        a, b = data["q3b"], data["q7b"]
        assert (a[5] == b[5]).all()
        P("\n## 小さい LLM ほど追加検索の利益が大きいか(3B と 7B の、利益の差。対応あり)\n")
        P("利益 = その経路の F1 - one-shot の F1。差 = 3B の利益 - 7B の利益。区間が 0 を含まなければ、差あり。\n")
        P("| データセット | 経路 | 3B の利益 | 7B の利益 | 差(3B - 7B) |\n|---|---|---|---|---|")
        out["claim_c"] = {}
        for d in DATASETS + ["全体"]:
            m = np.where(a[4] == d)[0] if d != "全体" else np.arange(len(a[5]))
            for j, r in ((1, "PRUNE"), (2, "IRCOT*")):
                ga = lambda i: (a[2][m][i, j] - a[2][m][i, 0]).mean()
                gb = lambda i: (b[2][m][i, j] - b[2][m][i, 0]).mean()
                diff = rc.bootstrap(lambda i: ga(i) - gb(i), len(m), B=B)
                P(f"| {d} | {r} | {ci(ga, len(m))} | {ci(gb, len(m))} | {fmt(diff)} |")
                out["claim_c"][f"{d}/{r}"] = [float(x) for x in diff]
        P("\n### 追加検索を使う必要が出る頻度(F1 が 0.1 以上改善する質問の割合)\n")
        for tag in ("q3b", "q7b"):
            rr = rc.recoverability(data[tag][2])
            P(f"- {MODELS[tag]}: {rr['bridge_helps'] * 100:.1f}%(one-shot が正解 {rr['one_shot_correct'] * 100:.1f}%)")

    DOCS.mkdir(exist_ok=True)
    (DOCS / "tables.md").write_text("# 縮小再現の結果表(analyze.py が生成)\n" + "\n".join(md) + "\n", encoding="utf-8")
    (RES / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    make_figures(out, data)
    print("tables.md と summary.json を書いた")


def make_figures(out, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = {t: out[t] for t in data}
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), squeeze=False)
    for ax, (tag, o) in zip(axes[0], models.items()):
        a = o["routes"]["全体"]
        ax.plot([c[2] for c in o["lam_curve"]], [c[1] for c in o["lam_curve"]], "-", color="#c2410c", label="RASER-3 (sweep lambda)")
        ax.plot([v["tokens"] for v in o["theta"].values()], [v["f1"] for v in o["theta"].values()], "o-", color="#1d4ed8", ms=3, label="RASER-2 (sweep theta)")
        for j, (nm, mk) in enumerate((("one-shot", "s"), ("PRUNE", "^"), ("IRCOT*", "D"))):
            ax.scatter(a["tokens"][j], a["f1"][j], marker=mk, color="k", zorder=5)
            ax.annotate(nm, (a["tokens"][j], a["f1"][j]), textcoords="offset points", xytext=(5, -10), fontsize=8)
        ax.scatter(o["router"]["全体"]["oracle"]["tokens"], o["router"]["全体"]["oracle"]["f1"], marker="*", color="#15803d", s=90, zorder=5)
        ax.annotate("oracle", (o["router"]["全体"]["oracle"]["tokens"], o["router"]["全体"]["oracle"]["f1"]), textcoords="offset points", xytext=(5, 3), fontsize=8)
        ax.set_title(MODELS[tag]); ax.set_xlabel("mean tokens per question"); ax.set_ylabel("mean F1"); ax.grid(alpha=.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(DOCS / "tradeoff.svg"); plt.close(fig)


if __name__ == "__main__":
    main()
