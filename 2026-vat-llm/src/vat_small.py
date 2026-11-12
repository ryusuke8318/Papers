"""オーストリア VAT 法での RAG 対 fine-tuning の向きを確かめる合成検証。

論文の主張 (Table 5 / Figure 5 の向き) を、numpy のみの確率シミュレーションでなぞる。
各アプローチを正誤生成過程としてモデル化する (仮定: 以下はすべて仮定の数値.
実測ではなく、順序の再現だけが目的)。

- plain: 事前学習のみ。小さい問題は解けるが複雑な chain は外す
- rag: 関連条文の検索に当たれば正解、外せば plain に後退
- ft: 学習済みパターンは seen タイプに強いが、held-out の chain に弱い
- rag_ft: 検索結果と学習パターンの衝突を起こし、両単独より悪化 (論文の核心)
- mocked: 検索 oracle (完全検索の上限の代理)
- majority / random: DEGENERATE 対照 (必ず失敗する側)
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import zlib


SEED = 0
N_TEXTBOOK = 74
COUNTRIES = ["AT", "DE", "IT", "HR", "DK", "FR"]

# 事例タイプの混合比 (textbook 74 件の代理。chain は少数難問の代理、仮定)
TYPE_MIX = {"goods": 0.30, "transport": 0.25, "b2b": 0.20, "b2c": 0.15, "chain": 0.10}
# plain のタイプ別正解率 (仮定)
PLAIN_P = {"goods": 0.80, "transport": 0.70, "b2b": 0.65, "b2c": 0.60, "chain": 0.30}
# rag の検索 hit 率 (仮定)
HIT_P = {"goods": 0.95, "transport": 0.95, "b2b": 0.90, "b2c": 0.90, "chain": 0.70}
# ft のタイプ別正解率。chain は held-out で崩壊 (仮定)
FT_P = {"goods": 0.92, "transport": 0.92, "b2b": 0.92, "b2c": 0.92, "chain": 0.35}
# rag_ft: 衝突による正解→誤答の反転率 (併用の adverse interaction の代理、仮定)
CONFLICT_FLIP = 0.12
# mocked: oracle 検索でも生成がたまに外す (仮定)
MOCKED_ERR = 0.03


def make_cases(n: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    types = list(TYPE_MIX)
    probs = np.array([TYPE_MIX[t] for t in types])
    t_idx = rng.choice(len(types), size=n, p=probs)
    # 正解国はタイプに弱く依存 (majority 対照が 3 割前後になるよう、仮定)
    gold = np.where(
        np.isin(t_idx, [0, 1]),
        rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2]),
        rng.integers(0, len(COUNTRIES), size=n),
    )
    return [types[i] for i in t_idx], gold


def predict(cases, gold, mode: str, seed: int = SEED):
    rng = np.random.default_rng(seed + zlib.crc32(mode.encode()) % 10_000)
    out = []
    for t, g in zip(cases, gold):
        if mode == "majority":
            out.append(0)  # 常に最頻国 AT。DEGENERATE
        elif mode == "random":
            out.append(int(rng.integers(0, len(COUNTRIES))))  # DEGENERATE
        elif mode == "plain":
            ok = rng.random() < PLAIN_P[t]
            out.append(int(g) if ok else int(rng.choice([c for c in range(len(COUNTRIES)) if c != g])))
        elif mode == "rag":
            if rng.random() < HIT_P[t]:
                out.append(int(g))
            else:
                ok = rng.random() < PLAIN_P[t]
                out.append(int(g) if ok else int(rng.choice([c for c in range(len(COUNTRIES)) if c != g])))
        elif mode == "ft":
            ok = rng.random() < FT_P[t]
            out.append(int(g) if ok else int(rng.choice([c for c in range(len(COUNTRIES)) if c != g])))
        elif mode == "rag_ft":
            if rng.random() < HIT_P[t]:
                pred = int(g)
            else:
                ok = rng.random() < PLAIN_P[t]
                pred = int(g) if ok else int(rng.choice([c for c in range(len(COUNTRIES)) if c != g]))
            if pred == g and rng.random() < CONFLICT_FLIP:
                pred = int(rng.choice([c for c in range(len(COUNTRIES)) if c != g]))
            out.append(pred)
        elif mode == "mocked":
            if rng.random() < MOCKED_ERR:
                out.append(int(rng.choice([c for c in range(len(COUNTRIES)) if c != g])))
            else:
                out.append(int(g))
        else:
            raise ValueError(mode)
    return np.array(out)


def acc(pred, gold) -> float:
    return float((pred == gold).mean())


def mcnemar_yates_p(b: int, c: int) -> float:
    """McNemar 検定 (Yates 補正)。自由度 1 のカイ二乗の上側確率。"""
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    return float(math.erfc(math.sqrt(chi2 / 2.0)))


def exp_table5():
    """Table 5 の向き: mocked >= rag > ft > rag_ft > plain >> degenerate。"""
    cases, gold = make_cases(N_TEXTBOOK)
    res = {m: acc(predict(cases, gold, m), gold)
           for m in ["mocked", "rag", "ft", "rag_ft", "plain", "majority", "random"]}
    print("[Table.5] textbook 風 74 件の正解率 (論文値との対照。絶対値は比べられない)")
    for m, paper in [("mocked", 0.9595), ("rag", 0.9324), ("ft", 0.8919),
                     ("rag_ft", 0.8649), ("plain", 0.6892), ("majority", None),
                     ("random", None)]:
        tag = ""
        if m in ("majority", "random"):
            tag = " <- DEGENERATE"
        print(f"[Table.5] {m:8s} small={res[m]:.3f} paper={paper} {tag}")
    order = ["mocked", "rag", "ft", "rag_ft", "plain", "majority", "random"]
    ok = all(res[a] >= res[b] for a, b in zip(order, order[1:]))
    print("[Table.5] 順序の一致:", ok)
    return res


def exp_chain():
    """chain 難問では ft が崩壊し rag が保つ (論文 §9.1 の向き)。"""
    cases, gold = make_cases(400, seed=1)
    is_chain = np.array([t == "chain" for t in cases])
    for m in ["ft", "rag"]:
        p = predict(cases, gold, m, seed=7)
        print(f"[chain] {m}: 全体={acc(p, gold):.3f} chain のみ={acc(p[is_chain], gold[is_chain]):.3f} "
              f"(n_chain={int(is_chain.sum())})")
    pf = predict(cases, gold, "ft", seed=7)
    pr = predict(cases, gold, "rag", seed=7)
    print("[chain] chain で ft < rag:", acc(pf[is_chain], gold[is_chain]) < acc(pr[is_chain], gold[is_chain]))


def exp_mcnemar():
    """n=20 の paired 比較では差が有意にならない (論文 Table 6, p=0.683 の向き)。"""
    cases, gold = make_cases(20, seed=2)
    pr = predict(cases, gold, "rag", seed=3) == gold
    pf = predict(cases, gold, "ft", seed=4) == gold
    b = int((pr & ~pf).sum())  # rag のみ正解
    c = int((~pr & pf).sum())  # ft のみ正解
    print(f"[McNemar] 20 件の内訳: 両正解={int((pr & pf).sum())} ragのみ={b} ftのみ={c} "
          f"両誤答={int((~pr & ~pf).sum())}")
    print(f"[McNemar] Yates 補正 p={mcnemar_yates_p(b, c):.3f} (論文 p=0.683。同様に n.s. か)")


if __name__ == "__main__":
    exp_table5()
    exp_chain()
    exp_mcnemar()
