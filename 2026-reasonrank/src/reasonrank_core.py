"""ReasonRank(Liu et al., 2026)の縮小再現の中核。外部ライブラリに依存しない純 Python。

- プロンプトの組み立てと、出力(<think>…</think><answer>[3] > [1] > …</answer>)の解析
- 指標: NDCG@k、Recall@k、RBO。報酬: NDCG@10 のみ / NDCG@10 + φ·Recall@10 + γ·RBO(論文の multi-view ranking reward)
- スライディングウィンドウ(窓幅 w、刻み s、後ろから前へ)
- GRPO のグループ内正規化(advantage)
"""
from __future__ import annotations

import math
import re
from typing import Callable, Sequence

PHI, GAMMA, RBO_P = 0.2, 0.1, 0.9  # 論文の φ=0.2, γ=0.1。RBO の p は論文が値を明記していないので 0.9 とする


# ---------------------------------------------------------------- プロンプトと解析
def truncate_words(text: str, max_words: int = 160) -> str:
    w = text.split()
    return text if len(w) <= max_words else " ".join(w[:max_words]) + " …"


def build_prompt(query: str, passages: Sequence[str], reasoning: bool = True, max_words: int = 160) -> str:
    n = len(passages)
    body = "\n".join(f"[{i + 1}] {truncate_words(p, max_words)}" for i, p in enumerate(passages))
    if reasoning:
        how = ("First think step by step inside <think></think>, comparing what each passage contributes to answering the query. "
               "Then write the final ranking inside <answer></answer> using the identifiers, for example "
               "<answer>[2] > [1] > [3]</answer>. Every identifier must appear exactly once.")
    else:
        how = ("Write only the final ranking inside <answer></answer> using the identifiers, for example "
               "<answer>[2] > [1] > [3]</answer>. Every identifier must appear exactly once.")
    return (f"I will give you {n} passages, each marked with an identifier [ ]. Rank the passages by how useful they are "
            f"for answering the search query. A passage is useful if it contains, or helps to derive, part of the answer.\n\n"
            f"{body}\n\nSearch query: {query}\n\nRank all {n} passages, most useful first. {how}")


_ANS = re.compile(r"<answer>(.*?)</answer>", re.S)


def parse_ranking(text: str, n: int) -> tuple[list[int], bool]:
    """出力から順位(0 始まりのインデックスの並び)を取り出す。

    戻り値: (順位, 厳密に正しいか)。厳密に正しい = 1..n がちょうど 1 回ずつ。
    足りない識別子は元の順で後ろに足す(RankGPT と同じ扱い)。<answer> がなければ空の解析として元の順を返す。
    """
    m = _ANS.findall(text)
    if not m:
        return list(range(n)), False
    seq = [int(x) - 1 for x in re.findall(r"\[(\d+)\]", m[-1])]
    seen, order = set(), []
    for i in seq:
        if 0 <= i < n and i not in seen:
            seen.add(i)
            order.append(i)
    exact = len(seq) == n and len(order) == n
    order += [i for i in range(n) if i not in seen]
    return order, exact


def split_think(text: str) -> tuple[str, str]:
    """(思考の文字列, 答えの文字列)。"""
    t = re.search(r"<think>(.*?)</think>", text, re.S)
    a = _ANS.findall(text)
    return (t.group(1).strip() if t else ""), (a[-1].strip() if a else "")


# ---------------------------------------------------------------- 指標
def dcg(gains: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked: Sequence, rels: dict, k: int = 10) -> float:
    """ranked: 識別子の並び。rels: {識別子: 関連度}(載っていないものは 0)。"""
    ideal = sorted(rels.values(), reverse=True)
    best = dcg(ideal, k)
    return dcg([rels.get(x, 0.0) for x in ranked], k) / best if best > 0 else 0.0


def recall_at_k(ranked: Sequence, rels: dict, k: int = 10) -> float:
    rel = {x for x, v in rels.items() if v > 0}
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def rbo(ranked: Sequence, gold: Sequence, p: float = RBO_P) -> float:
    """Rank-Biased Overlap(論文の式 4 の打ち切り版)。同じ並びなら 1 になるよう (1 - p^n) で割る。"""
    n = min(len(ranked), len(gold))
    if n == 0:
        return 0.0
    s, seen_a, seen_b, overlap = 0.0, set(), set(), 0
    for d in range(1, n + 1):
        a, b = ranked[d - 1], gold[d - 1]
        if a == b:
            overlap += 1
        else:
            overlap += (a in seen_b) + (b in seen_a)
        seen_a.add(a)
        seen_b.add(b)
        s += p ** (d - 1) * overlap / d
    return (1 - p) * s / (1 - p ** n)


def reward(ranked: Sequence, rels: dict, gold_list: Sequence, kind: str = "multi", exact: bool = True,
           phi: float = PHI, gamma: float = GAMMA) -> float:
    """kind: 'ndcg'(NDCG@10 のみ) / 'multi'(NDCG@10 + φ·Recall@10 + γ·RBO)。形式が崩れた出力は 0。"""
    if not exact:
        return 0.0
    r = ndcg_at_k(ranked, rels, 10)
    if kind == "ndcg":
        return r
    if kind == "multi":
        return r + phi * recall_at_k(ranked, rels, 10) + gamma * rbo(ranked, gold_list)
    raise ValueError(kind)


def group_advantages(rewards: Sequence[float], eps: float = 1e-6) -> list[float]:
    """GRPO: グループ内で平均を引き、標準偏差で割る。全員同じ報酬なら 0。"""
    n = len(rewards)
    mean = sum(rewards) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)
    if std < 1e-8:
        return [0.0] * n
    return [(r - mean) / (std + eps) for r in rewards]


# ---------------------------------------------------------------- スライディングウィンドウ
def window_starts(n: int, w: int = 20, s: int = 10) -> list[int]:
    """後ろから前へ。n=100, w=20, s=10 なら [80, 70, …, 0] の 9 回。n <= w なら [0]。"""
    if n <= w:
        return [0]
    out, start = [], n - w
    while True:
        out.append(start)
        if start == 0:
            break
        start = max(0, start - s)
    return out


def sliding_rerank(ids: list, rank_fn: Callable[[list], list[int]], w: int = 20, s: int = 10) -> list:
    """rank_fn(窓の識別子の並び) -> 窓内の順位(0 始まり)。全体の並べ替え結果を返す。"""
    ids = list(ids)
    for st in window_starts(len(ids), w, s):
        win = ids[st:st + w]
        order = rank_fn(win)
        ids[st:st + w] = [win[i] for i in order]
    return ids
