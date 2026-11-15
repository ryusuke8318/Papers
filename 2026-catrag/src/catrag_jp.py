"""日本語の多段検索向けアダプタ: 記事名の言及リンクを辿る活性化の拡散で、候補チャンクを再順位づけする。

入力は「候補チャンク(id、title、text、リランカーの score)のリスト」と「記事タイトルの辞書」だけで、
外部のモデル・データには依存しない純関数。

方針(CatRAG と SA-RAG の考え方を、LLM なしの 2 段のリンク辿りに落としたもの)
1. 候補チャンクの score を [0, 1] に min-max 正規化し、記事(title)ごとの初期の活性化にする(最大値)。
2. チャンク本文に別の記事名が出てくるとき、その記事への有向リンクを張る。
   辺の重み = min(1, link_weight * sqrt(言及回数)) / max(1, 被リンク数) ** hub_alpha
   (被リンク数 = その記事に言及しているチャンクの数。ハブになる記事ほど弱くする)
3. 辺の重みに SA-RAG の再スケール (w - c)/(1 - c) をかけ、hops 段(既定 2)だけ活性化を拡散する。
4. 質問に記事名が出ているとき、その記事に小さな起点 eps を足す(Symbolic Anchoring の代用)。
5. 最終点 = 正規化 score + gamma * 拡散で受け取った活性化(初期の活性化は含めない)。降順で返す。

記事名の照合は部分文字列の一致。min_title_len 文字未満の記事名と、チャンク自身の記事名は無視する。
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np

try:  # テストやスクリプトから、src を path に入れて import する想定
    from catrag_core import spreading_activation
except ImportError:  # pragma: no cover
    from .catrag_core import spreading_activation


def find_mentions(text: str, titles: Iterable[str], min_len: int = 2, exclude: str | None = None) -> dict[str, int]:
    """text に現れる記事名と、その出現回数。他の記事名に含まれる短い記事名は、長い方が一致した範囲では数えない。"""
    cands = sorted({t for t in titles if len(t) >= min_len and t != exclude}, key=len, reverse=True)
    out: dict[str, int] = {}
    masked = text
    for t in cands:
        n = masked.count(t)
        if n:
            out[t] = n
            masked = masked.replace(t, "\0" * len(t))
    return out


def build_links(
    chunks: Sequence[Mapping], titles: Iterable[str], min_len: int = 2
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """記事 -> {言及先の記事: 回数} と、記事ごとの被リンク数(言及しているチャンクの数)を返す。"""
    titles = list(titles)
    links: dict[str, dict[str, int]] = {}
    indeg: dict[str, int] = {}
    for ch in chunks:
        ms = find_mentions(ch["text"], titles, min_len, exclude=ch["title"])
        d = links.setdefault(ch["title"], {})
        for t, n in ms.items():
            d[t] = d.get(t, 0) + n
            indeg[t] = indeg.get(t, 0) + 1
    return links, indeg


def rerank_with_links(
    chunks: Sequence[Mapping],
    titles: Iterable[str],
    question: str | None = None,
    *,
    hops: int = 2,
    link_weight: float = 0.8,
    hub_alpha: float = 0.5,
    c: float = 0.0,
    gamma: float = 0.5,
    eps: float = 0.0,
    n_seed: int = 5,
    min_title_len: int = 2,
    external_indegree: Mapping[str, int] | None = None,
) -> list[dict]:
    """chunks: [{'id', 'title', 'text', 'score'}, ...]。再順位づけした dict のリストを返す(元の入力は変更しない)。

    返す dict は入力の項目に、'norm_score'、'activation'(拡散で受け取った活性化)、'final' を足したもの。
    n_seed: 拡散の起点にする記事の数(正規化 score の上位の記事)。external_indegree: 全コーパスでの被リンク数が分かる場合に渡す。
    """
    if not chunks:
        return []
    titles = list(titles)
    scores = np.array([float(ch["score"]) for ch in chunks])
    lo, hi = scores.min(), scores.max()
    norm = (scores - lo) / (hi - lo) if hi > lo else np.ones_like(scores)

    art_score: dict[str, float] = {}
    for ch, s in zip(chunks, norm):
        art_score[ch["title"]] = max(art_score.get(ch["title"], 0.0), float(s))

    links, indeg = build_links(chunks, titles, min_title_len)
    if external_indegree:
        indeg = {t: max(indeg.get(t, 0), int(external_indegree.get(t, 0))) for t in set(indeg) | set(external_indegree)}

    # 辺: 記事 -> 言及先の記事
    index: dict[str, int] = {}
    for t in list(art_score) + [t for d in links.values() for t in d]:
        index.setdefault(t, len(index))
    adj: dict[int, list[tuple[int, float]]] = {}
    for a, d in links.items():
        for t, n in d.items():
            w = min(1.0, link_weight * math.sqrt(n)) / max(1, indeg.get(t, 1)) ** hub_alpha
            adj.setdefault(index[a], []).append((index[t], w))

    seeds = sorted(art_score, key=lambda t: -art_score[t])[:n_seed]
    init = {index[t]: art_score[t] for t in seeds}
    if question and eps > 0:
        for t in find_mentions(question, titles, min_title_len):
            i = index.setdefault(t, len(index))
            init[i] = init.get(i, 0.0) + eps
    act = spreading_activation(adj, init, c=c, max_hops=hops, cap=2.0)
    received = {i: a - init.get(i, 0.0) for i, a in act.items()}

    out = []
    for ch, s in zip(chunks, norm):
        a = max(0.0, received.get(index[ch["title"]], 0.0)) if ch["title"] in index else 0.0
        d = dict(ch)
        d.update({"norm_score": float(s), "activation": float(a), "final": float(s + gamma * a)})
        out.append(d)
    out.sort(key=lambda d: -d["final"])
    return out
