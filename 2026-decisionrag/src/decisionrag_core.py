"""DecisionRAG(Wan et al., 2026)の中核。外部ライブラリに依存しない純 Python。

- コーパスの木(根 → … → 段落)と、判定モデル(Judge)のインターフェース
- Algorithm 1: 木をたどる検索(各段階で並列の二値判定 → 上位 k 個の枝だけ残す → 最後に段落を採点)
- 共有文脈の請求(request)への分割(1 つの請求に入る候補数の上限)
- 任意の複数選択(multiple-choice)による上位 m 件の並べ替え、検索器の候補の追加(+ dense candidates)
- 指標(R@k、Hit@1、nDCG@k)、逐次の判定回数と遅延の見積もり(式 4)

判定モデルの中身(Qwen3 に採点用の head を付けたもの)は decisionrag_mini/judge.py にある。ここは、判定モデルを「候補の列を受けて確率を返す関数」として扱う。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


# ---------------------------------------------------------------- 木
@dataclass
class Node:
    id: str
    text: str                                   # 判定モデルに見せる短い説明(段落なら本文、文書ならタイトル、クラスタなら要約)
    level: int = 0                              # 根が 0。段落(葉)が最も深い
    children: list["Node"] = field(default_factory=list)
    parent: "Node | None" = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return not self.children


def build_tree(spec: dict, _level: int = 0, _parent: "Node | None" = None) -> Node:
    """spec = {"id", "text", "children": [spec, …]} から木を作る。"""
    n = Node(id=spec["id"], text=spec.get("text", ""), level=_level, parent=_parent)
    n.children = [build_tree(c, _level + 1, n) for c in spec.get("children", [])]
    return n


def depth(root: Node) -> int:
    """根から葉までの段数(根だけなら 0)。木は一様な深さを仮定する。"""
    d, n = 0, root
    while n.children:
        n, d = n.children[0], d + 1
    return d


def leaves(node: Node) -> list[Node]:
    if node.is_leaf:
        return [node]
    out: list[Node] = []
    for c in node.children:
        out += leaves(c)
    return out


# ---------------------------------------------------------------- 判定モデルのインターフェース
class Judge(Protocol):
    def binary(self, query: str, candidates: Sequence[Node]) -> list[float]:
        """候補すべてを同じ文脈(共有文脈)に入れ、各候補が関連する確率 P(Yes) を返す(合計は 1 にならない)。"""

    def choice(self, query: str, candidates: Sequence[Node]) -> list[float]:
        """候補を競合する選択肢として 1 つの判断で比べ、確率分布(合計 1)を返す。"""


def split_requests(items: Sequence, size: int | None) -> list[list]:
    """候補を、1 請求あたり size 件以下の請求に分ける。size が None なら 1 請求。"""
    items = list(items)
    if not size or size >= len(items):
        return [items]
    return [items[i:i + size] for i in range(0, len(items), size)]


@dataclass
class Trace:
    """1 回の検索で何が行われたか(遅延と費用の見積もり用)。"""
    stages: list[dict] = field(default_factory=list)

    def add(self, kind: str, level: int, n_candidates: int, n_requests: int) -> None:
        self.stages.append({"kind": kind, "level": level, "candidates": n_candidates, "requests": n_requests})

    @property
    def sequential_stages(self) -> int:
        """逐次に実行される段階の数。1 段の中の請求は並列に出せるので、1 と数える。"""
        return len(self.stages)

    @property
    def judgments(self) -> int:
        return sum(s["candidates"] for s in self.stages)


def judge_pooled(judge: Judge, query: str, cands: Sequence[Node], request_size: int | None) -> tuple[list[float], int]:
    """候補を請求に分けて採点し、スコアを元の順にまとめる。戻り値: (スコア, 請求数)。"""
    chunks = split_requests(cands, request_size)
    scores: list[float] = []
    for ch in chunks:
        scores += list(judge.binary(query, ch))
    return scores, len(chunks)


def top_indices(scores: Sequence[float], k: int) -> list[int]:
    """スコアの降順で上位 k 個の添字(同点は元の順)。"""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


# ---------------------------------------------------------------- Algorithm 1
@dataclass
class NavConfig:
    budgets: Sequence[int]                    # k_1, …, k_{L-1}: 各段で残す枝の数(葉の 1 つ上の段までに使う)
    r: int = 20                               # 返す段落の数
    m: int = 0                                # 複数選択で並べ替える上位の数(0 なら使わない)
    request_size: int | None = None           # 1 請求に入れる候補数の上限(None は分割しない)
    weight_by_parent: bool = False            # 段落のスコアに、親の文書のスコアを掛ける(論文の Natural Questions)
    promote_only: bool = False                # 複数選択で、最も確率の高い 1 件だけを 1 位にする(論文の MTRAG)


def navigate(root: Node, query: str, judge: Judge, cfg: NavConfig, extra_candidates: Sequence[Node] = ()) -> tuple[list[Node], Trace]:
    """木をたどって、関連する段落を上位から返す。

    各段で、残っている候補の説明を 1 つの共有文脈に入れて並列に二値判定し、上位 k 個の枝の子だけを次の段へ進める。
    落とした枝は、この問いでは二度と見ない(刈り込みは不可逆)。最後の段で段落を採点し、上位 r 件を取る。
    extra_candidates は、検索器の上位などを最終の候補に足す(論文の + dense candidates)。
    """
    trace = Trace()
    L = depth(root)
    if L == 0:
        return [root], trace
    if len(cfg.budgets) < L - 1:
        raise ValueError(f"budgets は {L - 1} 個要る(段の数 {L} - 1)")
    cands = list(root.children)
    weight: dict[str, float] = {}               # 段落 id -> 親の文書のスコア(weight_by_parent 用)
    for lvl in range(1, L):
        scores, n_req = judge_pooled(judge, query, cands, cfg.request_size)
        trace.add("binary", lvl, len(cands), n_req)
        keep = top_indices(scores, cfg.budgets[lvl - 1])
        if lvl == L - 1:
            for i in keep:
                for ch in cands[i].children:
                    weight[ch.id] = scores[i]
        cands = [ch for i in keep for ch in cands[i].children]
    # 最後の段: 段落の採点。追加の候補は、まだ入っていないものだけ足す
    seen = {c.id for c in cands}
    pool = cands + [e for e in extra_candidates if e.id not in seen]
    scores, n_req = judge_pooled(judge, query, pool, cfg.request_size)
    trace.add("binary", L, len(pool), n_req)
    if cfg.weight_by_parent:
        scores = [s * weight.get(c.id, 1.0) for s, c in zip(scores, pool)]
    order = top_indices(scores, cfg.r)
    ranked = [pool[i] for i in order]
    # 任意の複数選択: 上位 m 件を競合する選択肢として比べ直す
    m = min(cfg.m, len(ranked))
    if m >= 2:
        probs = list(judge.choice(query, ranked[:m]))
        trace.add("choice", L, m, 1)
        if cfg.promote_only:
            best = max(range(m), key=lambda i: (probs[i], -i))
            ranked = [ranked[best]] + [c for i, c in enumerate(ranked) if i != best]
        else:
            head = sorted(range(m), key=lambda i: (-probs[i], i))
            ranked = [ranked[i] for i in head] + ranked[m:]
    return ranked, trace


def flat_rerank(judge: Judge, query: str, cands: Sequence[Node], r: int, m: int = 0, request_size: int | None = None) -> tuple[list[Node], Trace]:
    """比較用: 検索器の候補(木をたどらない)を、同じ判定モデルで 1 段だけ採点する(論文 Tab. 8 の Jev Rerank)。"""
    trace = Trace()
    scores, n_req = judge_pooled(judge, query, cands, request_size)
    trace.add("binary", 1, len(cands), n_req)
    ranked = [cands[i] for i in top_indices(scores, r)]
    mm = min(m, len(ranked))
    if mm >= 2:
        probs = list(judge.choice(query, ranked[:mm]))
        trace.add("choice", 1, mm, 1)
        head = sorted(range(mm), key=lambda i: (-probs[i], i))
        ranked = [ranked[i] for i in head] + ranked[mm:]
    return ranked, trace


# ---------------------------------------------------------------- 指標
def recall_at_k(ranked_ids: Sequence[str], gold: set[str], k: int) -> float:
    return len(gold & set(ranked_ids[:k])) / len(gold) if gold else 0.0


def hit_at_k(ranked_ids: Sequence[str], gold: set[str], k: int = 1) -> float:
    return float(bool(gold & set(ranked_ids[:k])))


def ndcg_at_k(ranked_ids: Sequence[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(ranked_ids[:k]) if x in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------- 遅延と費用の見積もり(式 4)
def levels_needed(n_passages: int, branching: int) -> int:
    """釣り合った木で、n_passages 件の段落に必要な段の数 ⌈log_b N⌉。"""
    if n_passages <= 1:
        return 0
    return math.ceil(math.log(n_passages) / math.log(branching) - 1e-12)


def latency_estimate(n_passages: int, branching: int = 100, t_request: float = 0.2) -> float:
    """T(N) ≈ t_req · ⌈log_b N⌉。段内の判定は並列に出せるので、遅延は段の数に比例する。"""
    return t_request * levels_needed(n_passages, branching)


def max_judgments_per_level(k_kept: int, branching: int) -> int:
    """各段で採点する候補の上限 k · b(段ごとに上位 k 個の節点の子だけを見るため)。"""
    return k_kept * branching
