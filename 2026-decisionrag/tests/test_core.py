import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decisionrag_core import (NavConfig, build_tree, depth, flat_rerank, hit_at_k, latency_estimate, leaves, levels_needed,  # noqa: E402
                              navigate, ndcg_at_k, recall_at_k, split_requests, top_indices)


def make_tree(n_cl=4, n_pg=5, n_pa=6):
    return build_tree({"id": "root", "text": "", "children": [
        {"id": f"c{c}", "text": f"cluster {c}", "children": [
            {"id": f"c{c}p{p}", "text": f"page {c}.{p}", "children": [
                {"id": f"c{c}p{p}a{a}", "text": f"para {c}.{p}.{a}"} for a in range(n_pa)]}
            for p in range(n_pg)]} for c in range(n_cl)]})


def subtree_ids(node):
    return {x.id for x in leaves(node)}


class OracleJudge:
    """正解の段落を含む枝には高い確率、それ以外には低い確率を付ける。呼ばれた候補をすべて記録する。"""

    def __init__(self, gold: set[str], blind: set[str] = frozenset()):
        self.gold, self.blind, self.calls, self.seen = gold, set(blind), [], []

    def _p(self, n):
        ids = subtree_ids(n)
        if n.id in self.blind:
            return 0.05                       # 判定が外れる枝(正解を含むのに低く見積もる)
        return 0.9 if ids & self.gold else 0.05

    def binary(self, query, candidates):
        self.calls.append(("binary", [c.id for c in candidates]))
        self.seen += [c.id for c in candidates]
        # 同点を避けるため、元の順で微小な差を付ける
        return [self._p(c) - 1e-4 * i for i, c in enumerate(candidates)]

    def choice(self, query, candidates):
        self.calls.append(("choice", [c.id for c in candidates]))
        w = [10.0 if subtree_ids(c) & self.gold else 1.0 for c in candidates]
        s = sum(w)
        return [x / s for x in w]


def test_tree_helpers():
    t = make_tree()
    assert depth(t) == 3 and len(leaves(t)) == 4 * 5 * 6


def test_navigate_finds_gold_first():
    t, gold = make_tree(), {"c2p3a4"}
    j = OracleJudge(gold)
    ranked, tr = navigate(t, "q", j, NavConfig(budgets=[1, 1], r=10))
    assert ranked[0].id == "c2p3a4"
    # 刈り込みで、他のクラスタの下は 1 度も採点されない(クラスタ 4 つ、ページ 5 つ、最後の段落 6 つ)
    assert all(not i.startswith(("c0", "c1", "c3")) or i in {"c0", "c1", "c2", "c3"} for i in j.seen)


def test_budget_too_small_prunes_irreversibly():
    t, gold = make_tree(), {"c1p0a0"}
    # クラスタ c1 が(判定の誤りで)低く見積もられると、予算 1 では落ち、落ちた枝の下は二度と見られない
    j = OracleJudge(gold, blind={"c1"})
    ranked, _ = navigate(t, "q", j, NavConfig(budgets=[1, 1], r=10))
    assert "c1p0a0" not in {x.id for x in ranked}
    assert not any(i.startswith("c1p") for i in j.seen)
    # 予算を増やせば、残る(c1 が 2 位以内に入る)
    j2 = OracleJudge(gold, blind={"c1"})
    ranked2, _ = navigate(t, "q", j2, NavConfig(budgets=[4, 1], r=10))
    assert "c1p0a0" in {x.id for x in ranked2}


def test_trace_counts_sequential_stages():
    t = make_tree()
    _, tr = navigate(t, "q", OracleJudge({"c0p0a0"}), NavConfig(budgets=[2, 2], r=10))
    assert tr.sequential_stages == depth(t)                      # 段ごとに 1 回(複数選択なし)
    _, tr2 = navigate(t, "q", OracleJudge({"c0p0a0"}), NavConfig(budgets=[2, 2], r=10, m=5))
    assert tr2.sequential_stages == depth(t) + 1                 # 最後に複数選択の段が 1 つ増える
    assert tr2.stages[-1]["kind"] == "choice" and tr2.stages[-1]["candidates"] == 5


def test_request_split_preserves_order_and_counts():
    assert [len(c) for c in split_requests(list(range(25)), 10)] == [10, 10, 5]
    assert split_requests(list(range(5)), None) == [list(range(5))]
    t = make_tree(n_cl=1, n_pg=1, n_pa=25)
    j = OracleJudge({"c0p0a7"})
    ranked, tr = navigate(t, "q", j, NavConfig(budgets=[1, 1], r=5, request_size=10))
    assert tr.stages[-1]["requests"] == 3                        # 段落 25 件は 3 請求に分かれる
    assert max(len(ids) for kind, ids in j.calls if kind == "binary") <= 10
    assert ranked[0].id == "c0p0a7"


def test_choice_reorders_only_top_m():
    t = make_tree(n_cl=1, n_pg=2, n_pa=6)
    j = OracleJudge({"c0p1a5"})                                  # 二値判定は粗いが、複数選択で 1 位に押し上げる
    ranked, _ = navigate(t, "q", j, NavConfig(budgets=[1, 2], r=12, m=6))
    assert ranked[0].id == "c0p1a5"
    base, _ = navigate(t, "q", OracleJudge({"c0p1a5"}), NavConfig(budgets=[1, 2], r=12, m=0))
    assert [x.id for x in ranked[6:]] == [x.id for x in base[6:]]  # 6 位以降は二値判定の順のまま


def test_promote_only_moves_one_item():
    t = make_tree(n_cl=1, n_pg=2, n_pa=6)
    base, _ = navigate(t, "q", OracleJudge({"c0p1a5"}), NavConfig(budgets=[1, 2], r=12))
    pro, _ = navigate(t, "q", OracleJudge({"c0p1a5"}), NavConfig(budgets=[1, 2], r=12, m=6, promote_only=True))
    assert pro[0].id == "c0p1a5"
    assert [x.id for x in pro[1:]] == [x.id for x in base if x.id != "c0p1a5"]


def test_extra_candidates_added_once():
    t = make_tree()
    extra = [leaves(t)[100], leaves(t)[0]]                       # 100 番目は刈り込まれた枝の下、0 番目は最初から残る
    gold = leaves(t)[100].id                                     # c3p1a4。クラスタ c3 を判定が外して、刈り込みで落ちる
    j = OracleJudge({gold}, blind={"c3"})
    ranked, tr = navigate(t, "q", j, NavConfig(budgets=[1, 1], r=10), extra_candidates=extra)
    ids = [x.id for x in ranked]
    assert ids[0] == gold and len(ids) == len(set(ids))          # 刈り込みで落ちた正解を、追加の候補が救う
    assert tr.stages[-1]["candidates"] == 6 + 1                  # 残った文書の段落 6 件 + 追加のうち新しい 1 件(c0p0a0 は重複)


def test_weight_by_parent_changes_order():
    class J:
        def binary(self, q, cands):
            # 文書の段(子が段落): 文書 B の方が高い。段落の段: 段落 A も B も同じ
            return [0.9 if c.id == "pB" else 0.5 if c.id == "pA" else 0.8 for c in cands]

        def choice(self, q, cands):
            return [1 / len(cands)] * len(cands)
    t = build_tree({"id": "r", "children": [
        {"id": "pA", "children": [{"id": "a1"}]}, {"id": "pB", "children": [{"id": "b1"}]}]})
    plain, _ = navigate(t, "q", J(), NavConfig(budgets=[2], r=2))
    weighted, _ = navigate(t, "q", J(), NavConfig(budgets=[2], r=2, weight_by_parent=True))
    assert [x.id for x in weighted] == ["b1", "a1"]
    assert {x.id for x in plain} == {"a1", "b1"}


def test_flat_rerank_and_metrics():
    t = make_tree(n_cl=1, n_pg=1, n_pa=10)
    cands = leaves(t)
    j = OracleJudge({"c0p0a3"})
    ranked, tr = flat_rerank(j, "q", cands, r=5, m=3)
    assert ranked[0].id == "c0p0a3" and tr.sequential_stages == 2
    ids = [x.id for x in ranked]
    assert hit_at_k(ids, {"c0p0a3"}) == 1.0 and recall_at_k(ids, {"c0p0a3", "zzz"}, 5) == 0.5
    assert ndcg_at_k(["x", "g"], {"g"}, 10) == 1 / math.log2(3)
    assert ndcg_at_k(["g"], {"g"}) == 1.0 and ndcg_at_k([], {"g"}) == 0.0


def test_top_indices_stable():
    assert top_indices([0.5, 0.9, 0.9, 0.1], 3) == [1, 2, 0]


def test_levels_and_latency_scale_with_depth_not_size():
    assert levels_needed(1, 100) == 0 and levels_needed(100, 100) == 1 and levels_needed(101, 100) == 2
    assert levels_needed(10 ** 9, 100) == 5                      # 論文の例: 10^9 段落でも 5 段
    assert math.isclose(latency_estimate(10 ** 9, 100, 0.2), 1.0)
    # 100 倍ごとに 1 段(0.2 秒)ずつ増える
    lat = [latency_estimate(100 ** k, 100) for k in range(1, 6)]
    assert all(math.isclose(b - a, 0.2) for a, b in zip(lat, lat[1:]))
