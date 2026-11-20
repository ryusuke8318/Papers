import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reasonrank_core import (build_prompt, group_advantages, ndcg_at_k, parse_ranking, rbo, recall_at_k, reward,  # noqa: E402
                             sliding_rerank, split_think, window_starts)


def test_parse_exact_and_repair():
    o, ok = parse_ranking("<think>x</think><answer>[3] > [1] > [2]</answer>", 3)
    assert o == [2, 0, 1] and ok
    o, ok = parse_ranking("<answer>[2] > [2] > [9]</answer>", 3)
    assert o == [1, 0, 2] and not ok          # 重複・範囲外は捨て、足りない分は元の順で後ろへ
    o, ok = parse_ranking("no answer", 3)
    assert o == [0, 1, 2] and not ok
    o, ok = parse_ranking("<answer>[1]</answer> <answer>[2] > [1]</answer>", 2)
    assert o == [1, 0] and ok                  # 最後の answer を使う


def test_split_think():
    t, a = split_think("<think> abc </think>\n<answer>[1] > [2]</answer>")
    assert t == "abc" and a == "[1] > [2]"


def test_ndcg_recall():
    rels = {"a": 1.0, "b": 1.0}
    assert ndcg_at_k(["a", "b", "c"], rels) == 1.0
    assert math.isclose(ndcg_at_k(["c", "a", "b"], rels), (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert ndcg_at_k(["c", "d"], rels) == 0.0
    assert recall_at_k(["a", "c", "b"], rels, 2) == 0.5
    assert recall_at_k(["a", "c", "b"], rels, 3) == 1.0


def test_rbo():
    g = list(range(10))
    assert math.isclose(rbo(g, g), 1.0)
    assert rbo(g[::-1], g) < rbo(g[:5][::-1] + g[5:], g) < 1.0
    assert rbo([], g) == 0.0


def test_reward_kinds():
    rels, gold = {0: 1.0, 1: 1.0}, [0, 1, 2, 3]
    assert reward([0, 1, 2, 3], rels, gold, "ndcg") == 1.0
    assert math.isclose(reward([0, 1, 2, 3], rels, gold, "multi"), 1.0 + 0.2 + 0.1)
    assert reward([0, 1, 2, 3], rels, gold, "multi", exact=False) == 0.0
    # NDCG@10 が同じでも、RBO が高いほうが multi の報酬は高い
    a, b = [0, 1, 3, 2], [0, 1, 2, 3]
    assert reward(a, rels, gold, "ndcg") == reward(b, rels, gold, "ndcg")
    assert reward(a, rels, gold, "multi") < reward(b, rels, gold, "multi")


def test_group_advantages():
    a = group_advantages([1.0, 0.0, 0.5, 0.5])
    assert math.isclose(sum(a), 0.0, abs_tol=1e-9) and a[0] > 0 > a[1]
    assert group_advantages([0.3, 0.3, 0.3]) == [0.0, 0.0, 0.0]


def test_window_starts():
    assert window_starts(100, 20, 10) == [80, 70, 60, 50, 40, 30, 20, 10, 0]
    assert window_starts(20, 20, 10) == [0]
    assert window_starts(25, 20, 10) == [5, 0]


def test_sliding_moves_best_to_top():
    # 関連文書 "x" を末尾に置き、各窓で最も良いものを先頭にする rank_fn なら、後ろから前へ運ばれて先頭に着く
    ids = list(range(99)) + ["x"]
    top = lambda win: sorted(range(len(win)), key=lambda i: (win[i] != "x", i))
    assert sliding_rerank(ids, top)[0] == "x"
    # 窓の外へは動かない: 刻み 10・窓 20 で、1 回の窓内の移動は 19 位まで
    ids2 = ["x"] + list(range(99))
    keep = lambda win: list(range(len(win)))
    assert sliding_rerank(ids2, keep) == ids2


def test_prompt_contains_ids():
    p = build_prompt("q?", ["aaa", "bbb"], reasoning=True)
    assert "[1] aaa" in p and "[2] bbb" in p and "<think>" in p
    assert "<think>" not in build_prompt("q?", ["aaa"], reasoning=False)
