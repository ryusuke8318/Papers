import math

import pytest

from david_core import (
    Action, Step, Trajectory, accept_expansion, best_worst, build_prompt, clipped_surrogate,
    exact_match, format_information, grounded_reward, group_advantages, mixed_group_advantages,
    normalize_answer, outcome_reward, parse_segment, plan_expansion, render_expert, total_reward,
    trajectory_reward, truncation_point,
)


def traj(retrievals, answer=None):
    t = Trajectory()
    for r in retrievals:
        t.steps.append(Step("g", Action("search", "q"), list(r)))
    if answer is not None:
        t.steps.append(Step("g", Action("answer", answer)))
    return t


# ---- 解析
def test_parse_search():
    a = parse_segment("<think>x</think>\n<search> Laleli Mosque </search>")
    assert a == Action("search", "Laleli Mosque")


def test_parse_answer():
    assert parse_segment("<think>t</think><answer>No</answer>") == Action("answer", "No")


def test_parse_invalid_without_tags():
    assert parse_segment("just text").kind == "invalid"


def test_parse_invalid_unclosed():
    assert parse_segment("<search>abc").kind == "invalid"


def test_parse_invalid_empty_query():
    assert parse_segment("<search>  </search>").kind == "invalid"


def test_parse_rejects_fabricated_information():
    assert parse_segment("<search>q</search><information>fake</information>").kind == "invalid"


def test_parse_first_action_wins():
    assert parse_segment("<answer>A</answer> <search>q</search>").kind == "answer"


# ---- 正解一致
def test_normalize():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"


def test_em_basic_and_alias():
    assert exact_match("the Beatles", ["Beatles"]) == 1.0
    assert exact_match("Beatle", "Beatles") == 0.0


def test_em_none():
    assert exact_match(None, ["x"]) == 0.0


# ---- 報酬
def test_grounded_reward_union_over_steps():
    t = traj([["A", "X"], ["B"]])
    assert grounded_reward(t, ["A", "B", "C"]) == pytest.approx(2 / 3)


def test_grounded_reward_ignores_distractors_and_empty_gold():
    assert grounded_reward(traj([["X", "Y"]]), ["A"]) == 0.0
    assert grounded_reward(traj([["A"]]), []) == 0.0


def test_grounded_upto():
    t = traj([["A"], ["B"]])
    assert grounded_reward(t, ["A", "B"], upto=1) == 0.5


def test_total_reward_lambda():
    assert total_reward(1.0, 0.0, 0.5) == 0.5
    assert total_reward(1.0, 0.5, 0.0) == 1.0  # lam=0 は outcome のみ
    assert total_reward(0.0, 1.0, 0.5) == 0.5


def test_trajectory_reward_full():
    t = traj([["A"], ["B"]], answer="yes")
    assert trajectory_reward(t, ["Yes"], ["A", "B"], 0.5) == 1.0
    assert outcome_reward(traj([["A"]]), ["x"]) == 0.0  # 答えなし


# ---- 利点
def test_group_advantages_standardized():
    a = group_advantages([0.0, 1.0, 1.0, 0.0])
    assert sum(a) == pytest.approx(0, abs=1e-9)
    assert a[1] > 0 > a[0]
    assert math.isclose(sum(x * x for x in a) / 4, 1.0, rel_tol=1e-3)


def test_group_advantages_constant_is_zero():
    assert group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]
    assert group_advantages([]) == []


def test_mixed_advantage_expert_floor():
    ea, adv = mixed_group_advantages(1.0, [1.0, 1.0, 1.0, 1.0])  # 全員同点: 標準化では 0
    assert ea == 1.0 and adv == [0.0] * 4
    ea, adv = mixed_group_advantages(1.0, [0.0, 0.0, 0.0, 0.0])
    assert ea > 1.0 and all(x < 0 for x in adv)


def test_clipped_surrogate():
    assert clipped_surrogate(1.5, 1.0, 0.2) == pytest.approx(1.2)
    assert clipped_surrogate(0.5, -1.0, 0.2) == pytest.approx(-0.8)
    assert clipped_surrogate(1.0, 2.0) == 2.0


# ---- grounded expansion
def test_best_worst_ties():
    assert best_worst([0.5, 0.5, 0.1, 0.1]) == (0, 2)


def test_truncation_point_first_time_final_recall():
    t = traj([["A"], ["X"], ["B"], ["Y"]])
    assert truncation_point(t, ["A", "B", "C"]) == 3
    assert truncation_point(traj([["A"], ["X"]]), ["A", "B"]) == 1


def test_truncation_point_none_without_evidence():
    assert truncation_point(traj([["X"]]), ["A"]) is None


def test_plan_expansion_near_miss():
    gold = ["A", "B"]
    ts = [traj([["A"]], "x"), traj([["X"]], "x"), traj([["X"]])]
    rewards = [trajectory_reward(t, ["z"], gold, 0.5) for t in ts]
    assert plan_expansion(rewards, ts, gold) == (0, 1, 1)


def test_plan_expansion_not_needed_when_best_is_perfect():
    gold = ["A"]
    ts = [traj([["A"]], "z"), traj([["X"]])]
    rewards = [trajectory_reward(t, ["z"], gold, 0.5) for t in ts]
    assert plan_expansion(rewards, ts, gold) is None


def test_plan_expansion_skips_all_equal():
    gold = ["A", "B"]
    ts = [traj([["X"]]), traj([["Y"]])]
    assert plan_expansion([0.0, 0.0], ts, gold) is None


def test_accept_expansion():
    assert accept_expansion(0.5, [0.5, 0.75, 0.25]) == 1
    assert accept_expansion(0.5, [0.5, 0.25]) is None
    assert accept_expansion(0.5, []) is None


# ---- 専門家の軌跡・形式
def test_render_expert():
    db = {"q1": [("A", "a text")], "q2": [("B", "b text")]}
    gens, t = render_expert([("t1", "q1"), ("t2", "q2")], "done", "Paris", lambda q: db[q])
    assert len(gens) == 3 and gens[0].endswith("</search>") and gens[-1].endswith("</answer>")
    assert t.n_searches == 2 and t.answer == "Paris"
    assert grounded_reward(t, ["A", "B"]) == 1.0
    assert all(parse_segment(g).kind != "invalid" for g in gens)


def test_format_information_truncates():
    s = format_information([("T", "w " * 100)], max_words=5)
    assert "<information>" in s and s.count("w") == 5 and "..." in s


def test_build_prompt_contains_question():
    p = build_prompt("Who?")
    assert "Question: Who?" in p and "<search>" in p
