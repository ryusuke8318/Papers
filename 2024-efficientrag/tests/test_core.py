import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest

from efficientrag_core import (align_labels, align_two, assemble_query, bootstrap_ci, build_examples, exact_match, extract_json_value, f1_score,
                               paired_bootstrap, parse_selfask, pick_negatives, pool_size, recall_at_pool, run_loop, select_words,
                               split_words, valid_decomposition)


# ---------------------------------------------------------------- ラベルの変換
def test_align_labels_keeps_order_and_ignores_punctuation_and_case():
    words = split_words("KGOT (101.3 FM) is a station, with studios in the Dimond Center.")
    labels, rate = align_labels(words, "KGOT studios in Dimond Center".split())
    assert select_words(words, labels) == ["KGOT", "studios", "in", "Dimond", "Center."]
    assert rate == 1.0


def test_align_labels_does_not_match_backwards():
    words = ["a", "b", "c", "b"]
    labels, rate = align_labels(words, ["c", "b"])
    assert labels == [0, 0, 1, 1] and rate == 1.0       # c の後ろの b に当たる
    labels, rate = align_labels(words, ["c", "a"])      # a は c より前にしかないので照合できない
    assert labels == [0, 0, 1, 0] and rate == 0.5


def test_align_labels_ignores_added_words_in_rate():
    labels, rate = align_labels(["x", "y"], ["x", "zzz", "y"])
    assert labels == [1, 1] and rate == pytest.approx(2 / 3)


def test_align_two_question_part_first_then_info():
    q = ["Where", "was", "the", "director", "born", "?"]
    info = ["Alice", "directed", "Film", "X"]
    ql, il, rate = align_two(q, info, ["Where", "was", "born", "?", "Alice", "Film"])
    assert ql == [1, 1, 0, 0, 1, 0] and il == [1, 0, 1, 0] and rate == 1.0
    ql, il, rate = align_two(q, info, ["Where", "was", "Alice", "born"])   # 順序が逆でも、質問側を先に照合する
    assert ql == [1, 1, 0, 0, 1, 0] and il == [1, 0, 0, 0]


def test_align_labels_empty():
    assert align_labels(["x"], [])[1] == 0.0


# ---------------------------------------------------------------- 次の問い
def test_assemble_query_order_and_spacing():
    q = split_words("How large is the shopping mall where KGOT radio station has its studios ?")
    ql = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    info = split_words("KGOT , in the  Dimond   Center")
    il = [0, 0, 0, 0, 1, 1]
    assert assemble_query(q, ql, info, il) == "How large Dimond Center"


def test_assemble_query_dedupes_words_already_in_question_part():
    q = ["Who", "founded", "Dimond", "Center"]
    out = assemble_query(q, [1, 0, 1, 0], ["Dimond", "Center", "Mall"], [1, 1, 1])
    assert out == "Who Dimond Center Mall"     # Dimond は重複するので Info 側から落ちる
    assert assemble_query(q, [1, 0, 1, 0], ["Dimond"], [1], dedupe=False) == "Who Dimond Dimond"


def test_assemble_query_empty_when_nothing_selected():
    assert assemble_query(["a"], [0], ["b"], [0]) == ""


# ---------------------------------------------------------------- 反復検索のループ
def _env(index, tag, nxt):
    """index: 問い -> チャンク ID の列。tag: チャンク ID -> P(Continue)。nxt: Info の語 -> 次の問い。"""
    calls = {"retrieve": 0, "label": 0}

    def retrieve(q, k):
        calls["retrieve"] += 1
        return index.get(q, [])[:k]

    def label(q, ids):
        calls["label"] += 1
        return [(tag[c], [c]) for c in ids]

    def filt(question, info):
        return nxt.get(tuple(info), "")
    return retrieve, label, filt, calls


def test_loop_two_hops_then_terminate():
    index = {"Q": ["A", "X"], "Q2": ["B", "Y"], "Q3": ["Z"]}
    tag = {"A": 0.9, "X": 0.1, "B": 0.8, "Y": 0.2, "Z": 0.0}
    r, l, f, calls = _env(index, tag, {("A",): "Q2", ("A", "B"): "Q3"})
    res = run_loop("Q", r, l, f, k=5, max_iter=5)
    assert res.pool == ["A", "B"]
    assert res.iterations == 3          # Q, Q2, Q3(全て Terminate で止まる)
    assert res.queries == ["Q", "Q2", "Q3"]


def test_loop_stops_when_all_terminate_in_first_round():
    r, l, f, _ = _env({"Q": ["X", "Y"]}, {"X": 0.2, "Y": 0.4}, {})
    res = run_loop("Q", r, l, f)
    assert res.pool == [] and res.iterations == 1


def test_loop_respects_max_iter():
    index = {"Q": ["A"], "Q2": ["B"], "Q3": ["C"], "Q4": ["D"]}
    tag = dict.fromkeys("ABCD", 1.0)
    nxt = {("A",): "Q2", ("A", "B"): "Q3", ("A", "B", "C"): "Q4"}
    r, l, f, _ = _env(index, tag, nxt)
    res = run_loop("Q", r, l, f, max_iter=2)
    assert res.pool == ["A", "B"] and res.iterations == 2


def test_loop_threshold_and_no_duplicate_chunks_or_queries():
    index = {"Q": ["A", "B"], "Q2": ["A", "B"]}     # 2 つ目の問いは同じチャンクを返す
    tag = {"A": 0.5, "B": 0.49}
    r, l, f, _ = _env(index, tag, {("A",): "Q2"})
    res = run_loop("Q", r, l, f, threshold=0.5)
    assert res.pool == ["A"]                         # 0.5 以上だけ(B は <Terminate>)
    assert res.iterations == 2 and res.queries == ["Q", "Q2"]


def test_loop_empty_filter_output_ends_branch():
    r, l, f, _ = _env({"Q": ["A"]}, {"A": 1.0}, {})
    res = run_loop("Q", r, l, f)
    assert res.pool == ["A"] and res.iterations == 1


def test_loop_branches_per_continue_chunk():
    index = {"Q": ["A", "B"], "QA": ["C"], "QB": ["D"]}
    tag = dict.fromkeys("ABCD", 1.0)
    r, l, f, _ = _env(index, tag, {("A",): "QA", ("B",): "QB"})
    res = run_loop("Q", r, l, f, max_iter=2)
    assert res.pool == ["A", "B", "C", "D"] and res.n_filter_calls == 4


# ---------------------------------------------------------------- 負例
def test_pick_negatives_skips_gold_and_keeps_rank_order():
    assert pick_negatives(["g1", "n1", "g2", "n2", "n3"], {"g1", "g2"}, n=2) == ["n1", "n2"]
    assert pick_negatives(["g1"], {"g1"}) == []


def test_build_examples_negatives_are_not_gold_and_tagged_terminate():
    words = {"d1": split_words("Alice directed Film X ."), "d2": split_words("Alice was born in Paris ."), "n": split_words("Unrelated text here")}
    rec = {"question": "Where was the director of Film X born ?",
           "subs": {"1": {"doc": "d1", "deps": [], "labeled": ["Alice", "directed", "Film", "X"], "filtered": None},
                    "2": {"doc": "d2", "deps": ["1"], "labeled": ["Alice", "born", "Paris"], "filtered": ["Where", "was", "Alice", "born", "?"]}}}
    lab, fil, st = build_examples(rec, words, {"1": ["n"], "2": ["n"]})
    assert st["labeler_pos"] == 2 and st["labeler_neg"] == 2 and st["filter"] == 1
    pos = [e for e in lab if e["tag"] == 1]
    neg = [e for e in lab if e["tag"] == 0]
    assert all(sum(e["lb"]) == 0 and e["b"] == words["n"] for e in neg)
    assert pos[0]["a"] == split_words(rec["question"])
    assert pos[1]["a"] == ["Where", "was", "born", "Alice"]      # 依存のある副問いの入力は、次の問い(記号だけの語は選ばれない)
    f = fil[0]
    assert f["b"] == ["Alice", "directed", "Film", "X"] and f["la"][0] == 1


def test_build_examples_drops_unmatched_llm_output():
    words = {"d1": split_words("alpha beta gamma delta")}
    rec = {"question": "q ?", "subs": {"1": {"doc": "d1", "deps": [], "labeled": ["foo", "bar", "baz"], "filtered": None}}}
    lab, fil, st = build_examples(rec, words, {})
    assert lab == [] and st["labeler_dropped"] == 1


def test_valid_decomposition():
    ok = {"1": {"doc": "a", "deps": []}, "2": {"doc": "b", "deps": ["1"]}}
    assert valid_decomposition(ok, ["a", "b"])
    assert not valid_decomposition(ok, ["a", "c"])
    assert not valid_decomposition({"1": {"doc": "a", "deps": ["2"]}, "2": {"doc": "b", "deps": ["1"]}}, ["a", "b"])
    assert not valid_decomposition({"1": {"doc": "a", "deps": ["9"]}, "2": {"doc": "b", "deps": []}}, ["a", "b"])


# ---------------------------------------------------------------- 指標
def test_recall_and_k():
    pool = ["a", "b", "a", "c"]
    assert pool_size(pool) == 3
    assert recall_at_pool(pool, ["a", "z"]) == 0.5
    assert recall_at_pool([], ["a"]) == 0.0


def test_em_f1():
    assert exact_match("The Little Richard.", "little richard") == 1.0
    assert f1_score("Little Richard Penniman", "Little Richard") == pytest.approx(0.8)
    assert f1_score("x", "y") == 0.0
    assert f1_score("yes", "yes") == 1.0


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    vals = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    m, lo, hi = bootstrap_ci(vals, n_boot=2000, seed=1)
    assert lo <= m <= hi and (m, lo, hi) == bootstrap_ci(vals, n_boot=2000, seed=1)
    d, dlo, dhi = paired_bootstrap([1, 1, 1, 1], [0, 0, 0, 0], n_boot=500)
    assert d == dlo == dhi == 1.0


# ---------------------------------------------------------------- 解析
def test_parse_selfask():
    assert parse_selfask("Are follow up questions needed here: Yes.\nFollow up: Who directed X?") == ("follow_up", "Who directed X?")
    assert parse_selfask("So the final answer is: Paris.") == ("final", "Paris")
    assert parse_selfask("Follow up: A?\nIntermediate answer: B\nSo the final answer is: C")[0] == "follow_up"
    assert parse_selfask("hmm") == ("none", "")


def test_extract_json_value_tolerates_broken_json():
    assert extract_json_value('```json\n{"answer": "Paris"}\n```', "answer") == "Paris"
    assert extract_json_value('{"answer": "Paris", oops', "answer") == "Paris"
    assert extract_json_value("nothing", "answer") is None


def test_build_examples_negatives_callback_receives_query():
    words = {"d1": split_words("Alice directed Film X ."), "n": split_words("Unrelated text")}
    rec = {"question": "Who directed Film X ?", "subs": {"1": {"doc": "d1", "deps": [], "labeled": ["Alice", "directed"], "filtered": None}}}
    seen = []

    def neg(sid, query):
        seen.append((sid, query))
        return pick_negatives(["d1", "n"], {"d1"}, 1)
    lab, _, st = build_examples(rec, words, neg)
    assert seen == [("1", "Who directed Film X ?")] and st["labeler_neg"] == 1 and lab[-1]["tag"] == 0
