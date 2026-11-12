"""rankrag_core の動作確認(numpy と scipy のみ。GPU 不要)。実行: python tests/test_core.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rankrag_core import (BM25, answer_in_text, contains_answer, exact_match, f1_score, mrr, normalize_answer,
                          parse_indexes, qa_prompt, rank_prompt, rar_prompt, recall_at_k, truncate_words)


def test_answer_metrics():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert exact_match("the Eiffel tower", ["Eiffel Tower"]) == 1.0
    assert exact_match("Eiffel", ["Eiffel Tower"]) == 0.0
    assert abs(f1_score("Eiffel", ["Eiffel Tower"]) - 2 / 3) < 1e-9
    assert contains_answer("It is the Eiffel Tower in Paris.", ["Eiffel Tower"]) == 1.0
    assert contains_answer("", ["x"]) == 0.0
    assert answer_in_text(["Paris"], "He lives in paris.")


def test_ranking_metrics():
    assert recall_at_k([3, 1, 2], {1, 2}, 1) == 0.0
    assert recall_at_k([3, 1, 2], {1, 2}, 3) == 1.0
    assert recall_at_k([1, 3, 2], {1, 2}, 2) == 0.5
    assert mrr([3, 1, 2], {2}) == 1 / 3
    assert mrr([3], {9}) == 0.0


def test_prompts():
    assert qa_prompt("q?", ["a"]).endswith("Passage: a")
    assert "Passage 2: b" in qa_prompt("q?", ["a", "b"])
    assert rank_prompt("q?", "a").startswith("For the question q?, access whether")
    assert "find all passages" in rar_prompt("q?", ["a", "b"])
    assert parse_indexes("2, 4", 5) == [1, 3]
    assert parse_indexes("Passages 2 and 9, 2", 5) == [1]  # 範囲外と重複は捨てる
    assert parse_indexes("none", 5) == []
    assert len(truncate_words("a b c d", 2).split()) == 2


def test_bm25():
    docs = ["the cat sat on the mat", "dogs bark at night", "quantum computing uses qubits", "a cat and a dog"]
    bm = BM25(docs)
    assert bm.topk("qubits computing", 1) == [2]
    assert bm.topk("cat", 2)[0] in (0, 3)
    assert set(bm.topk("cat", 4)) == {0, 1, 2, 3}
    assert bm.scores("zzzunknown").sum() == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("OK")
