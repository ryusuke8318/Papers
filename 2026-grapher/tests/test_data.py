import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grapher_mini"))
import data as D  # noqa: E402
import ents as E  # noqa: E402


def test_norm_entity():
    assert E.norm("  The  United States. ") == "united states"
    assert E.norm("Leonardo DiCaprio") == "leonardo dicaprio"


def test_corpus_dedups_and_marks_gold_by_paragraph():
    recs = [
        {"id": "a", "question": "q1", "answer": "x", "paras": [("T", ["s1."]), ("T", ["other."]), ("U", ["u."])], "flags": [True, False, True]},
        {"id": "b", "question": "q2", "answer": "y", "paras": [("U", ["u."]), ("V", ["v."])], "flags": [True, False]},
    ]
    c = D._corpus("musique", recs)
    assert len(c["passages"]) == 4                 # 共有される U は 1 件、同じ title の別段落は別
    assert c["questions"][0]["gold"] == [0, 2]
    assert c["questions"][1]["gold"] == [2]


def test_corpus_drops_question_without_gold():
    c = D._corpus("hotpot", [{"id": "a", "question": "q", "answer": "x", "paras": [("T", ["s."])], "flags": [False]}])
    assert c["questions"] == []
