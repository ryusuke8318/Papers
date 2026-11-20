import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reasonrank_mini"))
from data import BM25, make_lists, tokenize  # noqa: E402


def test_bm25_ranks_matching_doc_first():
    docs = ["the cat sat on the mat", "dogs bark loudly at night", "quantum field theory of electrons", "cats and dogs"]
    bm = BM25(docs)
    assert bm.top("electrons theory", 2)[0] == 2
    assert bm.top("cat mat", 1)[0] == 0
    assert bm.top("zzzz", 2) and len(bm.top("zzzz", 2)) == 2


def test_tokenize_drops_stopwords():
    assert tokenize("The Cat of the House") == ["cat", "house"]


def _rows(n=30):
    rows = []
    for i in range(n):
        paras = {f"T{i}a": f"T{i}a. alpha{i} beta{i} shared words", f"T{i}b": f"T{i}b. gamma{i} delta{i} other text",
                 **{f"N{i}_{j}": f"N{i}_{j}. filler{j} shared words" for j in range(8)}}
        rows.append({"qid": str(i), "question": f"what is alpha{i} beta{i} shared", "paras": paras, "gold": [f"T{i}a", f"T{i}b"]})
    return rows


def test_make_lists_train_contains_both_gold():
    rows = _rows()
    ls = make_lists(rows, rows, "train", n_cand=20, seed=1)
    assert len(ls) == len(rows)
    for l in ls:
        assert len(l["cands"]) == min(20, len(l["cands"])) and len(l["rels"]) == 2
        assert set(l["rels"]) <= {c["id"] for c in l["cands"]}
        assert l["gold_list"][:2] == [x for x in l["gold_list"] if x in l["rels"]][:2]


def test_make_lists_eval_drops_missed_queries():
    rows = _rows()
    ls = make_lists(rows, rows, "eval20", seed=1)
    assert all(len(l["rels"]) == 2 for l in ls)    # 分母は常に正解 2 つ
    # hard=True は、BM25 が正解 2 つを 1・2 位に置いている質問を除く
    easy = make_lists(rows, rows, "eval20", seed=1, hard=False)
    assert len(easy) >= len(ls)
