import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from raser_pipeline import Pipeline, clean_answer, merge_ids  # noqa: E402

TEXTS = {0: "Young Man Luther: a book by Erik Erikson.", 1: "Luther family.", 2: "Erik Erikson: his wife was Joan Erikson.", 3: "Erik Erikson was born in 1902.",
         4: "Unrelated A.", 5: "Unrelated B."}


def fake_retrieve(queries, k):
    ids, sc = [], []
    for q in queries:
        order = [2, 3, 0, 1, 4, 5] if "Erik Erikson" in q else [0, 1, 4, 5, 2, 3]
        ids.append(order[:k])
        sc.append([1.0 - 0.1 * j for j in range(k)])
    return ids, sc


def fake_gen(prompts, max_tokens):
    out = []
    for p in prompts:
        if "Bridge proposals" in p:
            t = '{"bridge_entity": "Erik Erikson", "confidence": 0.9}\n{"bridge_entity": "Nobody Here"}'
        elif "Fact:" in p and "extract ONE" in p:
            t = "DONE" if "Erik Erikson" in p.split("Facts already found:")[1] else "Erik Erikson"
        elif "Joan Erikson" in p and "Erik Erikson: his wife" in p:
            t = "Joan Erikson"
        else:
            t = "I don't know"
        out.append((t, len(p.split()), len(t.split())))
    return out


def test_all_routes():
    pipe = Pipeline(fake_gen, fake_retrieve, TEXTS, k=2, branch_k=2)
    r = pipe.run(["Who is the spouse of the author of Young Man Luther?"])[0]
    assert r["one_shot"]["answer"] == "I don't know"
    assert r["one_shot"]["features"]["confidence"] == 0.0
    assert r["prune"]["answer"] == "Joan Erikson"
    assert r["prune"]["kept"] == ["Erik Erikson"]                       # 空の枝(Nobody Here の検索結果は新規だが実体を含まない)は捨てる
    assert r["ircot"]["answer"] == "Joan Erikson" and r["ircot"]["facts"] == ["Erik Erikson"]
    # 費用は、PRUNE と IRCOT* に 1 回目の分を含む
    assert r["prune"]["tokens"] > r["one_shot"]["tokens"] and r["ircot"]["tokens"] > r["one_shot"]["tokens"]


def test_helpers():
    assert clean_answer("Answer: Paris.\nmore") == "Paris"
    assert merge_ids([1, 2], [2, 3], cap=3) == [1, 2, 3] and merge_ids([1, 2], [3, 4], cap=3) == [1, 2, 3]
