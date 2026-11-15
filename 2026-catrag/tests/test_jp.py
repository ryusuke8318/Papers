import numpy as np

import catrag_jp as jp


def chunk(i, title, text, score):
    return {"id": i, "title": title, "text": text, "score": score}


TITLES = ["織田信長", "本能寺の変", "明智光秀", "日本", "京都"]


def synthetic():
    return [
        chunk("a", "織田信長", "織田信長は戦国大名で、本能寺の変で没した。", 0.90),
        chunk("d1", "京都", "京都は多くの寺がある都市。", 0.55),
        chunk("d2", "日本", "東アジアの島国。", 0.50),
        chunk("b", "本能寺の変", "1582年、家臣が主君を襲った事件。", 0.20),
        chunk("x", "明智光秀", "戦国時代の武将である。", 0.10),
    ]


def test_find_mentions_longest_first_and_excludes_self():
    m = jp.find_mentions("本能寺の変と本能寺", ["本能寺", "本能寺の変", "変"], min_len=2)
    assert m == {"本能寺の変": 1, "本能寺": 1}
    assert "織田信長" not in jp.find_mentions("織田信長は", ["織田信長"], exclude="織田信長")


def test_linked_bridge_chunk_moves_above_unlinked_distractors():
    out = jp.rerank_with_links(synthetic(), TITLES, gamma=1.0, hub_alpha=0.0)
    ids = [d["id"] for d in out]
    assert ids[0] == "a"
    assert ids.index("b") < ids.index("d1")       # 織田信長が言及する記事 b が、無関係の d1 より上がる
    assert out[ids.index("b")]["activation"] > 0


def test_input_not_mutated_and_gamma_zero_keeps_score_order():
    ch = synthetic()
    before = [dict(c) for c in ch]
    out = jp.rerank_with_links(ch, TITLES, gamma=0.0)
    assert ch == before
    assert [d["id"] for d in out] == ["a", "d1", "d2", "b", "x"]


def test_hub_penalty_is_monotone():
    # 「日本」を 4 つのチャンクが言及し、「本能寺の変」は 1 つだけが言及する
    ch = [
        chunk("a", "織田信長", "織田信長は日本で本能寺の変により没した。", 0.9),
        chunk("h1", "京都", "日本の都市。", 0.4),
        chunk("h2", "大阪", "日本の都市。", 0.3),
        chunk("h3", "東京", "日本の首都。", 0.3),
        chunk("hub", "日本", "島国。", 0.1),
        chunk("b", "本能寺の変", "事件。", 0.1),
    ]
    titles = ["織田信長", "京都", "大阪", "東京", "日本", "本能寺の変"]
    acts = []
    for alpha in (0.0, 0.5, 1.0):
        out = {d["id"]: d for d in jp.rerank_with_links(ch, titles, hub_alpha=alpha, gamma=1.0)}
        acts.append(out["hub"]["activation"])
        assert out["b"]["activation"] >= out["hub"]["activation"] or alpha == 0.0
    assert acts[0] >= acts[1] >= acts[2]
    assert acts[0] > acts[2]


def test_rescale_reduces_activation():
    a = {d["id"]: d["activation"] for d in jp.rerank_with_links(synthetic(), TITLES, c=0.0, gamma=1.0, hub_alpha=0.0)}
    b = {d["id"]: d["activation"] for d in jp.rerank_with_links(synthetic(), TITLES, c=0.5, gamma=1.0, hub_alpha=0.0)}
    assert b["b"] < a["b"]


def test_question_anchor_boosts_mentioned_article():
    ch = [chunk("p", "京都", "京都の説明。", 0.5), chunk("q", "日本", "日本の説明。", 0.5), chunk("r", "織田信長", "武将。", 0.9)]
    out = jp.rerank_with_links(ch, TITLES, question="京都について", eps=0.2, gamma=1.0)
    # 質問に記事名が出ている「京都」が、同点の「日本」より上
    ids = [d["id"] for d in out]
    assert ids.index("p") < ids.index("q")


def test_empty_and_single_chunk():
    assert jp.rerank_with_links([], TITLES) == []
    out = jp.rerank_with_links([chunk("a", "日本", "x", 0.3)], TITLES)
    assert len(out) == 1 and out[0]["final"] >= 0
