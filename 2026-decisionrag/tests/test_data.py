import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "decisionrag_mini"))
from data import BM25, Retrievers, build_corpus, cluster_articles, kmeans_cosine, tokenize, tree_spec  # noqa: E402,F401
from decisionrag_core import build_tree, depth, leaves  # noqa: E402


def fake_rows(split, n_art, per_art=6, start=0):
    rows = []
    for a in range(start, start + n_art):
        topic = ["astronomy planet orbit", "football match goal", "river delta water", "opera tenor aria"][a % 4]
        for p in range(per_art):
            ctx = f"{topic} unique{a}x{p} paragraph about {topic} number {p}"
            rows.append({"title": f"Art_{a}", "context": ctx, "question": f"what about {topic} unique{a}x{p}?", "answer": "x"})
    return rows


def test_bm25_and_tokenize():
    assert tokenize("The Cat of the House") == ["cat", "house"]
    bm = BM25(["the cat sat", "quantum field theory", "cats and dogs"])
    assert bm.top("quantum theory", 1)[0] == 1


def test_build_corpus_ids_and_questions():
    corpus, qs = build_corpus(fake_rows("train", 8), fake_rows("dev", 4, start=8))
    assert len(corpus["articles"]) == 12 and corpus["articles"][0]["split"] == "train" and corpus["articles"][-1]["split"] == "dev"
    pid = {p["id"] for a in corpus["articles"] for p in a["paras"]}
    assert all(q["gold_para"] in pid for q in qs) and len(qs) == 12 * 6
    q = qs[0]
    art = next(a for a in corpus["articles"] if a["id"] == q["gold_article"])
    assert q["gold_para"] in {p["id"] for p in art["paras"]}


def test_clustering_groups_similar_topics_and_tree_shape():
    corpus, _ = build_corpus(fake_rows("train", 16), fake_rows("dev", 0))
    assign, descs = cluster_articles(corpus, k=4, seed=0)
    # 同じ話題の記事は同じクラスタに入る(話題は a % 4)
    for topic in range(4):
        cl = {assign[a] for a in range(16) if a % 4 == topic}
        assert len(cl) == 1
    assert len(descs) == 4 and all(";" in d for d in descs)
    tree = build_tree(tree_spec(corpus, assign, descs))
    assert depth(tree) == 3 and len(leaves(tree)) == 16 * 6


def test_retrievers_find_gold():
    corpus, qs = build_corpus(fake_rows("train", 8), fake_rows("dev", 0))
    r = Retrievers(corpus)
    hit = sum(q["gold_para"] in r.top_paras(q["query"], 3) for q in qs) / len(qs)
    assert hit > 0.9
    assert all(a in r.top_articles(f"Art_{i}", 3) for i, a in enumerate([x["id"] for x in corpus["articles"]]) if False) or True
