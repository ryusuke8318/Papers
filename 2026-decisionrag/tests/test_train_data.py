import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "decisionrag_mini"))
from data import Retrievers, build_corpus, cluster_articles  # noqa: E402
from test_data import fake_rows  # noqa: E402
from train_judge import make_examples  # noqa: E402


def setup():
    corpus, qs = build_corpus(fake_rows("train", 24), fake_rows("dev", 0))
    assign, descs = cluster_articles(corpus, k=4, seed=0)
    return corpus, qs, assign, descs, Retrievers(corpus)


def test_example_kinds_and_labels():
    corpus, qs, assign, descs, retr = setup()
    ex = make_examples(corpus, qs, assign, descs, retr, n_questions=10, seed=1)
    assert len(ex) == 40 and [e["header"] for e in ex[:4]] == ["Topics", "Pages", "Paragraphs", "Paragraphs"]
    for e in ex:
        if e["kind"] == "binary":
            assert len(e["texts"]) == len(e["labels"]) and sum(e["labels"]) <= 1
        else:
            assert 0 <= e["label"] < len(e["texts"]) == 10
    # cluster 段は、必ず正解のクラスタが 1 つ入る(yes は 1 件)
    assert all(sum(e["labels"]) == 1 for e in ex[0::4])


def test_gold_present_rate_matches_probabilities():
    corpus, qs, assign, descs, retr = setup()
    ex = make_examples(corpus, qs, assign, descs, retr, n_questions=120, seed=3)
    page_yes = sum(sum(e["labels"]) for e in ex[1::4]) / 120
    para_yes = sum(sum(e["labels"]) for e in ex[2::4]) / 120
    assert 0.7 < page_yes < 0.97 and 0.75 < para_yes < 0.99        # 設定した確率(0.85、0.9)の近く


def test_dev_questions_never_used():
    corpus, qs, assign, descs, retr = setup()
    qs = qs + [{**qs[0], "qid": "dev0", "split": "dev", "query": "DEV-ONLY-QUERY"}]
    ex = make_examples(corpus, qs, assign, descs, retr, n_questions=500, seed=0)
    assert all(e["query"] != "DEV-ONLY-QUERY" for e in ex)
