import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import raser_core as rc  # noqa: E402


def test_f1_basic():
    assert rc.f1_score("The Beatles", "beatles") == 1.0
    assert rc.f1_score("Cabo Delgado Province", "Cabo Delgado") == pytest.approx(0.8)
    assert rc.f1_score("I don't know", "Joan Erikson") == 0.0
    assert rc.f1_max("Obama", ["Barack Obama", "Obama"]) == 1.0


def test_chunking_limit_and_overlap():
    count = lambda s: len(s.split())
    text = " ".join(f"Sentence {i} has exactly six words." for i in range(30))  # 1 文 6 語
    ch = rc.chunk_passage(text, count, max_tokens=20, overlap=6)
    assert all(count(c) <= 20 for c in ch) and len(ch) > 1
    last_of_first = rc.split_sentences(ch[0])[-1]
    assert rc.split_sentences(ch[1])[0] == last_of_first         # 重なり
    assert rc.chunk_passage("Short one.", count) == ["Short one."]


def test_question_type_rules():
    assert rc.question_type("Were Scott Derrickson and Ed Wood of the same nationality?") == "yes_no"
    assert rc.question_type("How many members are in the band?") == "count"
    assert rc.question_type("When did the person who ended the Achaemenid Empire die?") == "date"
    assert rc.question_type("Who is the spouse of the Green performer?") == "entity"
    assert rc.question_type("Why is the sky blue?") == "other"
    assert rc.QTYPES == {"entity": 0, "date": 1, "yes_no": 2, "count": 3, "other": 4}


def test_bridge_cues():
    assert rc.bridge_cues("Who is the spouse of the author of Young Man Luther?") == 1
    assert rc.bridge_cues("What is the country where the Eiffel Tower is located in?") == 1
    assert rc.bridge_cues("Is Paris big?") == 0


def test_confidence_rules():
    assert rc.confidence("I don't know", "date", ["x"]) == 0.0
    assert rc.confidence("Paris", "entity", ["Paris is a city"]) == pytest.approx(0.8)
    assert rc.confidence("Paris", "entity", ["London"]) == pytest.approx(0.6)
    assert rc.confidence("yes", "yes_no", ["yes yes"]) == pytest.approx(0.9)
    assert rc.confidence("1990", "date", ["in 1990"]) == pytest.approx(1.0)


def test_features_definition():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    f = rc.extract_features("Who is the spouse of the author of X?", "Joan Erikson", scores, ["Joan Erikson was"])
    assert set(f) == set(rc.FEATURES)
    assert f["ans_len"] == 2 and f["score_top1"] == 0.9
    assert f["score_gap"] == pytest.approx(0.4)        # top-1 と top-5
    assert f["bridge_cues"] == 1 and f["qtype"] == 0
    assert f["confidence"] == pytest.approx(0.8)
    # スコアが昇順で渡されても、降順に直してから計算する
    f2 = rc.extract_features("q?", "a", scores[::-1], [])
    assert f2["score_top1"] == 0.9


def test_parse_bridges():
    t = '{"bridge_entity": "Erik Erikson", "confidence": 0.9}\nnoise\n{"bridge_entity": "Martin Luther"}\n{"bridge_entity": "Third"}'
    assert rc.parse_bridges(t) == ["Erik Erikson", "Martin Luther"]            # 最大 2 つ
    assert rc.parse_bridges('{"bridge_entity": "A"}\n{"bridge_entity": "a"}') == ["A"]   # 重複
    assert rc.parse_bridges("{broken json") == []
    multi = '```json\n{\n  "bridge_entity": "Erik Erikson",\n  "confidence": 0.9\n}\n```'
    assert rc.parse_bridges(multi) == ["Erik Erikson"]                           # 複数行・コードフェンス


def test_verifier_rules():
    texts = {0: "x", 1: "Erik Erikson wrote", 2: "other", 3: "Erik Erikson born", 4: "y"}
    # 新規 2 件(1, 3)で、どちらも実体を含む -> 残す
    v = rc.verify_branch("Erik Erikson", [0, 1, 3], [0, 2, 4], texts)
    assert v["novelty"] == pytest.approx(2 / 3) and v["support"] == 1.0 and v["keep"]
    assert v["info_gain"] == pytest.approx(1 - 1 / 5)
    # すべて 1 回目と同じ塊 -> novelty 0 で捨てる
    assert not rc.verify_branch("Erik", [0, 2], [0, 2, 4], texts)["keep"]
    # 新しい塊が実体を含まない -> support 0 で捨てる
    assert not rc.verify_branch("Zed", [1, 3], [0], texts)["keep"]
    assert not rc.verify_branch("Zed", [], [0], texts)["keep"]


def test_label_tau():
    y = rc.bridgeable_label([0.5, 0.2, 0.0, 1.0], [0.4, 0.1, 0.0, 0.5], tau=0.1)
    assert list(y) == [0, 0, 0, 1]       # 差がちょうど tau のときは 0(厳密な不等号)


def test_argmax_lambda():
    pred = np.array([[0.3, 0.5, 0.6], [0.5, 0.4, 0.3]])
    cost = np.array([1000.0, 3000.0, 4000.0])
    assert list(rc.cost_aware_argmax(pred, cost, 0.0)) == [2, 0]          # lam=0: 予測 F1 が最大
    assert list(rc.cost_aware_argmax(pred, cost, 1.0)) == [0, 0]          # lam が大きいと one-shot
    assert list(rc.cost_aware_argmax(pred, cost, 1e-4)) == [1, 0]         # 0.3-0.1, 0.5-0.3, 0.6-0.4 -> prune が最大
    assert list(rc.cost_aware_argmax(pred, cost, np.array([0.0, 1.0]))) == [2, 0]   # 行ごとの lam


def test_choose_lambda_respects_budget():
    rng = np.random.RandomState(0)
    n = 200
    pred = rng.rand(n, 3)
    T = np.tile([1000.0, 3000.0, 4000.0], (n, 1))
    grid = rc.default_lam_grid()
    for b in (0.3, 0.6, 1.0):
        lam = rc.choose_lambda(pred, T, T.mean(0), b, grid)
        r = rc.cost_aware_argmax(pred, T.mean(0), lam)
        assert T[np.arange(n), r].mean() <= b * 4000 + 1e-9
    # 予算が大きいほど lam は小さい
    assert rc.choose_lambda(pred, T, T.mean(0), 1.0, grid) <= rc.choose_lambda(pred, T, T.mean(0), 0.3, grid)


def test_group_folds_by_question():
    groups = np.array([f"q{i // 3}" for i in range(60)])         # 1 質問が 3 行
    fold = rc.group_folds(groups, 5, seed=1)
    for g in np.unique(groups):
        assert len(set(fold[groups == g])) == 1
    assert set(fold) == set(range(5))
    assert (fold == rc.group_folds(groups, 5, seed=1)).all()


def test_cv_router_never_trains_on_eval_rows():
    rng = np.random.RandomState(0)
    n = 150
    X = rng.rand(n, 6)
    F1 = rng.rand(n, 3)
    F1[:, 1] = F1[:, 0] + (X[:, 3] > 0.5) * 0.5               # score_gap が大きいと PRUNE が効く
    T = np.tile([1000.0, 3000.0, 4000.0], (n, 1))
    out = rc.cv_router(X, F1, T, np.arange(n), budgets=(0.6,))
    assert out["p2"].shape == (n,) and out["pred3"].shape == (n, 3) and out["lam"][0.6].shape == (n,)
    assert np.corrcoef(out["p2"], out["y"])[0, 1] > 0.5         # 信号を学習している
    assert np.allclose(out["cbar"], T.mean(0))


def test_cv_router_single_class_fold():
    X = np.random.rand(40, 6)
    F1 = np.zeros((40, 3))
    out = rc.cv_router(X, F1, np.ones((40, 3)), np.arange(40))
    assert (out["p2"] == 0).all()


def test_route_stats_and_oracle():
    F1 = np.array([[0.0, 1.0, 0.5], [1.0, 1.0, 1.0]])
    T = np.array([[1.0, 3.0, 4.0]] * 2)
    s = rc.route_stats(np.array([1, 0]), F1, T)
    assert s["f1"] == 1.0 and s["tokens"] == 2.0 and s["mix"] == [0.5, 0.5, 0.0]
    assert list(rc.oracle_route(F1, T)) == [1, 0]               # 同点は先の経路


def test_recoverability_partition():
    F1 = np.array([[0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.2, 0.25, 0.2], [0.8, 1.0, 0.8]])
    r = rc.recoverability(F1)
    assert sum(r.values()) == pytest.approx(1.0)
    assert r["no_method_helps"] == pytest.approx(0.2) and r["bridge_helps"] == pytest.approx(0.4)
    assert r["small_gain"] == pytest.approx(0.2) and r["one_shot_correct"] == pytest.approx(0.2)


def test_bootstrap_contains_point():
    x = np.random.RandomState(0).rand(200)
    p, lo, hi = rc.bootstrap(lambda i: x[i].mean(), 200, B=300)
    assert lo < p < hi and hi - lo < 0.2


def test_escalate_threshold():
    assert list(rc.escalate([0.19, 0.2, 0.9], 0.2)) == [0, 1, 1]       # 閾値ちょうどは上げる(論文: 確率が θ 以上)


def test_match_tokens():
    rng = np.random.RandomState(0)
    n = 400
    T = np.tile([1000.0, 3000.0, 4000.0], (n, 1))
    p2 = rng.rand(n)
    r = rc.match_tokens_r2(p2, T, 1500.0)
    assert r.sum() == 100 and set(np.where(r == 1)[0]) == set(np.argsort(-p2)[:100])   # p2 が高い 100 件だけ上げる
    assert rc.match_tokens_r2(p2, T, 99999.0).sum() == n and rc.match_tokens_r2(p2, T, 0.0).sum() == 0
    pred3 = rng.rand(n, 3)
    r3 = rc.match_tokens_r3(pred3, T[0], T, 2000.0)
    assert abs(T[np.arange(n), r3].mean() - 2000.0) < 250
