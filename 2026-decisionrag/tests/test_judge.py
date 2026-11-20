import sys
import zlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "decisionrag_mini"))
from judge import DecisionJudge, DecisionModel, build_binary_layout, build_choice_layout  # noqa: E402
from decisionrag_core import Node  # noqa: E402


class FakeTok:
    """単語を決定的な id に変える簡易トークナイザ(ダウンロード不要)。"""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [zlib.crc32(w.encode()) % 900 + 50 for w in text.replace("\n", " \n ").split()]}


def tiny_model():
    from transformers import Qwen3Config, Qwen3Model

    cfg = Qwen3Config(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=8192)
    torch.manual_seed(0)
    return DecisionModel(Qwen3Model(cfg).float(), 64, proj_dim=16, init_scale=10.0).eval()


def block_logits(model, lay):
    with torch.no_grad():
        return model(lay, "cpu", torch.float32)


def test_layout_shapes_and_mask_rules():
    tok = FakeTok()
    lay = build_binary_layout(tok, "Pages", "who wrote it", ["alpha beta", "gamma delta", "eps"])
    n = 3
    assert lay.q_ends.shape == (n,) and lay.opt_ends.shape == (n, 2)
    C = lay.ctx_len
    # 文脈は質問を見ない(因果的)
    assert not lay.allowed[:C, C:].any()
    # 各質問の末尾は、文脈の全体と、自分の質問のトークンだけを見る。他の候補の質問は見ない
    qa, qb = int(lay.q_ends[0]), int(lay.q_ends[1])
    assert lay.allowed[qb, :C].all() and not lay.allowed[qb, qa]
    # 質問の位置 id は、どの候補でも、文脈の直後から始まる(並列)
    first_pos = [int(lay.position_ids[lay.q_ends[i] - (lay.q_ends[i] - (C if i == 0 else int(lay.opt_ends[i - 1][1]) + 1))]) for i in range(n)]
    assert first_pos == [C, C, C]
    # no と yes の選択肢は、既定では互いに見えず、質問は見える
    no_end, yes_end = int(lay.opt_ends[0][0]), int(lay.opt_ends[0][1])
    assert not lay.allowed[yes_end, no_end] and lay.allowed[yes_end, int(lay.q_ends[0])]
    lay2 = build_binary_layout(tok, "Pages", "who wrote it", ["alpha beta", "gamma delta", "eps"], option_link=True)
    assert lay2.allowed[int(lay2.opt_ends[0][1]), int(lay2.opt_ends[0][0])]


def test_candidates_are_independent_given_the_context():
    """ある候補の質問のトークンを書き換えても、他の候補の質問のロジットは変わらない(共有文脈だけで決まる)。"""
    model, tok = tiny_model(), FakeTok()
    lay = build_binary_layout(tok, "Paragraphs", "what is the capital", ["paris is a city", "rome is a city", "bonn was a capital"])
    base = block_logits(model, lay)
    C = lay.ctx_len
    lo, hi = int(lay.opt_ends[0][1]) + 1, int(lay.opt_ends[1][1]) + 1       # 候補 2 番目(添字 1)の質問のトークン範囲
    ids2 = lay.input_ids.clone()
    ids2[lo:hi] = torch.randint(50, 900, (hi - lo,))
    lay_mod = type(lay)(ids2, lay.position_ids, lay.allowed, lay.q_ends, lay.opt_ends, lay.ctx_len)
    mod = block_logits(model, lay_mod)
    assert torch.allclose(base[0], mod[0], atol=1e-5) and torch.allclose(base[2], mod[2], atol=1e-5)
    assert not torch.allclose(base[1], mod[1], atol=1e-5)                    # 書き換えた候補自身は変わる
    assert C > 0


def test_context_is_shared_not_per_candidate():
    """1 回の請求で採点した確率は、同じ文脈を使う。文脈(候補の説明)を変えると、全候補の確率が変わる。"""
    model, tok = tiny_model(), FakeTok()
    a = build_binary_layout(tok, "Pages", "q one", ["x y", "z w"])
    b = build_binary_layout(tok, "Pages", "q one", ["x y", "different text here"])
    la, lb = block_logits(model, a), block_logits(model, b)
    assert la.shape == (2, 2) and not torch.allclose(la[0], lb[0], atol=1e-6)


def test_choice_layout_and_probabilities():
    model, tok = tiny_model(), FakeTok()
    lay = build_choice_layout(tok, "Paragraphs", "q", ["a b", "c d", "e f", "g h"])
    assert lay.q_ends.shape == (1,) and lay.opt_ends.shape == (1, 4)
    # 選択肢どうしは互いに見えない
    assert not lay.allowed[int(lay.opt_ends[0][2]), int(lay.opt_ends[0][1])]
    logits = block_logits(model, lay)
    assert torch.isclose(torch.softmax(logits[0], -1).sum(), torch.tensor(1.0))


def test_judge_interface_and_split_invariance():
    model, tok = tiny_model(), FakeTok()
    j = DecisionJudge(model, tok, device="cpu", dtype=torch.float32, max_tokens=10 ** 6)
    nodes = [Node(id=f"n{i}", text=f"text number {i} about topic {i % 3}", level=3) for i in range(6)]
    p = j.binary("question words", nodes)
    assert len(p) == 6 and all(0.0 <= x <= 1.0 for x in p) and j.stats["requests"] == 1
    c = j.choice("question words", nodes)
    assert abs(sum(c) - 1.0) < 1e-5 and len(c) == 6
    # 入力が長すぎるときの自動分割は、確率を返す個数を変えない
    j2 = DecisionJudge(model, tok, device="cpu", dtype=torch.float32, max_tokens=60)
    assert len(j2.binary("question words", nodes)) == 6 and j2.stats["requests"] >= 2


def test_scale_is_capped():
    model = tiny_model()
    with torch.no_grad():
        model.log_scale.fill_(10.0)
    assert float(model.scale) == 100.0


def test_train_loop_reduces_loss_on_tiny_model():
    """小さなランダムモデルで、学習ループが動き、同じ少数の例への損失が下がる(CPU、数秒)。"""
    from train_judge import train_loop

    model, tok = tiny_model(), FakeTok()
    ex = []
    for i in range(6):
        texts = [f"alpha {i} one", f"beta {i} two", f"gamma {i} three", f"delta {i} four"]
        ex.append({"kind": "binary", "header": "Pages", "query": f"find beta {i}", "texts": texts, "labels": [0, 1, 0, 0]})
        ex.append({"kind": "choice", "header": "Paragraphs", "query": f"find gamma {i}", "texts": texts, "label": 2})
    res = train_loop(model, tok, ex, None, {}, epochs=40, lr_backbone=3e-3, lr_head=3e-3, batch=4, max_tokens=4000, device="cpu", log=lambda *_: None)
    assert res["steps"] == 40 * 3 and res["loss_last"] < 0.6 * res["loss_first"]
    # 学習後、同じ入力での P(yes) が、正解の候補で最大になる
    lay = build_binary_layout(tok, "Pages", "find beta 0", ex[0]["texts"])
    with torch.no_grad():
        p = torch.softmax(model(lay, "cpu", torch.float32), -1)[:, 1]
    assert int(p.argmax()) == 1
