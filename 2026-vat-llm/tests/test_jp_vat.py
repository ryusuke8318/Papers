"""採点・検索・事例生成の不変条件の確認(torch 不要)。  python -m pytest tests/ -q"""
import json
import sys
from pathlib import Path

JP = Path(__file__).resolve().parent.parent / "jp_vat"
sys.path.insert(0, str(JP))
import cases  # noqa: E402
import jp_common as C  # noqa: E402
import retrieval as R  # noqa: E402

UNITS = json.loads((JP / "sources" / "corpus.json").read_text())


def test_score_parsing():
    c = {"label": True, "gold_cites": ["令第6条第1項第7号"]}
    assert C.score("根拠: 消費税法施行令第6条第1項第7号\n理由: x\n結論: 国内取引", c)["strict"]
    assert C.score("根拠: 令第6条第1項第7号\n結論: 国外取引", c)["correct"] is False
    assert C.score("結論がありません", c)["pred"] is None
    assert C.score("結論：国内取引", c)["cite"] is False


def test_cases_are_disjoint_and_cites_exist():
    d = cases.build()
    ids = {u["id"] for u in UNITS}
    train = {c["text"] for c in d["train"]}
    for sp in ("test_seen", "test_exc_seen", "test_exc_unseen", "dev"):
        assert not train & {c["text"] for c in d[sp]}, sp
    for sp, cs in d.items():
        for c in cs:
            assert set(c["gold"]) <= ids, (sp, c["gold"])
            assert c["label"] in (True, False)


def test_unseen_families_not_in_train():
    d = cases.build()
    assert not {c["family"] for c in d["train"]} & {"u_bol", "u_mixed"}


def test_chunks_cover_corpus_and_hit():
    cover = set().union(*[c["units"] for c in R.build_chunks(UNITS, "fixed300")])
    assert len(cover) >= 0.95 * len(UNITS)  # 境界をまたぐ短い unit だけが漏れうる
    ch = R.build_chunks(UNITS, "unit")
    assert R.hit(ch[:1], [ch[0]["text"] and list(ch[0]["units"])[0]])
    r = R.Retriever(UNITS, "unit", "bm25")
    q = "船荷証券の荷揚地が国内である譲渡"
    assert any("5－7－11" in c["text"] for c in r.topk(q, 3))
