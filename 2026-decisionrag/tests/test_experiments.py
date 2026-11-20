import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "decisionrag_mini"))
from run_local import prepare, run_eval, get_judge  # noqa: E402
from test_data import fake_rows  # noqa: E402


def make_work(tmp_path):
    rows = (fake_rows("train", 24), fake_rows("dev", 8, start=24))
    return prepare(tmp_path, n_clusters=4, n_train_questions=20, seed=0, rows=rows)


def test_prepare_writes_files(tmp_path):
    info = make_work(tmp_path)
    assert info["articles"] == 32 and info["paragraphs"] == 32 * 6 and info["clusters"] == 4 and info["train_examples"] == 80
    for f in ("corpus.json", "questions.json", "tree.json", "clusters.json", "train_examples.jsonl"):
        assert (tmp_path / f).exists()


def test_all_experiments_run_with_lexical_judge(tmp_path):
    make_work(tmp_path)
    exps = ["main", "pool_scaling", "request_size", "choice", "budget", "latency"]
    res = run_eval(tmp_path, get_judge("lexical"), exps, n_eval=12,
                   extra={"pool_scaling": {"ns": (5, 25)}, "request_size": {"pool_n": 40, "sizes": (1, 10, 40)}, "budget": {"k1s": (1, 2, None), "k2s": (1, 2)}})
    json.dumps(res)                                              # JSON にできる
    m = res["main"]
    assert set(m) == {"bm25", "bm25_top100_judge", "decisionrag", "decisionrag_bm25"}
    assert all(0.0 <= m[k]["summary"]["hit1"] <= 1.0 for k in m)
    assert m["decisionrag"]["summary"]["stages"] == 4          # クラスタ、記事、段落、複数選択
    assert m["bm25"]["summary"]["judgments"] == 0 and m["decisionrag"]["summary"]["judgments"] > 0
    b = res["budget"]
    assert b["k1=1,k2=1"]["summary"]["judgments"] < b["k1=all,k2=2"]["summary"]["judgments"]
    assert b["k1=all,k2=2"]["summary"]["r10"] >= b["k1=1,k2=1"]["summary"]["r10"] - 1e-9
    assert set(res["pool_scaling"]["5"]["summary"]) >= {"binary_separate", "binary_joint", "choice", "binary_then_choice"}
    assert "1" in res["request_size"] and "choice_all" in res["request_size"]
    assert abs(res["latency"]["analytic_sec_b100"]["1e9"] - 1.0) < 1e-9
