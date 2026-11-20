"""判定モデルの学習と、実験の実行(Modal でも、Colab CLI や手元の GPU でも、同じスクリプトで動く)。

  python decisionrag_mini/run_local.py prepare --work work                       # SQuAD から木・問い・学習例を作る(CPU)
  python decisionrag_mini/run_local.py train   --work work --out work/judge      # 判定モデルの学習(GPU)
  python decisionrag_mini/run_local.py eval    --work work --judge work/judge --exps main,pool_scaling
  python decisionrag_mini/run_local.py eval    --work work --judge lexical --exps main     # GPU なしの配線確認(語彙の重なりで採点)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def prepare(work: Path, n_clusters: int = 20, n_train_questions: int = 4000, seed: int = 0, rows=None) -> dict:
    from data import Retrievers, build_corpus, cluster_articles, load_squad, tree_spec
    from train_judge import make_examples

    work.mkdir(parents=True, exist_ok=True)
    tr, dv = rows if rows else (load_squad("train"), load_squad("validation"))
    corpus, questions = build_corpus(tr, dv)
    assign, descs = cluster_articles(corpus, n_clusters, seed=seed)
    spec = tree_spec(corpus, assign, descs)
    (work / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False))
    (work / "questions.json").write_text(json.dumps(questions, ensure_ascii=False))
    (work / "tree.json").write_text(json.dumps(spec, ensure_ascii=False))
    (work / "clusters.json").write_text(json.dumps({"assign": assign, "descs": descs}, ensure_ascii=False))
    retr = Retrievers(corpus)
    ex = make_examples(corpus, questions, assign, descs, retr, n_train_questions, seed)
    (work / "train_examples.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in ex), encoding="utf-8")
    return {"articles": len(corpus["articles"]), "paragraphs": sum(len(a["paras"]) for a in corpus["articles"]), "questions": len(questions),
            "clusters": len(descs), "train_examples": len(ex)}


def load_ctx(work: Path):
    from experiments import Ctx

    return Ctx(json.loads((work / "corpus.json").read_text()), json.loads((work / "questions.json").read_text()), json.loads((work / "tree.json").read_text()))


def get_judge(spec: str, device: str = "cuda"):
    if spec == "lexical":
        from experiments import LexicalJudge

        return LexicalJudge()
    import torch

    from judge import DecisionJudge, load_judge_model

    tok, model = load_judge_model(spec, device, torch.bfloat16)
    return DecisionJudge(model, tok, device, torch.bfloat16)


def run_eval(work: Path, judge, exps: list[str], n_eval: int, extra: dict | None = None, result: str = "") -> dict:
    """実験を順に実行する。result を渡すと、1 つ終わるごとに結果を書き出す(途中で止まっても、終わった実験の結果は残る)。"""
    from experiments import EXPERIMENTS

    ctx = load_ctx(work)
    qs = ctx.dev(n_eval)
    out = {"setup": {"articles": len(ctx.corpus["articles"]), "clusters": ctx.n_clusters, "pages": ctx.n_pages, "paragraphs": ctx.n_paras, "n_eval": len(qs)}}
    for name in exps:
        t0 = time.time()
        fn = EXPERIMENTS[name]
        res = fn(ctx, judge) if name == "latency" else fn(ctx, judge, qs[:(150 if name in ("pool_scaling", "request_size") else n_eval)], **(extra or {}).get(name, {}))
        out[name] = res
        print(f"[{name}] {time.time() - t0:.0f}s", flush=True)
        if result:
            Path(result).parent.mkdir(parents=True, exist_ok=True)
            Path(result).write_text(json.dumps(out, ensure_ascii=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train", "eval"])
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="work/judge")
    ap.add_argument("--judge", default="work/judge")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exps", default="main")
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--n-train-questions", type=int, default=4000)
    ap.add_argument("--result", default="")
    a = ap.parse_args()
    work = Path(a.work)
    if a.stage == "prepare":
        print(json.dumps(prepare(work, n_train_questions=a.n_train_questions)))
    elif a.stage == "train":
        from train_judge import train

        ex = [json.loads(l) for l in open(work / "train_examples.jsonl", encoding="utf-8")]
        print(json.dumps(train(ex, a.model, a.out, not a.full, a.epochs, 1e-5 if a.full else 1e-4, 1e-4, a.batch, 12000, ckpt_every=100, limit=a.limit)))
    else:
        res = run_eval(work, get_judge(a.judge), a.exps.split(","), a.n_eval, result=a.result)
        print(json.dumps({k: (v.get("summary") if isinstance(v, dict) and "summary" in v else "…") for k, v in res.items() if k != "setup"}, ensure_ascii=False)[:800])
