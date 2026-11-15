"""学習(コンテナ内で `python train.py '<json>'` として実行)。

build : 合成の結果(synth.jsonl)から、難しい負例を検索して選び、Labeler / Filter の学習例を作る(質問ごとに保存)
fit   : 質問数 n の先頭から学習例を取り、Labeler と Filter を学習して保存する
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/pkg/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from config import ENCODER, MAX_CHUNK_WORDS, TRAIN  # noqa: E402
from efficientrag_core import build_examples, pick_negatives, split_words  # noqa: E402


def build(root: str, neg_per_query: int) -> dict:
    from retriever import Index

    r = Path(root)
    recs = [json.loads(l) for l in (r / "synth.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    corpus = json.loads((r / "train_corpus.json").read_text(encoding="utf-8"))
    ids = list(corpus)
    t0 = time.time()
    index = Index(ids, [corpus[i] for i in ids])
    print(f"index {len(ids)} chunks {time.time() - t0:.0f}s", flush=True)
    rows, tot = [], {}
    for rec in recs:
        gold = {s["doc"] for s in rec["subs"].values()}
        words = {i: split_words(corpus[i])[:MAX_CHUNK_WORDS] for i in gold}

        def neg(sid, query, gold=gold, words=words):
            top = index.search([query], 30)[0]
            ns = pick_negatives(top, gold, neg_per_query)
            for n in ns:
                words[n] = split_words(corpus[n])[:MAX_CHUNK_WORDS]
            return ns
        lab, fil, st = build_examples(rec, words, neg)
        rows.append({"qid": rec["qid"], "labeler": lab, "filter": fil})
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + v
    (r / "examples.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    return {"records": len(recs), **tot, "sec": round(time.time() - t0)}


def fit(root: str, n: int, tag: str, epochs: int, lr: float, only: str | None = None) -> dict:
    from models import train

    r = Path(root)
    rows = [json.loads(l) for l in (r / "examples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:n]
    out = {}
    for kind in ("labeler", "filter"):
        if only and only != kind:
            continue
        exs = [e for row in rows for e in row[kind]]
        t0 = time.time()
        print(f"{tag} {kind}: {len(rows)} 問 / {len(exs)} 例", flush=True)
        model, _ = train(ENCODER, exs, epochs, lr, TRAIN["batch"], TRAIN["max_len"], TRAIN["warmup"])
        path = r / "models" / tag / f"{kind}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()}, path)
        out[kind] = {"examples": len(exs), "sec": round(time.time() - t0), "path": str(path)}
        del model
        torch.cuda.empty_cache()
    return {"questions": len(rows), **out}


if __name__ == "__main__":
    a = json.loads(sys.argv[1])
    if a["task"] == "build":
        print(json.dumps(build(a["root"], a.get("neg_per_query", TRAIN["neg_per_query"]))))
    else:
        print(json.dumps(fit(a["root"], a["n"], a["tag"], a.get("epochs", TRAIN["epochs"]), a.get("lr", TRAIN["lr"]), a.get("only"))))
