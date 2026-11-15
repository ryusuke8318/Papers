"""3 つのデータセット(MuSiQue answerable、2WikiMultiHopQA、HotpotQA distractor)から、検証用の質問を無作為に抽出して JSONL にする。

  python raser_mini/prepare_data.py --n 400

出力: raser_mini/data/<dataset>.jsonl(1 行 = 1 問。question、gold、aliases、paragraphs=[{title, text}]、hops)
どのデータセットも、開発用の分割(validation)から取る。乱数の種は固定する。
"""
from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

HERE = Path(__file__).parent


def _lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def rows_musique(ds):
    for r in ds:
        if str(r["answerable"]) != "True" and r["answerable"] is not True:
            continue
        paras = _lit(r["paragraphs"])
        al = _lit(r["answer_aliases"]) or []
        yield {"id": r["id"], "question": r["question"], "gold": r["answer"], "aliases": list(al),
               "paragraphs": [{"title": p["title"], "text": p["paragraph_text"]} for p in paras],
               "hops": int(r["id"].split("hop")[0][0])}


def _title_sentences(ctx):
    ctx = _lit(ctx)
    if isinstance(ctx, dict):
        titles = ctx["title"]
        sents = ctx.get("sentences") or ctx.get("content")
        return [{"title": t, "text": " ".join(s).strip()} for t, s in zip(titles, sents)]
    return [{"title": t, "text": " ".join(s).strip()} for t, s in ctx]


def rows_hotpot(ds):
    for r in ds:
        yield {"id": r["id"], "question": r["question"], "gold": r["answer"], "aliases": [], "paragraphs": _title_sentences(r["context"]), "hops": 2}


def rows_2wiki(ds):
    for r in ds:
        yield {"id": r["id"], "question": r["question"], "gold": r["answer"], "aliases": [], "paragraphs": _title_sentences(r["context"]), "hops": 2}


def main():
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    specs = [("musique", ("bdsaglam/musique", "answerable"), rows_musique), ("2wiki", ("framolfese/2WikiMultihopQA", None), rows_2wiki),
             ("hotpotqa", ("hotpotqa/hotpot_qa", "distractor"), rows_hotpot)]
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    for name, (repo, cfg), fn in specs:
        ds = load_dataset(repo, cfg, split="validation")
        rows = [r for r in fn(ds) if r["gold"] and r["paragraphs"]]
        random.Random(a.seed).shuffle(rows)
        rows = rows[: a.n]
        (out / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        print(name, repo, "分割の件数", len(ds), "抽出", len(rows), "段落/問", sum(len(r["paragraphs"]) for r in rows) / len(rows))


if __name__ == "__main__":
    main()
