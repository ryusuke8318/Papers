"""データの準備: 学習用の HotpotQA(train, distractor)と、評価用の 3 つの dev から、質問と共通のコーパスを作る。

出力(david_mini/work/):
  train.json            学習用の質問 [{id, question, answer, gold_titles}]
  dev_{hotpot,2wiki,musique}.json  評価用の質問(各 n_dev 件、学習には使わない)
  corpus.json           全質問の候補段落(支持+ distractor)の和集合 [{id, title, text}]
取得元(Hugging Face): hotpotqa/hotpot_qa (distractor)、framolfese/2WikiMultihopQA、bdsaglam/musique (answerable)。
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path

WORK = Path(__file__).resolve().parent / "work"
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def _load(name: str, split: str):
    from datasets import load_dataset
    if name == "hotpot":
        return load_dataset("hotpotqa/hotpot_qa", "distractor")[split]
    if name == "2wiki":
        return load_dataset("framolfese/2WikiMultihopQA")[split]
    if name == "musique":
        return load_dataset("bdsaglam/musique", "answerable")[split]
    raise ValueError(name)


def _paras(name: str, ex):
    """([(title, text)], 支持 title 集合, 答えの別名のリスト)"""
    if name in ("hotpot", "2wiki"):
        ctx = _lit(ex["context"])
        paras = [(t, " ".join(s.strip() for s in ss)) for t, ss in zip(ctx["title"], ctx["sentences"])]
        sup = set(_lit(ex["supporting_facts"])["title"])
        return paras, sup, [ex["answer"]]
    paras, sup = [], set()
    for p in ex["paragraphs"]:
        paras.append((p["title"], p["paragraph_text"]))
        if p["is_supporting"]:
            sup.add(p["title"])
    answers = [ex["answer"]] + list(ex.get("answer_aliases") or [])
    return paras, sup, answers


def build(name: str, split: str, n: int, seed: int, passages: dict):
    ds = _load(name, split)
    ids = list(range(len(ds)))
    random.Random(seed).shuffle(ids)
    qs = []
    for i in ids:
        if len(qs) >= n:
            break
        ex = ds[i]
        if name == "musique" and not ex.get("answerable", True):
            continue
        paras, sup, answers = _paras(name, ex)
        titles = {t for t, _ in paras}
        if not sup or not sup <= titles:
            continue
        for t, text in paras:
            passages.setdefault((t, text), len(passages))
        qs.append({"id": str(ex["id"]), "question": ex["question"], "answer": answers[0], "answers": answers,
                   "gold_titles": sorted(sup), "type": str(ex.get("type", ""))})
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_dev", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    WORK.mkdir(exist_ok=True)
    passages: dict = {}
    train = build("hotpot", "train", a.n_train, a.seed, passages)
    (WORK / "train.json").write_text(json.dumps(train, ensure_ascii=False))
    for name in ("hotpot", "2wiki", "musique"):
        dev = build(name, "validation", a.n_dev, a.seed, passages)
        (WORK / f"dev_{name}.json").write_text(json.dumps(dev, ensure_ascii=False))
        print(name, len(dev))
    corpus = [{"id": i, "title": t, "text": x} for (t, x), i in passages.items()]
    (WORK / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False))
    print("train", len(train), "corpus", len(corpus))


if __name__ == "__main__":
    main()
