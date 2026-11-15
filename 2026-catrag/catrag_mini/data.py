"""HotpotQA(distractor)・2WikiMultiHopQA・MuSiQue から、HippoRAG と同じ方式の小規模コーパスを作る。

質問を n 件抽出し、その候補段落(支持+ distractor)の和集合をコーパスにする。
取得元(Hugging Face): hotpotqa/hotpot_qa (distractor)、framolfese/2WikiMultihopQA、bdsaglam/musique (answerable)。
いずれも validation を使う。
"""
from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path

WORK = Path(__file__).resolve().parent / "work"
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def _load(name: str):
    from datasets import load_dataset
    if name == "hotpot":
        return load_dataset("hotpotqa/hotpot_qa", "distractor")["validation"]
    if name == "2wiki":
        return load_dataset("framolfese/2WikiMultihopQA")["validation"]
    if name == "musique":
        return load_dataset("bdsaglam/musique", "answerable")["validation"]
    raise ValueError(name)


def _paragraphs(name: str, ex) -> tuple[list[tuple[str, list[str]]], set[str], str]:
    """(段落 [(title, sents)], 正解の支持段落の title 集合, 答え) を返す。"""
    if name in ("hotpot", "2wiki"):
        ctx = ex["context"]
        paras = list(zip(ctx["title"], ctx["sentences"]))
        sup = set(ex["supporting_facts"]["title"])
        return paras, sup, ex["answer"]
    paras, sup = [], set()
    for p in ex["paragraphs"]:
        key = p["title"]
        paras.append((key, [s for s in _SPLIT.split(p["paragraph_text"]) if s.strip()]))
        if p["is_supporting"]:
            sup.add(key)
    return paras, sup, ex["answer"]


def make_corpus(name: str, n: int, seed: int, skip: set[str] | None = None) -> dict:
    ds = _load(name)
    ids = list(range(len(ds)))
    random.Random(seed).shuffle(ids)
    passages: list[dict] = []
    pid: dict[tuple[str, str], int] = {}
    questions = []
    for i in ids:
        if len(questions) >= n:
            break
        ex = ds[i]
        if skip and ex["id"] in skip:
            continue
        paras, sup, ans = _paragraphs(name, ex)
        if name == "musique" and not ex["answerable"]:
            continue
        gold = set()
        for title, sents in paras:
            text = " ".join(s.strip() for s in sents)
            key = (title, text)
            if key not in pid:
                pid[key] = len(passages)
                passages.append({"id": len(passages), "title": title, "sents": [s.strip() for s in sents if s.strip()], "text": text})
            if title in sup:
                gold.add(pid[key])
        if len(gold) != len(sup) or not gold:
            continue
        questions.append({"id": ex["id"], "question": ex["question"], "answer": ans, "gold": sorted(gold), "type": str(ex.get("type", ""))})
    return {"name": name, "passages": passages, "questions": questions}


def get_corpus(name: str, n: int, seed: int, tag: str = "test", skip: set[str] | None = None) -> dict:
    WORK.mkdir(exist_ok=True)
    path = WORK / f"corpus_{name}_{tag}_{n}_{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    c = make_corpus(name, n, seed, skip)
    path.write_text(json.dumps(c, ensure_ascii=False))
    return c
