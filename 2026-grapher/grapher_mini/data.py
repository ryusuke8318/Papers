"""HotpotQA(distractor)・2WikiMultiHopQA・MuSiQue(answerable) の validation から、論文と同じ方式でコーパスを作る。

質問を乱数 (seed 0) で並べ、先頭から [(tag, n), ...] の順に重ならないよう切り出す。
各分割のコーパスは、その分割の質問の候補段落(支持+ distractor)の和集合。段落の同一性は (title, 本文)。
取得元: hotpotqa/hotpot_qa (distractor)、framolfese/2WikiMultihopQA、bdsaglam/musique (answerable)。
"""
from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path

WORK = Path(__file__).resolve().parent / "work"
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _load(name: str):
    from datasets import load_dataset
    if name == "hotpot":
        return load_dataset("hotpotqa/hotpot_qa", "distractor")["validation"]
    if name == "2wiki":
        return load_dataset("framolfese/2WikiMultihopQA")["validation"]
    if name == "musique":
        return load_dataset("bdsaglam/musique", "answerable")["validation"]
    raise ValueError(name)


def _paragraphs(name: str, ex):
    if name in ("hotpot", "2wiki"):
        ctx = ex["context"]
        sup = set(ex["supporting_facts"]["title"])
        return list(zip(ctx["title"], ctx["sentences"])), [t in sup for t in ctx["title"]]
    paras, flags = [], []
    for p in ex["paragraphs"]:
        paras.append((p["title"], [s for s in _SPLIT.split(p["paragraph_text"]) if s.strip()]))
        flags.append(bool(p["is_supporting"]))   # MuSiQue は同じ title の別段落があるので、title ではなく段落ごとの印を使う
    return paras, flags


def _record(name, ex):
    paras, flags = _paragraphs(name, ex)
    return {"id": ex["id"], "question": ex["question"], "answer": ex["answer"], "paras": paras, "flags": flags}


def make_splits(name: str, spec: list[tuple[str, int]], seed: int = 0) -> dict[str, dict]:
    """spec = [("eval", 2000), ("tune", 500), ...]。n が 0 以下なら使える質問すべて(その分割が最後のとき)。"""
    ds = _load(name)
    ids = list(range(len(ds)))
    random.Random(seed).shuffle(ids)
    out, it = {}, iter(ids)
    for tag, n in spec:
        recs = []
        for i in it:
            ex = ds[i]
            if name == "musique" and not ex["answerable"]:
                continue
            recs.append(_record(name, ex))
            if n > 0 and len(recs) >= n:
                break
        out[tag] = _corpus(name, recs)
    return out


def _corpus(name, recs):
    passages, pid, questions = [], {}, []
    for r in recs:
        gold = set()
        for (title, sents), is_sup in zip(r["paras"], r["flags"]):
            text = " ".join(s.strip() for s in sents)
            key = (title, text)
            if key not in pid:
                pid[key] = len(passages)
                passages.append({"id": len(passages), "title": title, "text": text})
            if is_sup:
                gold.add(pid[key])
        if not gold:
            continue
        questions.append({"id": r["id"], "question": r["question"], "answer": r["answer"], "gold": sorted(gold)})
    return {"name": name, "passages": passages, "questions": questions}


def get_split(name: str, tag: str, spec: list[tuple[str, int]], seed: int = 0) -> dict:
    WORK.mkdir(exist_ok=True)
    path = WORK / f"corpus_{name}_{tag}.json"
    if path.exists():
        return json.loads(path.read_text())
    for t, c in make_splits(name, spec, seed).items():
        (WORK / f"corpus_{name}_{t}.json").write_text(json.dumps(c, ensure_ascii=False))
    return json.loads(path.read_text())


SPECS = {
    "hotpot": [("eval", 2000), ("tune", 500), ("train", 2000)],
    "2wiki": [("eval", 2000), ("train", 1000)],
    "musique": [("eval", 0)],
}
