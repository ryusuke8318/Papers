"""HotpotQA(distractor)・2WikiMultihopQA・MuSiQue を、共通の形に直す。

質問 = {"id", "question", "answer", "type", "paras": [{"id", "text"}...], "gold": [段落 id...]}
段落(チャンク)の text は「タイトル: 本文」の形(論文の図 3 と同じ)。id は text のハッシュ。
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset


def chunk_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _para(title: str, body: str) -> dict:
    text = f"{title}: {body.strip()}"
    return {"id": chunk_id(text), "title": title, "text": text}


def _as_dict(x):
    return x if isinstance(x, dict) else ast.literal_eval(x)


def _from_titles(r: dict, ctx: dict, sf: dict) -> dict | None:
    paras, by_title = [], {}
    for t, sents in zip(ctx["title"], ctx["sentences"]):
        p = _para(t, " ".join(s.strip() for s in sents))
        paras.append(p)
        by_title.setdefault(t, p["id"])
    gold = []
    for t in dict.fromkeys(sf["title"]):
        if t not in by_title:
            return None
        gold.append(by_title[t])
    return {"id": r["id"], "question": r["question"], "answer": r["answer"], "type": r.get("type", ""),
            "paras": [{"id": p["id"], "text": p["text"]} for p in paras], "gold": gold}


def load_questions(name: str, split: str) -> list[dict]:
    """split: "train" または "validation"。"""
    out = []
    if name == "hotpotqa":
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
        for r in ds:
            q = _from_titles(r, _as_dict(r["context"]), _as_dict(r["supporting_facts"]))
            if q:
                out.append(q)
    elif name == "2wiki":
        ds = load_dataset("framolfese/2WikiMultihopQA", split=split)
        for r in ds:
            q = _from_titles(r, _as_dict(r["context"]), _as_dict(r["supporting_facts"]))
            if q:
                out.append(q)
    elif name == "musique":
        ds = load_dataset("dgslibisey/MuSiQue", split=split)
        for r in ds:
            if not r["answerable"]:
                continue
            paras = [{"id": chunk_id(f"{p['title']}: {p['paragraph_text'].strip()}"), "text": f"{p['title']}: {p['paragraph_text'].strip()}"} for p in r["paragraphs"]]
            gold = [chunk_id(f"{p['title']}: {p['paragraph_text'].strip()}") for p in r["paragraphs"] if p["is_supporting"]]
            out.append({"id": r["id"], "question": r["question"], "answer": r["answer"], "type": f"{len(r['question_decomposition'])}hop",
                        "paras": paras, "gold": list(dict.fromkeys(gold))})
    else:
        raise ValueError(name)
    return out


def build_splits(name: str, n_train: int, n_eval: int, corpus_questions: int, seed: int = 0) -> dict:
    """学習用: train から n_train 問。評価: validation から n_eval 問。検索コーパス: validation の corpus_questions 問の全段落(評価質問を含む)。"""
    rng = random.Random(seed)
    tr = load_questions(name, "train")
    dev = load_questions(name, "validation")
    rng.shuffle(tr)
    rng.shuffle(dev)
    train = tr[:n_train]
    corp_qs = dev[:max(corpus_questions, n_eval)]
    ev = corp_qs[:n_eval]
    corpus = {}
    for q in corp_qs:
        for p in q["paras"]:
            corpus[p["id"]] = p["text"]
    return {"train": train, "eval": ev, "corpus": corpus,
            "train_corpus": {p["id"]: p["text"] for q in train for p in q["paras"]}}


def save_jsonl(rows, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def load_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
