"""LLM を使わない名前付き実体の抽出(spaCy en_core_web_sm)。論文は GPT-4o。結果は work/ents_<name>_<tag>.json に保存。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402

DROP = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY"}


def norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.lower().strip())
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return s.strip(" .,;:'\"()")


def extract(texts: list[str], n_process: int = 8) -> list[list[str]]:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "attribute_ruler", "tagger", "senter"])
    out = []
    for doc in nlp.pipe(texts, batch_size=128, n_process=n_process):
        e = sorted({norm(x.text) for x in doc.ents if x.label_ not in DROP} - {""})
        out.append(e)
    return out


def get_ents(name: str, tag: str, passages: list[dict]) -> list[list[str]]:
    path = D.WORK / f"ents_{name}_{tag}.json"
    if path.exists():
        return json.loads(path.read_text())
    e = extract([p["text"] for p in passages])
    path.write_text(json.dumps(e, ensure_ascii=False))
    return e


if __name__ == "__main__":
    import time
    for name in sys.argv[1:]:
        for tag, _ in D.SPECS[name]:
            t = time.time()
            c = D.get_split(name, tag, D.SPECS[name])
            e = get_ents(name, tag, c["passages"])
            n = [len(x) for x in e]
            print(name, tag, len(c["questions"]), "q", len(c["passages"]), "p", "mean ents", sum(n) / len(n), "empty", sum(1 for k in n if k == 0), f"{time.time()-t:.0f}s", flush=True)
