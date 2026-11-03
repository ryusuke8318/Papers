"""RAFT 縮小再現の共通処理 (Modal とローカルで共用)。modal は import しない。

- データ作成: build_rows (HotpotQA distractor → RAFT 式の学習・評価行)
- 学習・評価: train_and_eval (Qwen2.5-0.5B + LoRA、CPU/GPU どちらも可)
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N_TRAIN = 3000
N_TEST = 300
N_DISTRACTORS = 4
EPOCHS = 2
SEED = 0
MAX_WORDS_PER_DOC = 110
BATCH = 4
MAX_LEN = 2048


# ---------------------------------------------------------------- プロンプト
def as_pairs(context) -> list:
    """context を [(title, [sents])] に正規化 (dict 形式と pair 形式の両対応)。"""
    if isinstance(context, dict):
        return list(zip(context["title"], context["sentences"]))
    return [(t, list(s)) for t, s in context]


def as_facts(sf) -> list:
    if isinstance(sf, dict):
        return list(zip(sf["title"], sf["sent_id"]))
    return [(t, int(i)) for t, i in sf]


def trunc(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def build_cot(question: str, gold_docs: list, quotes: list, answer: str) -> str:
    bits = [f"The question asks: {question}"]
    for title, quote in quotes:
        bits.append(f"According to '{title}', ##begin_quote##{quote}##end_quote##.")
    bits.append(f"Therefore, the answer is {answer}.")
    return f"##Reason: {' '.join(bits)} ##Answer: {answer}"


def rag_prompt(question: str, docs: list) -> str:
    body = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
    return f"Question: {question}\n\nDocuments:\n{body}\n\nAnswer:"


def dsf_prompt(question: str) -> str:
    return f"Question: {question}\n\nAnswer:"


# ---------------------------------------------------------------- データ作成
TRAIN_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


def fetch_split(split: str, cache_dir: Path) -> list:
    """HotpotQA の JSON を取得する (distractor 設定の train/dev)。初回のみ download する。"""
    import urllib.request

    url = TRAIN_URL if split == "train" else DEV_URL
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / url.rsplit("/", 1)[-1]
    if not path.exists():
        print(f"download: {url}")
        urllib.request.urlretrieve(url, path)
    return json.loads(path.read_text())


def build_rows(split: str, n: int, cache_dir: Path, n_dist: int = N_DISTRACTORS,
               seed: int = SEED, max_words: int = MAX_WORDS_PER_DOC) -> list:
    """HotpotQA distractor から n 件の行を作る。1 行 = Q + golden + distractor + CoT。"""
    rng = random.Random(seed)
    data = fetch_split(split, cache_dir)
    data = data[:]  # シャッフル前にコピー
    rng.shuffle(data)
    rows = []
    for ex in data[:n]:
        paras = as_pairs(ex["context"])
        by_title = {t: s for t, s in paras}
        sup = as_facts(ex["supporting_facts"])
        gold_titles = [t for t, _ in sup if t in by_title]
        if not gold_titles:
            continue
        gold_docs = [f"{t}: {trunc(' '.join(by_title[t]), max_words)}"
                     for t in dict.fromkeys(gold_titles)]
        pool = [f"{t}: {trunc(' '.join(s), max_words)}"
                for t, s in paras if t not in set(gold_titles)]
        distractors = rng.sample(pool, min(n_dist, len(pool)))
        docs = gold_docs + distractors
        order = list(range(len(docs)))
        rng.shuffle(order)
        docs = [docs[i] for i in order]
        quotes = []
        for t, i in sup:
            if t in by_title and 0 <= i < len(by_title[t]):
                quotes.append((t, by_title[t][i].strip()))
        rows.append({
            "id": ex.get("id", ex.get("_id")), "question": ex["question"], "answer": ex["answer"],
            "docs": docs, "cot": build_cot(ex["question"], gold_docs, quotes, ex["answer"]),
        })
    return rows


def save_rows(rows: list, path: Path) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False))


def load_rows(path: Path) -> list:
    return json.loads(path.read_text())


# ---------------------------------------------------------------- 評価
def norm(s: str) -> str:
    import re
    import string

    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_f1(pred: str, gold: str) -> tuple:
    pn, gn = norm(pred), norm(gold)
    em = float(pn == gn)
    pt, gt = pn.split(), gn.split()
    if not pt or not gt:
        return em, em
    common = set(pt) & set(gt)
    if not common:
        return em, 0.0
    prec = sum(min(pt.count(w), gt.count(w)) for w in common) / len(pt)
    rec = sum(min(pt.count(w), gt.count(w)) for w in common) / len(gt)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return em, f1


def extract_answer(text: str) -> str:
    import re

    m = re.search(r"##Answer:\s*(.+)", text, re.S)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------- 学習 + 評価
def train_and_eval(kind: str, train: list, test: list,
                   model_id: str = MODEL_ID, epochs: int = EPOCHS,
                   seed: int = SEED, device: str = "cuda",
                   batch: int = BATCH, log=print) -> dict:
    """LoRA 学習して RAG 評価。kind: raft / dsf (dsf は dsf と dsf_rag を評価)。"""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(seed)
    rng = random.Random(seed)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    def encode(prompt: str, target: str):
        full = tok(prompt + " " + target, truncation=True, max_length=MAX_LEN,
                   return_tensors="pt")
        plen = len(tok(prompt, truncation=True, max_length=MAX_LEN)["input_ids"])
        labels = full["input_ids"].clone()
        labels[0, :plen] = -100
        return full["input_ids"].to(device), full["attention_mask"].to(device), labels.to(device)

    for ep in range(epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        nb = 0
        for i in range(0, len(order), batch):
            opt.zero_grad()
            loss_acc = 0.0
            chunk = order[i:i + batch]
            for j in chunk:
                r = train[j]
                if kind == "raft":
                    p = rag_prompt(r["question"], r["docs"])
                    ids, mask, labels = encode(p, r["cot"])
                else:
                    p = dsf_prompt(r["question"])
                    ids, mask, labels = encode(p, r["answer"])
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                (out.loss / len(chunk)).backward()
                loss_acc += out.loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            nb += 1
            if nb % 100 == 0:
                log(f"[{kind}] epoch {ep + 1}/{epochs} batch {nb} loss {loss_acc / len(chunk):.3f}")

    model.eval()
    results = {}

    def run_eval(prompt_fn, tag: str):
        ems, f1s, examples = [], [], []
        for n, r in enumerate(test):
            p = prompt_fn(r)
            ids = tok(p, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=256, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
            pred = extract_answer(text)
            em, f1 = em_f1(pred, r["answer"])
            ems.append(em)
            f1s.append(f1)
            if len(examples) < 5:
                examples.append({"id": r["id"], "pred": pred, "gold": r["answer"]})
            if (n + 1) % 50 == 0:
                log(f"[{kind}/{tag}] eval {n + 1}/{len(test)}")
        return {"variant": tag, "model": model_id, "kind": kind,
                "n_train": len(train), "n_test": len(test),
                "em": sum(ems) / len(ems), "f1": sum(f1s) / len(f1s),
                "wall_s": round(time.time() - t0),
                "examples": examples}

    if kind == "raft":
        results["raft"] = run_eval(lambda r: rag_prompt(r["question"], r["docs"]), "raft")
    else:
        results["dsf"] = run_eval(lambda r: dsf_prompt(r["question"]), "dsf")
        results["dsf_rag"] = run_eval(lambda r: rag_prompt(r["question"], r["docs"]), "dsf_rag")
    return results
