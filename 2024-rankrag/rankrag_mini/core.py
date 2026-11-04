"""RankRAG 縮小再現の共通処理 (Modal とローカルで共用)。modal は import しない。

- データ作成: build_rows (HotpotQA distractor → Q + 文書10件 + gold 位置)
- 学習: train_model (Qwen2.5-0.5B + LoRA)。kind で ranking データ量を変える:
    qa_only    生成のみで学習 (ranking データ 0。ChatQA-1.5 の代理)
    rank_small 生成 + ranking 400 件 (QA 4000 の 1 割。論文の少量+統一の代理)
    rank_full  生成 + ranking 4000 件 (1:1)
    specialist ranking のみ 8000 件で学習 (別立ての専門 ranker の代理。生成は未学習)
    rankonly4k ranking のみ 4000 件 (rank_full と ranking 量マッチ。QA→ranking 転移の対照)
    mix_matched10 QA3600 + ranking 400 件 = 4000 行 (qa_only と総量マッチ)
    mix_half QA2000 + ranking 2000 件 = 4000 行 (総量マッチの 1:1 混合)
- 評価: evaluate (語彙 overlap の弱 retriever 順 vs 自身の True/False スコア順で top-3 生成 EM/F1)
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N_TRAIN = 4000
N_TEST = 150
N_DOCS = 10          # HotpotQA distractor の文脈数 (gold 2 + distractor 8)
K_GEN = 3            # 生成に使う top-k (論文は 5。mini では 3)
EPOCHS = 2
SEED = 0
# kind ごとの ranking 行上限。論文の Stage-II (ranking は全体の約 1 割) に寄せ、
# rank_small は QA 4000 に対し 400 (1 割)、rank_full は 4000 (1:1)、
# specialist は QA なしで ranking のみ 8000 (別立ての専門 ranker の代理)。
RANK_CAPS = {"qa_only": 0, "rank_small": 400, "rank_full": 4000, "specialist": 8000,
              "rankonly4k": 4000, "mix_matched10": 400, "mix_half": 2000}
# 総行数を揃える対照用の QA 上限。mix_matched10 は QA3600+ranking400=4000 行で
# qa_only (4000 行) と総量マッチ、mix_half は QA2000+ranking2000=4000 行。
QA_CAPS = {"mix_matched10": 3600, "mix_half": 2000}
# 生成学習なし (ranking のみ) の kind。別立ての専門 ranker の代理。
NO_QA = {"specialist", "rankonly4k"}
MAX_WORDS_PER_DOC = 60
BATCH = 4
MAX_LEN = 1024
GEN_TOKENS = 64


# ---------------------------------------------------------------- プロンプト
def rag_prompt(question: str, docs: list) -> str:
    body = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
    return f"Question: {question}\n\nDocuments:\n{body}\n\nAnswer:"


def rank_prompt(question: str, doc: str) -> str:
    return (f"Passage: {doc}\n\nQuestion: {question}\n"
            f"Is the passage relevant to the question? Answer with True or False.\n\nAnswer:")


# ---------------------------------------------------------------- データ作成
# HotpotQA の取得元 (curtis.ml.cmu.edu) は名前解決できないため、
# HF Hub キャッシュから固定 subset を切り出した data_seed.json (train 2000 / validation 300、
# seed 0 でシャッフル済み) を同梱し、ローカル・Modal ともにこれを読む。ネットワーク不要。


def load_seed(seed_path: Path) -> dict:
    return json.loads(Path(seed_path).read_text())


def trunc(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def build_rows(split: str, n: int, seed_data: dict, seed: int = SEED,
               max_words: int = MAX_WORDS_PER_DOC) -> list:
    """seed subset から n 件の行を作る。gold 文書の位置 (gold_idx) を保持する。"""
    rng = random.Random(seed)
    data = seed_data[split][:]
    rng.shuffle(data)
    rows = []
    for ex in data:
        if len(rows) >= n:
            break
        ctx = ex["context"]
        if isinstance(ctx, dict):
            paras = list(zip(ctx["title"], ctx["sentences"]))
        else:
            paras = [(t, list(s)) for t, s in ctx]
        if len(paras) < N_DOCS:
            continue  # 10 件未満は検索が自明になるため除外
        by_title = {t: s for t, s in paras}
        sf = ex["supporting_facts"]
        if isinstance(sf, dict):
            sup = list(zip(sf["title"], sf["sent_id"]))
        else:
            sup = [(t, int(i)) for t, i in sf]
        gold_titles = [t for t, _ in sup if t in by_title]
        if not gold_titles:
            continue
        docs, is_gold = [], []
        for t, s in paras[:N_DOCS]:
            docs.append(f"{t}: {trunc(' '.join(s), max_words)}")
            is_gold.append(t in set(gold_titles))
        if sum(is_gold) == 0:
            continue
        order = list(range(len(docs)))
        rng.shuffle(order)
        rows.append({
            "id": ex.get("id", ex.get("_id")), "question": ex["question"],
            "answer": ex["answer"],
            "docs": [docs[i] for i in order],
            "gold_idx": [k for k, i in enumerate(order) if is_gold[i]],
        })
    return rows


def build_rank_rows(rows: list, n_max: int, seed: int = SEED) -> list:
    """(質問, 文書, True/False) の ranking 行を作る。gold→True、distractor→False。"""
    rng = random.Random(seed)
    pool = []
    for r in rows:
        golds = [r["docs"][i] for i in r["gold_idx"]]
        negs = [d for k, d in enumerate(r["docs"]) if k not in set(r["gold_idx"])]
        for d in golds:
            pool.append({"question": r["question"], "doc": d, "label": "True"})
        for d in rng.sample(negs, min(2, len(negs))):
            pool.append({"question": r["question"], "doc": d, "label": "False"})
    rng.shuffle(pool)
    return pool[:n_max]


def save_rows(rows: list, path: Path) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False))


def load_rows(path: Path) -> list:
    return json.loads(path.read_text())


# ---------------------------------------------------------------- 弱 retriever と評価
def overlap_score(question: str, doc: str) -> float:
    """語彙 overlap F1 (BM25-lite のつもりの弱 retriever。仮定: 真の retriever より弱い)。"""
    q, d = set(question.lower().split()), doc.lower().split()
    if not q or not d:
        return 0.0
    common = sum(min(list(q).count(w), 0) + d.count(w) for w in q if w in d)
    if not common:
        return 0.0
    prec = common / len(d)
    rec = common / len(q)
    return 2 * prec * rec / (prec + rec)


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


def auroc(pos: list, neg: list) -> float:
    """gold 文書スコア群 vs distractor スコア群の分離度 (Mann-Whitney 式)。
    sklearn 不要の自前実装。recall@3 より感度が高く、少件数でも差が見える。"""
    if not pos or not neg:
        return 0.5
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


# ---------------------------------------------------------------- 学習 + 評価
def train_and_eval(kind: str, train_qa: list, train_rank_pool: list, test: list,
                   model_id: str = MODEL_ID, epochs: int = EPOCHS,
                   seed: int = SEED, device: str = "cuda",
                   batch: int = BATCH, log=print, dtype: str | None = None) -> dict:
    """LoRA 学習して評価。kind: qa_only / rank_small / rank_full。
    dtype: None なら fp16 (cuda) / fp32 (cpu)。Modal 側では bf16 を指定する。"""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(seed)
    rng = random.Random(seed)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if dtype is None:
        dt = torch.float16 if device.startswith("cuda") else torch.float32
    else:
        dt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt)
    model.to(device)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()  # checkpointing+LoRA の backward に必要
    except AttributeError:
        pass
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    if kind in NO_QA:
        train = []  # ranking のみ (生成の学習なし)
    else:
        qa_rows = train_qa
        if kind in QA_CAPS:  # 総行数マッチ対照用に QA を seed 決定的に subsample
            qa_rows = rng.sample(train_qa, min(QA_CAPS[kind], len(train_qa)))
        train = [{"p": rag_prompt(r["question"], r["docs"]), "t": " " + r["answer"]}
                 for r in qa_rows]
    cap = RANK_CAPS.get(kind, 0)
    if cap:
        rank_rows = rng.sample(train_rank_pool, min(cap, len(train_rank_pool)))
        train += [{"p": rank_prompt(r["question"], r["doc"]), "t": " " + r["label"]}
                  for r in rank_rows]
        rng.shuffle(train)
    log(f"[{kind}] train rows: {len(train)}")

    def encode(prompt: str, target: str):
        full = tok(prompt + target, truncation=True, max_length=MAX_LEN,
                   return_tensors="pt")
        plen = len(tok(prompt, truncation=True, max_length=MAX_LEN)["input_ids"])
        labels = full["input_ids"].clone()
        labels[0, :plen] = -100
        return full["input_ids"].to(device), full["attention_mask"].to(device), labels.to(device)

    for ep in range(epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        nb = 0
        n_skip = 0
        for i in range(0, len(order), batch):
            opt.zero_grad()
            loss_acc, n_valid = 0.0, 0
            chunk = order[i:i + batch]
            for j in chunk:
                ids, mask, labels = encode(train[j]["p"], train[j]["t"])
                if (labels != -100).sum() == 0:
                    # プロンプトだけで MAX_LEN を超え、答えが切り落とされた行。
                    # 学習にならず loss=nan の元なので除外する
                    n_skip += 1
                    continue
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                (out.loss / len(chunk)).backward()
                loss_acc += out.loss.item()
                n_valid += 1
            if n_valid == 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            nb += 1
            if nb % 100 == 0:
                log(f"[{kind}] epoch {ep + 1}/{epochs} batch {nb} loss {loss_acc / n_valid:.3f}")
        if n_skip:
            log(f"[{kind}] epoch {ep + 1}/{epochs} skipped {n_skip} rows (prompt over MAX_LEN)")

    # ---- True/False スコア (最終位置 logit の差。単トークンの場合のみ厳密)
    tid = tok("True", add_special_tokens=False)["input_ids"]
    fid = tok("False", add_special_tokens=False)["input_ids"]
    single = len(tid) == 1 and len(fid) == 1
    if not single:
        log(f"[{kind}] 注意: True/False が単トークンでないため生成テキストで判定する")
    model.eval()
    try:
        model.gradient_checkpointing_disable()  # generate との競合回避
    except AttributeError:
        pass

    @torch.no_grad()
    def rel_score(question: str, doc: str) -> float:
        ids = tok(rank_prompt(question, doc), truncation=True, max_length=MAX_LEN,
                  return_tensors="pt")["input_ids"].to(device)
        if single:
            lp = torch.log_softmax(model(ids).logits[0, -1], dim=-1)
            return float(lp[tid[0]] - lp[fid[0]])
        gen = model.generate(ids, max_new_tokens=4, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        if "True" in text and "False" not in text:
            return 1.0
        if "False" in text and "True" not in text:
            return -1.0
        return 0.0

    @torch.no_grad()
    def generate(question: str, docs: list) -> str:
        ids = tok(rag_prompt(question, docs), truncation=True, max_length=MAX_LEN,
                  return_tensors="pt")["input_ids"].to(device)
        gen = model.generate(ids, max_new_tokens=GEN_TOKENS, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
        return lines[-1] if lines else ""

    def gold_recall(order: list, gold_idx: list, k: int) -> float:
        hit = sum(1 for i in gold_idx if i in order[:k])
        return hit / max(1, len(gold_idx))

    stats = {}
    apos, aneg = [], []  # reranked スコアの gold/distractor 分離 (AUROC 用)
    for tag in ("norerank", "reranked"):
        ems, f1s, recs, examples = [], [], [], []
        for n, r in enumerate(test):
            if tag == "norerank":
                order = sorted(range(len(r["docs"])),
                               key=lambda i: -overlap_score(r["question"], r["docs"][i]))
            else:
                scored = [(rel_score(r["question"], d), -i)
                          for i, d in enumerate(r["docs"])]
                scored.sort(reverse=True)
                order = [-s[1] for s in scored]
                goldset = set(r["gold_idx"])
                apos.extend(sc for sc, ni in scored if -ni in goldset)
                aneg.extend(sc for sc, ni in scored if -ni not in goldset)
            topk = [r["docs"][i] for i in order[:K_GEN]]
            pred = generate(r["question"], topk)
            em, f1 = em_f1(pred, r["answer"])
            ems.append(em)
            f1s.append(f1)
            recs.append(gold_recall(order, r["gold_idx"], K_GEN))
            if len(examples) < 3:
                examples.append({"id": r["id"], "pred": pred, "gold": r["answer"]})
            if (n + 1) % 25 == 0:
                log(f"[{kind}/{tag}] eval {n + 1}/{len(test)}")
        stats[tag] = {"em": sum(ems) / len(ems), "f1": sum(f1s) / len(f1s),
                      "recall": sum(recs) / len(recs), "examples": examples}

    out = {"kind": kind, "model": model_id, "n_train_qa": len(train_qa),
           "n_train_rows": len(train), "n_test": len(test),
           "wall_s": round(time.time() - t0)}
    for tag, s in stats.items():
        out[tag] = s
        log(f"[{kind}/{tag}] EM={s['em']:.3f} F1={s['f1']:.3f} recall@{K_GEN}={s['recall']:.3f}")
    out["reranked"]["auroc"] = auroc(apos, aneg)
    log(f"[{kind}/reranked] AUROC={out['reranked']['auroc']:.3f}")
    return {f"{kind}": out}
