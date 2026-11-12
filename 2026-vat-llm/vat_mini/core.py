"""VAT 縮小再現の共通処理 (Modal とローカルで共用)。modal は import しない。

- データ: data_seed.json (規則 corpus + train/test pool。確定的に生成し同梱。
  ローカル・Modal ともにこれを読む。ネットワーク不要)。
- 学習: train_model 相当 (Qwen2.5-0.5B + LoRA)。kind で条件を変える:
    plain   学習なし・文脈なし (素の GPT-4o の代理)
    rag     学習なし + 検索 chunk 挿入
    mocked  学習なし + 正解 chunk 挿入 (完全検索の上限の代理)
    ft      LoRA 学習 (chain を学習から除外。held-out の代理)・文脈なし
    rag_ft  LoRA 学習 + 検索 chunk 挿入
- 評価: 国名一致 accuracy + タイプ別 + 検索 hit 率。

注意: 規則文・事例はすべて合成 (synthetic paraphrase) であり、実際の
オーストリア VAT 法の条文・実事例ではない。向きの再現だけが目的。
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N_TRAIN = 1200  # chain を除く 4 タイプ x 300
N_TEST = 150  # 5 タイプ x 30
TOP_K = 3
EPOCHS = 2
SEED = 0
KINDS = ("plain", "rag", "mocked", "ft", "rag_ft")
NO_TRAIN = {"plain", "rag", "mocked"}
CHAIN_HELD_OUT = True  # ft の学習から chain を除外する (held-out の代理)
MAX_LEN = 1024
BATCH = 4
GEN_TOKENS = 64

COUNTRIES = ["Österreich", "Deutschland", "Italien", "Kroatien", "Dänemark", "Frankreich"]
TYPES = ["goods", "transport", "b2b", "b2c", "chain"]

STOPWORDS = set(
    "der die das den dem des ein eine einer einem einen eines und oder mit von zu im in am an auf "
    "für ist sind wird werden hat haben eine einem welcher welche welches einer einem vom beim zum zur "
    "sich als auch nicht nur eine der die das".split()
)


def tokens(s: str) -> list:
    return [t for t in re.findall(r"[a-zäöüß]+", s.lower()) if t not in STOPWORDS]


def overlap_score(question: str, doc: str) -> int:
    q = set(tokens(question))
    return sum(1 for t in tokens(doc) if t in q)


def retrieve(question: str, rules: list, top_k: int = TOP_K) -> list:
    docs = [r["cite"] + " " + r["text"] for r in rules]
    scored = sorted(range(len(rules)), key=lambda i: -overlap_score(question, docs[i]))
    return scored[:top_k]


# ---------------------------------------------------------------- プロンプト
def qa_prompt(question: str, contexts: list) -> str:
    ctx = ""
    if contexts:
        body = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        ctx = f"\nRechtsgrundlagen:\n{body}\n"
    return (
        "Du bist ein Steuerberater für österreichisches Umsatzsteuerrecht.\n"
        f"Frage: {question}\n{ctx}"
        "Nenne zuerst das Besteuerungsland, dann einen Satz Begründung.\nAntwort:"
    )


def norm_country(s: str) -> str:
    s = s.lower().replace("oe", "ö").replace("ue", "ü").replace("ae", "ä")
    s = s.replace("ö", "o").replace("ä", "a").replace("ü", "u")
    return "".join(c for c in s if c.isalpha())


def country_hit(pred: str, gold: str) -> float:
    return float(norm_country(gold) in norm_country(pred))


def country_first(pred: str, gold: str) -> float:
    """strict 版: pred 中で最初に言及された国が gold と一致するか。
    質問文の反復 (echo) による偶然の一致を割り引くための副指標。"""
    positions = [(norm_country(pred).find(norm_country(c)), c) for c in COUNTRIES]
    positions = [(p, c) for p, c in positions if p >= 0]
    if not positions:
        return 0.0
    return float(min(positions)[1] == gold)


# ---------------------------------------------------------------- 学習 + 評価
def train_and_eval(kind: str, train_pool: list, test: list, rules: list,
                   model_id: str = MODEL_ID, epochs: int = EPOCHS,
                   seed: int = SEED, device: str = "cuda",
                   dtype: str | None = None, lr: float = 2e-4, log=print) -> dict:
    """LoRA 学習 (ft 系のみ) して評価。戻り値は {kind: out}。"""
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

    rule_by_id = {r["id"]: r["cite"] + ": " + r["text"] for r in rules}
    train = []
    if kind not in NO_TRAIN:
        rows = [r for r in train_pool if not (CHAIN_HELD_OUT and r["type"] == "chain")]
        train = [{"p": qa_prompt(r["question"], []), "t": " " + r["answer"]} for r in rows]
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
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        log(f"[{kind}] model={model_id} lr={lr} epochs={epochs} seed={seed} rows={len(train)}")

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
            for i in range(0, len(order), BATCH):
                opt.zero_grad()
                loss_acc, n_valid = 0.0, 0
                chunk = order[i:i + BATCH]
                for j in chunk:
                    ids, mask, labels = encode(train[j]["p"], train[j]["t"])
                    if (labels != -100).sum() == 0:
                        continue  # 答えが切り落とされた行は除外
                    out_ = model(input_ids=ids, attention_mask=mask, labels=labels)
                    (out_.loss / len(chunk)).backward()
                    loss_acc += out_.loss.item()
                    n_valid += 1
                if n_valid == 0:
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                nb += 1
                if nb % 100 == 0:
                    log(f"[{kind}] epoch {ep + 1}/{epochs} batch {nb} loss {loss_acc / n_valid:.3f}")
    else:
        log(f"[{kind}] no training (eval only)")
    model.eval()
    try:
        model.gradient_checkpointing_disable()  # generate との競合回避
    except AttributeError:
        pass

    @torch.no_grad()
    def generate(question: str, docs: list) -> str:
        ids = tok(qa_prompt(question, docs), truncation=True, max_length=MAX_LEN,
                  return_tensors="pt")["input_ids"].to(device)
        gen = model.generate(ids, max_new_tokens=GEN_TOKENS, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
        return lines[0] if lines else ""

    hits, oks, stricts = 0, 0, 0
    by_type: dict = {}
    examples = []
    for n, r in enumerate(test):
        if kind in ("rag", "rag_ft"):
            order = retrieve(r["question"], rules, TOP_K)
            docs = [rules[i]["cite"] + ": " + rules[i]["text"] for i in order]
            hits += int(r["gold_rule"] in [rules[i]["id"] for i in order])
        elif kind == "mocked":
            docs = [rule_by_id[r["gold_rule"]]]
            hits += 1
        else:
            docs = []
        pred = generate(r["question"], docs)
        ok = country_hit(pred, r["gold"])
        oks += ok
        stricts += country_first(pred, r["gold"])
        st = by_type.setdefault(r["type"], [0, 0])
        st[0] += ok
        st[1] += 1
        if len(examples) < 3:
            examples.append({"id": r["id"], "type": r["type"], "pred": pred,
                             "gold": r["gold"], "ok": bool(ok)})
        if (n + 1) % 50 == 0:
            log(f"[{kind}] eval {n + 1}/{len(test)}")
    out = {"kind": kind, "model": model_id, "n_train_rows": len(train),
           "n_test": len(test), "wall_s": round(time.time() - t0),
           "acc": oks / max(1, len(test)),
           "acc_strict": stricts / max(1, len(test)),
           "acc_by_type": {t: v[0] / v[1] for t, v in sorted(by_type.items())},
           "hit_rate": hits / max(1, len(test)) if kind in ("rag", "rag_ft", "mocked") else None,
           "examples": examples}
    log(f"[{kind}] acc={out['acc']:.3f} strict={out['acc_strict']:.3f} hit={out['hit_rate']} by_type=" +
        " ".join(f"{t}={v:.2f}" for t, v in out["acc_by_type"].items()))
    return {kind: out}


def load_seed(seed_path: Path):
    return json.loads(Path(seed_path).read_text())
