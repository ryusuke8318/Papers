"""内外判定の LLM 評価と LoRA 学習(torch / transformers / peft が必要)。

条件(論文 Tab. 5 の構成に対応)
  plain      学習なし・文脈なし
  rag        学習なし + 検索した chunk を挿入(設定は dev で選ぶ)
  mocked     学習なし + 正解の unit を直接挿入(完全検索の上限)
  long       学習なし + 条文・通達の全文を挿入(長文脈)
  ft         LoRA 学習・文脈なし
  rag_ft     LoRA 学習 + 検索 chunk
  mocked_ft  LoRA 学習 + 正解の unit
出力は「根拠 / 理由 / 結論」の 3 行(論文と同じく理由を結論の前に書かせる)。
採点: 結論一致、根拠一致(正解の引用が出力の根拠に含まれる)、両方一致(strict)。
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jp_common import SPLITS, agg, make_contexts, prompt, score, target  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# ---------------------------------------------------------------- モデル
def load(model_name: str, adapter: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return tok, model.eval()


def chat(tok, p: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(tok, model, prompts: list[str], max_new=160, max_tokens=24000) -> list[str]:
    texts = [chat(tok, p) for p in prompts]
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: lens[i])
    out = [""] * len(texts)
    cur: list[int] = []

    def flush(b):
        enc = tok([texts[i] for i in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, i in enumerate(b):
            out[i] = tok.decode(gen[r, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    for i in order:
        if cur and ((len(cur) + 1) * (lens[i] + max_new) > max_tokens or len(cur) >= 32):
            flush(cur)
            cur = []
        cur.append(i)
    if cur:
        flush(cur)
    return out


# ---------------------------------------------------------------- 学習
def train(cfg: dict, cases: list[dict], adapter_dir: Path, log=print) -> dict:
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    tok, base = load(cfg["model"])
    model = get_peft_model(base, LoraConfig(r=cfg.get("rank", 32), lora_alpha=2 * cfg.get("rank", 32), lora_dropout=0.05,
                                            target_modules=TARGETS, task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    inner = model.get_base_model()
    inner.gradient_checkpointing_enable()
    inner.enable_input_require_grads()
    inner.config.use_cache = False
    model.train()
    end = tok("<|im_end|>", add_special_tokens=False)["input_ids"]
    ex = []
    for c in cases:
        p = tok(chat(tok, prompt(c["text"])), add_special_tokens=False)["input_ids"]
        a = tok(target(c), add_special_tokens=False)["input_ids"] + end
        ex.append((p, a))
    bsz, epochs = cfg["batch"], cfg["epochs"]
    steps = epochs * math.ceil(len(ex) / bsz)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.98), weight_decay=0.0)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    log(f"[train] {cfg['name']}: 例 {len(ex)}  epoch {epochs}  step {steps}")
    t0, losses, s = time.time(), [], 0
    for ep in range(epochs):
        perm = rng.permutation(len(ex)).tolist()
        for i in range(0, len(perm), bsz):
            batch = [ex[j] for j in perm[i:i + bsz]]
            n_ans = sum(len(a) for _, a in batch)
            L = max(len(p) + len(a) for p, a in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            lab = torch.full((len(batch), L), -100, dtype=torch.long)
            for r, (p, a) in enumerate(batch):
                n = len(p) + len(a)
                ids[r, :n], att[r, :n] = torch.tensor(p + a), 1
                lab[r, len(p):n] = torch.tensor(a)
            ids, att, lab = ids.cuda(), att.cuda(), lab.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
                sel = lab[:, 1:] != -100
                logits = inner.lm_head(hs[:, :-1][sel])
            loss = F.cross_entropy(logits.float(), lab[:, 1:][sel], reduction="sum") / n_ans
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss))
            s += 1
            if s % max(1, steps // 8) == 0 or s == 1:
                log(f"  step {s}/{steps} loss {np.mean(losses[-10:]):.4f} ({time.time() - t0:.0f}s)")
    model.save_pretrained(str(adapter_dir))
    del model, base, inner
    torch.cuda.empty_cache()
    return {"examples": len(ex), "steps": steps, "sec": round(time.time() - t0), "loss_first": float(np.mean(losses[:5])),
            "loss_last": float(np.mean(losses[-5:]))}


# ---------------------------------------------------------------- 評価
def evaluate(tok, model, data: dict, units: list[dict], conds: list[str], rag: dict, log=print) -> dict:
    by_id = {u["id"]: u for u in units}
    res = {}
    for cond in conds:
        res[cond] = {}
        for sp in SPLITS:
            cs = data[sp]
            t0 = time.time()
            ps = [prompt(c["text"], make_contexts(cond, c, by_id, units, rag)) for c in cs]
            outs = generate(tok, model, ps)
            rows = [score(o, c) for o, c in zip(outs, cs)]
            m = agg(rows)
            m["sec_per_case"] = (time.time() - t0) / len(cs)
            m["by_family"] = {}
            for fam in sorted({c["family"] for c in cs}):
                sub = [r for r, c in zip(rows, cs) if c["family"] == fam]
                m["by_family"][fam] = {"n": len(sub), "acc": sum(r["correct"] for r in sub) / len(sub)}
            if sp == "real":
                m["items"] = [{"id": c["id"], "label": c["label"], "pred": r["pred"], "cite": r["cite"], "out": o[:300]}
                              for c, r, o in zip(cs, rows, outs)]
            else:
                m["examples"] = [{"text": c["text"][:200], "out": o[:300], "label": c["label"]} for c, o in list(zip(cs, outs))[:3]]
            res[cond][sp] = m
            log(f"  [{cond}/{sp}] acc {m['acc']:.3f} cite {m['cite']:.3f} strict {m['strict']:.3f} unparsed {m['unparsed']:.2f}")
    return res


def run_job(cfg: dict, root: str, work_dir: str, log=print) -> dict:
    root = Path(root)
    sys.path.insert(0, str(root))
    data = json.loads((root / "cases.json").read_text())
    units = json.loads((root / "sources" / "corpus.json").read_text())
    rag = json.loads((root / "sources" / "rag_contexts.json").read_text())  # make_rag_contexts.py が作る(検索結果の固定)
    if cfg.get("limit"):  # 動作確認用: 各 split の先頭だけ
        data = {k: (v[:cfg["limit"]] if k in SPLITS else v) for k, v in data.items()}
    if cfg.get("n_train"):
        data["train"] = data["train"][:cfg["n_train"]]
    out = {"cfg": cfg}
    adapter = None
    if cfg["train"]:
        adapter = Path(work_dir) / "adapters" / cfg["name"]
        adapter.mkdir(parents=True, exist_ok=True)
        out["train"] = train(cfg, data["train"], adapter, log)
    tok, model = load(cfg["model"], str(adapter) if adapter else None)
    t0 = time.time()
    out["eval"] = evaluate(tok, model, data, units, cfg["conds"], rag, log)
    out["eval_sec"] = round(time.time() - t0)
    return out
