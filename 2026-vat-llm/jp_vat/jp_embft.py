"""bge-m3 の追加学習(LoRA)。事例の本文から、正解の unit を引き当てるようにする(torch / transformers / peft が必要)。

学習: 学習用 800 件の本文をクエリ、gold の unit を正例にする。corpus が 48 unit しかないので、負例は他の全 unit(全 unit に対する softmax)。
出力: 評価に使う全事例 × 全 unit のスコア(results/ft_scores.json)。検索の組み立て(hybrid、graph)は retrieval_exp.py で行う。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EVAL = ("dev", "test_seen", "test_exc_seen", "test_exc_unseen", "real")


def run(root: str, name="BAAI/bge-m3", epochs=3, lr=2e-4, batch=16, rank=16, tau=0.05, seed=0, log=print) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    root = Path(root)
    units = json.loads((root / "sources" / "corpus.json").read_text())
    data = json.loads((root / "cases.json").read_text())
    ids = [u["id"] for u in units]
    utexts = [f"【{u['cite']}】{u['text']}" for u in units]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float32).cuda()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=["query", "value"]))
    model.base_model.model.gradient_checkpointing_enable()
    model.base_model.model.enable_input_require_grads()

    def enc(texts, grad):
        e = tok(texts, padding=True, truncation=True, max_length=320, return_tensors="pt").to("cuda")
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**e).last_hidden_state[:, 0]
        return F.normalize(h.float(), dim=-1)

    train = data["train"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = epochs * math.ceil(len(train) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    model.train()
    t0, s = time.time(), 0
    for ep in range(epochs):
        perm = rng.permutation(len(train)).tolist()
        for i in range(0, len(perm), batch):
            b = [train[j] for j in perm[i:i + batch]]
            q = enc([c["text"] for c in b], True)
            u = enc(utexts, True)
            logp = F.log_softmax(q @ u.T / tau, dim=-1)
            pos = torch.zeros_like(logp)
            for r, c in enumerate(b):
                for g in c["gold"]:
                    pos[r, ids.index(g)] = 1.0
            loss = -(logp * pos).sum() / pos.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            s += 1
            if s % 20 == 0 or s == 1:
                log(f"  step {s}/{steps} loss {float(loss):.4f} ({time.time() - t0:.0f}s)")
    model.eval()
    u = enc(utexts, False)
    scores = {}
    for sp in EVAL:
        qs = [c["text"] for c in data[sp]]
        out = []
        for i in range(0, len(qs), 32):
            out.append((enc(qs[i:i + 32], False) @ u.T).cpu().numpy())
        scores[sp] = np.concatenate(out).round(5).tolist()
    return {"cfg": {"name": f"embft_s{seed}", "epochs": epochs, "lr": lr, "rank": rank, "tau": tau, "seed": seed}, "scores": scores,
            "sec": round(time.time() - t0)}
