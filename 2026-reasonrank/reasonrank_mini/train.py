"""学習(コンテナ内で実行)。SFT と、GRPO の 1 ラウンド分の更新。

  sft : {prompt, completion} の jsonl で全パラメータを学習(論文のコールドスタート SFT)
  grpo: rollout の jsonl(窓ごとに G 個の出力と報酬)から、グループ内正規化の advantage で 1 ラウンド更新する。
        PPO 型のクリップ付き方策勾配(ε=0.2)。ミニバッチごとに更新するので、後ろのミニバッチは少し古い方策のサンプルになる。
        参照モデルへの KL 項は使わない(β=0。論文は β を明記していない)。
学習のパラメータは fp32 で持ち、bf16 の autocast で計算する。
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/pkg/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reasonrank_core import group_advantages  # noqa: E402


def load(model: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).cuda()
    m.gradient_checkpointing_enable()
    m.config.use_cache = False
    return tok, m


def encode(tok, prompt: str, completion: str, max_len: int):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(completion + tok.eos_token if not completion.endswith(tok.eos_token) else completion, add_special_tokens=False)["input_ids"]
    if len(p) + len(c) > max_len:
        return None
    return p, c


def token_logps(model, ids, plen):
    """完了部分の各トークンの対数確率(長さ = 完了のトークン数)。"""
    import torch
    import torch.nn.functional as F

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids).logits
    lg = logits[0, plen - 1:-1].float()
    return -F.cross_entropy(lg, ids[0, plen:], reduction="none")


def sft(model: str, data: str, out: str, epochs: int, lr: float, batch: int, max_len: int, ckpt: bool = False):
    import torch

    tok, m = load(model)
    rows = [json.loads(l) for l in open(data, encoding="utf-8")]
    ex = [e for e in (encode(tok, r["prompt"], r["completion"], max_len) for r in rows) if e]
    print(f"[sft] 例 {len(ex)} / {len(rows)}(長さ上限 {max_len} で {len(rows) - len(ex)} 件除外)", flush=True)
    steps = epochs * math.ceil(len(ex) / batch)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    m.train()
    t0, s, losses = time.time(), 0, []
    for ep in range(epochs):
        perm = random.Random(ep).sample(range(len(ex)), len(ex))
        for i in range(0, len(perm), batch):
            mb = [ex[j] for j in perm[i:i + batch]]
            ntok = sum(len(c) for _, c in mb)
            tot = 0.0
            for p, c in mb:
                ids = torch.tensor([p + c]).cuda()
                lp = token_logps(m, ids, len(p))
                loss = -lp.sum() / ntok
                loss.backward()
                tot += float(loss)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            s += 1
            losses.append(tot)
            if s % max(1, steps // 10) == 0 or s == 1:
                print(f"  step {s}/{steps} loss {sum(losses[-10:]) / len(losses[-10:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
        if ep < epochs - 1 and ckpt:
            save_ckpt(m, tok, f"{out}_ep{ep + 1}")   # 最後のエポックは、下の save で out に保存する
    save(m, tok, out)
    return {"examples": len(ex), "steps": steps, "loss_first": sum(losses[:3]) / 3, "loss_last": sum(losses[-3:]) / 3, "sec": round(time.time() - t0)}


def grpo(model: str, rollouts: str, out: str, lr: float, minibatch: int, max_len: int, eps: float = 0.2):
    import torch

    tok, m = load(model)
    groups = [json.loads(l) for l in open(rollouts, encoding="utf-8")]
    items = []   # (prompt_ids, completion_ids, advantage)
    n_zero = 0
    for g in groups:
        adv = group_advantages([s["reward"] for s in g])
        if not any(adv):
            n_zero += 1
            continue    # 全員同じ報酬のグループは勾配が 0
        for s, a in zip(g, adv):
            e = encode(tok, s["prompt"], s["completion"], max_len)
            if e:
                items.append((e[0], e[1], a, id(g)))
    print(f"[grpo] グループ {len(groups)}(勾配あり {len(groups) - n_zero})、サンプル {len(items)}", flush=True)
    if not items:
        save(m, tok, out)
        return {"updates": 0}
    # 古い方策の対数確率(更新前に 1 回だけ計算)
    m.eval()
    old = []
    with torch.no_grad():
        for p, c, _, _ in items:
            old.append(token_logps(m, torch.tensor([p + c]).cuda(), len(p)).detach())
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    order = list(range(len(items)))
    # グループ単位でミニバッチにする(同じグループのサンプルを同じ更新に入れる)
    by_group: dict[int, list[int]] = {}
    for i, it in enumerate(items):
        by_group.setdefault(it[3], []).append(i)
    gl = list(by_group.values())
    random.Random(0).shuffle(gl)
    t0, updates, clipfrac, kl_sum = time.time(), 0, 0.0, 0.0
    for b in range(0, len(gl), minibatch):
        idxs = [i for g in gl[b:b + minibatch] for i in g]
        ntok = sum(len(items[i][1]) for i in idxs)
        tot_clip = tot_n = 0
        for i in idxs:
            p, c, a, _ = items[i]
            lp = token_logps(m, torch.tensor([p + c]).cuda(), len(p))
            ratio = torch.exp(lp - old[i])
            unclipped, clipped = ratio * a, torch.clamp(ratio, 1 - eps, 1 + eps) * a
            loss = -torch.minimum(unclipped, clipped).sum() / ntok
            loss.backward()
            tot_clip += int(((ratio - 1).abs() > eps).sum())
            tot_n += len(c)
            kl_sum += float((old[i] - lp.detach()).sum())
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        updates += 1
        clipfrac += tot_clip / max(1, tot_n)
    save(m, tok, out)
    return {"updates": updates, "samples": len(items), "clipfrac": clipfrac / max(1, updates), "sec": round(time.time() - t0)}


def save_ckpt(m, tok, out: str):
    """学習中のモデル(fp32)を変えずに、bf16 のコピーを保存する(途中で止まったときの救済用)。"""
    import torch

    Path(out).mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in m.state_dict().items()}
    m.save_pretrained(out, state_dict=sd)
    tok.save_pretrained(out)
    print(f"  [ckpt] {out}", flush=True)


def save(m, tok, out: str):
    import torch

    Path(out).mkdir(parents=True, exist_ok=True)
    m.to(torch.bfloat16).save_pretrained(out)
    tok.save_pretrained(out)


if __name__ == "__main__":
    a = json.loads(sys.argv[1])
    kind = a.pop("task")
    print(json.dumps({"sft": sft, "grpo": grpo}[kind](**a), ensure_ascii=False))
