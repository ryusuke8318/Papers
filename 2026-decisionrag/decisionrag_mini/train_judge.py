"""判定モデルの学習(論文の付録 G を、SQuAD に合わせて縮めたもの)。

学習例は、ナビゲーションの各段と同じ入力にする(推論時の入力と合わせるため):
  cluster   全クラスタの説明を並べ、問いの記事を含むクラスタに yes
  page      クラスタ 3 つ(確率 0.85 で正解のクラスタを含む)の記事のタイトルを並べ、正解の記事に yes
  paragraph 記事 2 つ(確率 0.9 で正解の記事を含む。もう 1 つは BM25 の難しい負例)の全段落を並べ、正解の段落に yes
  choice    正解の段落と、BM25 の難しい負例 9 件を並べ、正解の番号を選ぶ
SQuAD の train の問いだけを使う(評価に使う dev の問いは学習に入れない)。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make_examples(corpus: dict, questions: list[dict], assign: list[int], descs: list[str], retr, n_questions: int, seed: int = 0,
                  para_words: int = 60, p_gold_cluster: float = 0.85, p_gold_page: float = 0.9) -> list[dict]:
    from data import sample_questions, truncate_words

    rng = random.Random(seed)
    arts = {a["id"]: a for a in corpus["articles"]}
    art_cluster = {a["id"]: c for a, c in zip(corpus["articles"], assign)}
    members: dict[int, list[str]] = {}
    for a, c in zip(corpus["articles"], assign):
        members.setdefault(c, []).append(a["id"])
    ptext = {p["id"]: truncate_words(p["text"], para_words) for a in corpus["articles"] for p in a["paras"]}
    clusters = sorted(members)
    out: list[dict] = []
    for q in sample_questions(questions, "train", n_questions, seed):
        gc, ga, gp = art_cluster[q["gold_article"]], q["gold_article"], q["gold_para"]
        # --- cluster 段
        order = clusters[:]
        rng.shuffle(order)
        out.append({"kind": "binary", "header": "Topics", "query": q["query"], "texts": [descs[c] for c in order], "labels": [int(c == gc) for c in order]})
        # --- page 段
        others = [c for c in clusters if c != gc]
        rng.shuffle(others)
        use_gc = rng.random() < p_gold_cluster                            # 正解のクラスタを含めない例も混ぜる(全部 no になる)
        chosen = ([gc] if use_gc else []) + others[:(2 if use_gc else 3)]
        pages = sorted(a for c in chosen for a in members[c])               # 文書の順(id 順)
        out.append({"kind": "binary", "header": "Pages", "query": q["query"], "texts": [arts[a]["title"].replace("_", " ") for a in pages],
                    "labels": [int(a == ga) for a in pages]})
        # --- paragraph 段
        negs = [a for a in retr.top_articles(q["query"], 6) if a != ga]
        use_gold = rng.random() < p_gold_page
        pg = ([ga] if use_gold else []) + negs[:(1 if use_gold else 2)]
        paras = [p["id"] for a in sorted(pg) for p in arts[a]["paras"]]
        out.append({"kind": "binary", "header": "Paragraphs", "query": q["query"], "texts": [ptext[p] for p in paras], "labels": [int(p == gp) for p in paras]})
        # --- choice 段
        hard = [p for p in retr.top_paras(q["query"], 12) if p != gp][:9]
        cand = [gp] + hard
        rng.shuffle(cand)
        out.append({"kind": "choice", "header": "Paragraphs", "query": q["query"], "texts": [ptext[p] for p in cand], "label": cand.index(gp)})
    return out


def train(examples: list[dict], model_name: str, out: str, lora: bool, epochs: int, lr_backbone: float, lr_head: float, batch: int,
          max_tokens: int, ckpt_every: int = 0, device: str = "cuda", pos_weight: float = 1.0, limit: int = 0, log=print) -> dict:
    import torch

    from judge import load_model

    dtype = torch.bfloat16 if lora else torch.float32          # 全パラメータの学習は fp32 で持ち、bf16 の autocast で計算する
    tok, model = load_model(model_name, device, dtype, lora={"r": 64, "alpha": 128} if lora else None, gradient_checkpointing=True)
    meta = {"model": model_name, "lora": lora, "proj_dim": 256}
    return train_loop(model, tok, examples, out, meta, epochs, lr_backbone, lr_head, batch, max_tokens, ckpt_every, device, pos_weight, limit, log)


def train_loop(model, tok, examples: list[dict], out: str | None, meta: dict, epochs: int, lr_backbone: float, lr_head: float, batch: int,
               max_tokens: int, ckpt_every: int = 0, device: str = "cuda", pos_weight: float = 1.0, limit: int = 0, log=print) -> dict:
    """読み込み済みのモデルを学習する。out が None なら保存しない(テスト用)。"""
    import torch
    import torch.nn.functional as F

    from judge import build_binary_layout, build_choice_layout, save_judge

    model.train()
    if limit:
        examples = examples[:limit]
    head_params = list(model.proj.parameters()) + [model.log_scale]
    back_params = [p for p in model.base.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": back_params, "lr": lr_backbone}, {"params": head_params, "lr": lr_head}], betas=(0.9, 0.95), weight_decay=0.0)
    steps_total = epochs * math.ceil(len(examples) / batch)
    warm = min(50, max(1, steps_total // 10))
    base_lrs = [lr_backbone, lr_head]

    def set_lr(step):
        f = (step + 1) / warm if step < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps_total - warm)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f

    t0, step, losses, skipped = time.time(), 0, [], 0
    for ep in range(epochs):
        order = random.Random(ep).sample(range(len(examples)), len(examples))
        for i in range(0, len(order), batch):
            set_lr(step)
            used = 0
            tot = 0.0
            for j in order[i:i + batch]:
                ex = examples[j]
                lay = (build_binary_layout(tok, ex["header"], ex["query"], ex["texts"]) if ex["kind"] == "binary"
                       else build_choice_layout(tok, ex["header"], ex["query"], ex["texts"]))
                if lay.length > max_tokens:
                    skipped += 1                                  # 長すぎる例は、切らずに捨てる(論文も同様)
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    logits = model(lay, device)
                if ex["kind"] == "binary":
                    y = torch.tensor(ex["labels"], device=logits.device)
                    w = torch.where(y == 1, torch.tensor(pos_weight, device=logits.device), torch.tensor(1.0, device=logits.device))
                    loss = (F.cross_entropy(logits, y, reduction="none") * w).mean()
                else:
                    loss = F.cross_entropy(logits, torch.tensor([ex["label"]], device=logits.device))
                (loss / batch).backward()
                tot += float(loss)
                used += 1
            if used:
                torch.nn.utils.clip_grad_norm_(back_params + head_params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                losses.append(tot / used)
            step += 1
            if step % max(1, steps_total // 20) == 0 or step == 1:
                log(f"  step {step}/{steps_total} loss {sum(losses[-10:]) / max(1, len(losses[-10:])):.4f} scale {float(model.scale):.1f} ({time.time() - t0:.0f}s)")
            if out and ckpt_every and step % ckpt_every == 0 and step < steps_total:
                save_judge(model, tok, f"{out}_step{step}", meta)
                log(f"  [ckpt] {out}_step{step}")
    if out:
        save_judge(model, tok, out, meta)
    return {"examples": len(examples), "skipped_too_long": skipped, "steps": step, "loss_first": sum(losses[:5]) / max(1, len(losses[:5])),
            "loss_last": sum(losses[-5:]) / max(1, len(losses[-5:])), "sec": round(time.time() - t0)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", action="store_true", help="全パラメータの学習(既定は LoRA)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    ex = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    print(json.dumps(train(ex, a.model, a.out, not a.full, a.epochs, a.lr_backbone, a.lr_head, a.batch, a.max_tokens, a.ckpt_every, limit=a.limit)))
