"""Labeler & Tagger と Filter(エンコーダ + 語ごとの 2 値分類 head + 平均プーリングのチャンク tag head)。

入力は 2 つの語列 a, b を [CLS] a [SEP] b [SEP] でつないだもの。
- Labeler: a = 問い, b = チャンク。語ラベルは b の語、tag は <Continue>(1) / <Terminate>(0)。
- Filter : a = 原問い, b = 集めた Info の語。語ラベルは a, b の両方(次の問いに残す語)。tag は使わない。
語ラベルは、語の最初のサブトークンの位置で予測する。
"""
from __future__ import annotations

import random
import time

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


class Tagger(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name).float()
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.tok_head = nn.Linear(h, 2)
        self.tag_head = nn.Linear(h, 2)

    def forward(self, input_ids, attention_mask):
        hs = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        return self.tok_head(self.drop(hs)), self.tag_head(self.drop(pooled))


def collate(tok, exs: list[dict], max_len: int, device: str):
    """学習・推論の共通の入力作り。語 → サブトークンの対応(first[i][(seg, word)] = 位置)も返す。"""
    enc = tok([e["a"] for e in exs], [e["b"] for e in exs], is_split_into_words=True, truncation="only_second", max_length=max_len,
              padding=True, return_tensors="pt")
    n, L = enc["input_ids"].shape
    labels = torch.full((n, L), -100, dtype=torch.long)
    first: list[dict] = []
    for i, e in enumerate(exs):
        wid, sid = enc.word_ids(i), enc.sequence_ids(i)
        seen, pos = set(), {}
        for t, (w, s) in enumerate(zip(wid, sid)):
            if w is None or s is None or (s, w) in seen:
                continue
            seen.add((s, w))
            pos[(s, w)] = t
            lab = e["la"] if s == 0 else e["lb"]
            if lab is not None:
                labels[i, t] = lab[w]
        first.append(pos)
    tags = torch.tensor([(-100 if e.get("tag") is None else e["tag"]) for e in exs], dtype=torch.long)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), labels.to(device), tags.to(device), first


def train(name: str, examples: list[dict], epochs: int, lr: float, batch: int, max_len: int, warmup: float = 0.1, seed: int = 0,
          device: str = "cuda", log=print):
    random.seed(seed)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(name)
    model = Tagger(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(examples) + batch - 1) // batch)
    sch = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    model.train()
    t0, step = time.time(), 0
    for ep in range(epochs):
        order = list(range(len(examples)))
        random.shuffle(order)
        for s in range(0, len(order), batch):
            exs = [examples[i] for i in order[s:s + batch]]
            ids, mask, labels, tags, _ = collate(tok, exs, max_len, device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device == "cuda"):
                tl, gl = model(ids, mask)
            loss = ce(tl.float().reshape(-1, 2), labels.reshape(-1))
            if (tags != -100).any():
                loss = loss + ce(gl.float(), tags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                log(f"  step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s")
    model.eval()
    return model, tok


@torch.no_grad()
def predict(model, tok, exs: list[dict], max_len: int = 512, bs: int = 64, device: str = "cuda"):
    """各例について (P(tag=1), a の語ごとの確率, b の語ごとの確率) を返す。切り詰めで落ちた語の確率は 0。"""
    model.eval()
    res = [None] * len(exs)
    order = sorted(range(len(exs)), key=lambda i: len(exs[i]["a"]) + len(exs[i]["b"]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        batch = [exs[i] for i in idx]
        ids, mask, _, _, first = collate(tok, batch, max_len, device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device == "cuda"):
            tl, gl = model(ids, mask)
        pt = tl.float().softmax(-1)[..., 1].cpu()
        pg = gl.float().softmax(-1)[:, 1].cpu()
        for j, i in enumerate(idx):
            e = exs[i]
            pa = [float(pt[j, first[j][(0, w)]]) if (0, w) in first[j] else 0.0 for w in range(len(e["a"]))]
            pb = [float(pt[j, first[j][(1, w)]]) if (1, w) in first[j] else 0.0 for w in range(len(e["b"]))]
            res[i] = (float(pg[j]), pa, pb)
    return res
