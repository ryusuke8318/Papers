"""E5-base-v2 の密検索(コーパスの埋め込みは fp16 の行列で持ち、GPU の行列積で検索する)。"""
from __future__ import annotations

import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common import WORK

RETR_ID = "intfloat/e5-base-v2"


class Retriever:
    def __init__(self, corpus_path=None, emb_path=None, device="cuda"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(RETR_ID)
        self.enc = AutoModel.from_pretrained(RETR_ID, torch_dtype=torch.float16).to(device).eval()
        self.corpus = json.loads((corpus_path or WORK / "corpus.json").read_text())
        emb_path = emb_path or WORK / "corpus_e5.npy"
        if emb_path.exists():
            emb = np.load(emb_path)
        else:
            emb = self._encode([f"passage: {p['title']}. {p['text']}" for p in self.corpus], bs=256, max_len=192)
            np.save(emb_path, emb)
        self.emb = torch.from_numpy(emb).to(device)

    @torch.no_grad()
    def _encode(self, texts, bs=128, max_len=64):
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
            h = self.enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(v.float(), dim=-1).half().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def search(self, queries: list[str], k: int = 3) -> list[list[tuple[str, str]]]:
        """各クエリの上位 k 件 [(title, text)]。同じ title は 1 件にまとめる。"""
        q = torch.from_numpy(self._encode([f"query: {x}" for x in queries])).to(self.device)
        top = (q @ self.emb.T).topk(k + 3, dim=-1).indices.cpu().tolist()
        res = []
        for row in top:
            seen, docs = set(), []
            for i in row:
                p = self.corpus[i]
                if p["title"] in seen:
                    continue
                seen.add(p["title"])
                docs.append((p["title"], p["text"]))
                if len(docs) == k:
                    break
            res.append(docs)
        return res
