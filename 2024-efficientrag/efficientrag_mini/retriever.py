"""小さな公開の埋め込みモデル(bge-small-en-v1.5)による密な検索。"""
from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

from config import EMBED, QUERY_PREFIX


class Index:
    def __init__(self, ids: list[str], texts: list[str], device: str = "cuda", name: str = EMBED, bs: int = 256):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device=device, dtype=torch.float16 if device == "cuda" else torch.float32).eval()
        self.ids = ids
        self.emb = self._encode(texts, bs, max_len=512)

    @torch.no_grad()
    def _encode(self, texts: list[str], bs: int, max_len: int) -> torch.Tensor:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for s in range(0, len(texts), bs):
            idx = order[s:s + bs]
            b = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
            e = self.model(**b).last_hidden_state[:, 0]
            e = torch.nn.functional.normalize(e.float(), dim=-1).to(self.model.dtype)
            for j, i in enumerate(idx):
                out[i] = e[j]
        return torch.stack(out)

    def search(self, queries: list[str], k: int) -> list[list[str]]:
        q = self._encode([QUERY_PREFIX + x for x in queries], 256, max_len=128)
        top = (q @ self.emb.T).topk(min(k, len(self.ids)), dim=-1).indices.cpu().tolist()
        return [[self.ids[j] for j in row] for row in top]
