"""判定モデル(論文の付録 G): 事前学習済みの Qwen3-Base に、言語モデルの head の代わりに採点用の head を付けたもの。

- 共有文脈 s = 「問い」と「番号つきの候補の説明」。そのあとに、候補ごとの質問を 1 つずつ並べる(各質問に、選択肢 "no" と "yes" を付ける)。
- 候補ごとの質問は、文脈の直後の位置から始まり(位置 id が同じ)、互いには注意しない。文脈は質問を見ない(因果的)。
  このため、1 回の順伝播で全候補を並列に採点でき、各候補の確率は、共有文脈だけで決まる(論文の式 2)。
- head: 質問の末尾トークンと、各選択肢の末尾トークンの隠れ状態を、共通の空間に射影し、コサイン類似度に学習する逆温度(上限 100)を掛け、
  質問の選択肢の上で softmax を取る。P(yes) は "yes" の確率。学習は選択肢の上の交差エントロピー。
- 複数選択(式 3)は、1 つの質問に、候補の番号 "[1]" "[2]" … を選択肢として付け、全選択肢の上で softmax を取る。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

BIN_Q = "Is the information needed to answer the user's question in item [{i}]?"
CHOICE_Q = "Which item contains the information needed to answer the user's question?"
HEADERS = {1: "Topics", 2: "Pages", 3: "Paragraphs"}
MAX_LOGIT_SCALE = 100.0


@dataclass
class Layout:
    input_ids: torch.Tensor          # (L,)
    position_ids: torch.Tensor       # (L,)
    allowed: torch.Tensor            # (L, L) bool。True なら、行のトークンが列のトークンを見られる
    q_ends: torch.Tensor             # (n,) 各質問の末尾トークンの位置
    opt_ends: torch.Tensor           # (n, n_opt) 各選択肢の末尾トークンの位置
    ctx_len: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return int(self.input_ids.shape[0])


def _enc(tok, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=False)["input_ids"])


def _finish(ids, pos, block, seg, q_ends, opt_ends, ctx_len, option_link) -> Layout:
    idx = torch.arange(len(ids))
    bi, si = torch.tensor(block), torch.tensor(seg)
    causal = idx[None, :] <= idx[:, None]
    same_block = bi[None, :] == bi[:, None]
    in_ctx = bi[None, :] == -1
    seg_ok = (si[None, :] == 0) | (si[None, :] == si[:, None])
    if option_link:                      # 同じ質問の選択肢どうしは、互いに見える(論文の general・MD2D-only の checkpoint)
        seg_ok = seg_ok | ((si[None, :] > 0) & (si[:, None] > 0))
    allowed = causal & (in_ctx | (same_block & seg_ok))
    return Layout(torch.tensor(ids), torch.tensor(pos), allowed, torch.tensor(q_ends), torch.tensor(opt_ends), ctx_len)


def build_binary_layout(tok, header: str, query: str, texts: list[str], option_link: bool = False) -> Layout:
    """候補 n 件を、共有文脈に入れ、候補ごとの二値質問(選択肢 no / yes)を並べる。"""
    ctx = f"{header}\nUser question: {query}\n\n" + "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts)) + "\n\n"
    ids = _enc(tok, ctx)
    C = len(ids)
    pos, block, seg = list(range(C)), [-1] * C, [0] * C
    no_ids, yes_ids = _enc(tok, " no"), _enc(tok, " yes")
    q_ends, opt_ends = [], []
    for i in range(len(texts)):
        q = _enc(tok, BIN_Q.format(i=i + 1))
        start = len(ids)
        ids += q
        pos += list(range(C, C + len(q)))
        block += [i] * len(q)
        seg += [0] * len(q)
        q_ends.append(len(ids) - 1)
        ends = []
        for s, o in enumerate((no_ids, yes_ids), start=1):
            ids += o
            pos += list(range(C + len(q), C + len(q) + len(o)))   # 選択肢の位置は、質問の直後から始まる(選択肢どうしは対称)
            block += [i] * len(o)
            seg += [s] * len(o)
            ends.append(len(ids) - 1)
        opt_ends.append(ends)
        assert start >= C
    return _finish(ids, pos, block, seg, q_ends, opt_ends, C, option_link)


def build_choice_layout(tok, header: str, query: str, texts: list[str]) -> Layout:
    """候補 m 件を共有文脈に入れ、1 つの質問に、番号を選択肢として付ける。"""
    ctx = f"{header}\nUser question: {query}\n\n" + "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts)) + "\n\n"
    ids = _enc(tok, ctx)
    C = len(ids)
    pos, block, seg = list(range(C)), [-1] * C, [0] * C
    q = _enc(tok, CHOICE_Q)
    ids += q
    pos += list(range(C, C + len(q)))
    block += [0] * len(q)
    seg += [0] * len(q)
    q_ends = [len(ids) - 1]
    ends = []
    for j in range(len(texts)):
        o = _enc(tok, f" [{j + 1}]")
        ids += o
        pos += list(range(C + len(q), C + len(q) + len(o)))
        block += [0] * len(o)
        seg += [j + 1] * len(o)
        ends.append(len(ids) - 1)
    return _finish(ids, pos, block, seg, q_ends, [ends], C, False)


class DecisionModel(nn.Module):
    def __init__(self, base: nn.Module, hidden: int, proj_dim: int = 256, init_scale: float = 20.0):
        super().__init__()
        self.base = base
        self.proj = nn.Linear(hidden, proj_dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=MAX_LOGIT_SCALE)

    def hidden_states(self, lay: Layout, device, dtype) -> torch.Tensor:
        mask = torch.zeros(lay.allowed.shape, dtype=dtype)
        mask.masked_fill_(~lay.allowed, torch.finfo(dtype).min)
        out = self.base(input_ids=lay.input_ids[None].to(device), position_ids=lay.position_ids[None].to(device),
                        attention_mask=mask[None, None].to(device))
        return out.last_hidden_state[0]

    def forward(self, lay: Layout, device=None, dtype=None) -> torch.Tensor:
        """各質問の、選択肢の上のロジット (n, n_opt)。"""
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.base.parameters()).dtype
        h = self.hidden_states(lay, device, dtype)
        zq = F.normalize(self.proj(h[lay.q_ends.to(device)].float()), dim=-1)                    # (n, d)
        zo = F.normalize(self.proj(h[lay.opt_ends.to(device)].float()), dim=-1)                  # (n, n_opt, d)
        return self.scale.float() * (zq[:, None, :] * zo).sum(-1)


def load_model(name: str, device: str = "cuda", dtype=torch.bfloat16, proj_dim: int = 256, lora: dict | None = None, gradient_checkpointing: bool = False):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    base = AutoModel.from_pretrained(name, dtype=dtype)
    if gradient_checkpointing:
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
    if lora:
        from peft import LoraConfig, get_peft_model

        base = get_peft_model(base, LoraConfig(r=lora.get("r", 64), lora_alpha=lora.get("alpha", 128), lora_dropout=0.0,
                                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model = DecisionModel(base, base.config.hidden_size, proj_dim)
    return tok, model.to(device)


def save_judge(model: DecisionModel, tok, out: str, meta: dict) -> None:
    """LoRA のときは adapter だけ、全パラメータの学習のときは base の重みを、head と一緒に保存する。"""
    import json
    from pathlib import Path

    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    model.base.save_pretrained(str(d / "base"))
    torch.save({"proj": model.proj.state_dict(), "log_scale": model.log_scale.detach().cpu()}, d / "head.pt")
    tok.save_pretrained(str(d / "base"))
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))


def load_judge_model(path: str, device: str = "cuda", dtype=torch.bfloat16):
    import json
    from pathlib import Path

    from transformers import AutoModel, AutoTokenizer

    d = Path(path)
    meta = json.loads((d / "meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(d / "base"))
    if meta.get("lora"):
        from peft import PeftModel

        base = AutoModel.from_pretrained(meta["model"], dtype=dtype)
        base = PeftModel.from_pretrained(base, str(d / "base")).merge_and_unload()
    else:
        base = AutoModel.from_pretrained(str(d / "base"), dtype=dtype)
    model = DecisionModel(base, base.config.hidden_size, meta.get("proj_dim", 256))
    head = torch.load(d / "head.pt", map_location="cpu")
    model.proj.load_state_dict(head["proj"])
    with torch.no_grad():
        model.log_scale.copy_(head["log_scale"])
    return tok, model.to(device)


class DecisionJudge:
    """decisionrag_core.Judge の実装。候補(Node)の text を、共有文脈に入れて採点する。"""

    def __init__(self, model: DecisionModel, tok, device="cuda", dtype=torch.bfloat16, max_tokens: int = 24000, option_link: bool = False):
        self.model, self.tok, self.device, self.dtype = model.eval(), tok, device, dtype
        self.max_tokens, self.option_link = max_tokens, option_link
        self.stats = {"requests": 0, "tokens": 0, "judgments": 0}

    def _header(self, cands) -> str:
        return HEADERS.get(cands[0].level, "Passages") if cands else "Passages"

    @torch.no_grad()
    def binary(self, query: str, candidates) -> list[float]:
        cands = list(candidates)
        if not cands:
            return []
        lay = build_binary_layout(self.tok, self._header(cands), query, [c.text for c in cands], self.option_link)
        if lay.length > self.max_tokens and len(cands) > 1:        # 入力が長すぎるときだけ、半分に分ける(請求数の指定は core の request_size)
            h = len(cands) // 2
            return self.binary(query, cands[:h]) + self.binary(query, cands[h:])
        self.stats["requests"] += 1
        self.stats["tokens"] += lay.length
        self.stats["judgments"] += len(cands)
        logits = self.model(lay, self.device, self.dtype)
        return F.softmax(logits, dim=-1)[:, 1].tolist()

    @torch.no_grad()
    def choice(self, query: str, candidates) -> list[float]:
        cands = list(candidates)
        lay = build_choice_layout(self.tok, self._header(cands), query, [c.text for c in cands])
        self.stats["requests"] += 1
        self.stats["tokens"] += lay.length
        self.stats["judgments"] += len(cands)
        logits = self.model(lay, self.device, self.dtype)
        return F.softmax(logits[0], dim=-1).tolist()
