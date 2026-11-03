"""Flickr30k での text JEPA(good view / bad view)と Next Token Prediction の比較 (Table 1 の縮小版)。

画像側は学習せず、凍結した DINOv2 の CLS 特徴量を使う。text エンコーダは BERT を最初から学習する。
JEPA の目的関数は LeJEPA: 「複数ビューの平均への近さ」+ SIGReg (λ=0.02)。

論文に記載のない設定は、ここで選んだ値(コメントに「仮定」と書いた)。
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizerFast

from metrics import evaluate_alignment, mutual_knn

PAD, CLS, SEP, MASK = 0, 101, 102, 103  # bert-base-uncased の特殊トークン
N_CAPS = 5  # Flickr30k は 1 画像 5 キャプション
MAX_BODY = 32  # 仮定: 特殊トークンを除いて 32 トークンで切る


# ---------------------------------------------------------------- データ
def tokenize_captions(captions: list[list[str]]):
    """captions[i] は画像 i の 5 キャプション。body: (N,5,MAX_BODY), lens: (N,5)。"""
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    flat = []
    for caps in captions:
        assert len(caps) >= N_CAPS, "1 画像あたり 5 キャプション以上が必要"
        flat.extend(caps[:N_CAPS])
    enc = tok(flat, add_special_tokens=False, truncation=True, max_length=MAX_BODY)["input_ids"]
    n = len(captions)
    body = np.zeros((n * N_CAPS, MAX_BODY), dtype=np.int64)
    lens = np.zeros(n * N_CAPS, dtype=np.int64)
    for i, ids in enumerate(enc):
        ids = ids[:MAX_BODY] or [100]  # 空なら [UNK]
        body[i, : len(ids)] = ids
        lens[i] = len(ids)
    return torch.from_numpy(body.reshape(n, N_CAPS, MAX_BODY)), torch.from_numpy(lens.reshape(n, N_CAPS))


def wrap_special(tok: torch.Tensor, m: torch.Tensor):
    """tok: (B,L) 先頭 m 個が有効。 [CLS] tok [SEP] の ids と attention mask を作る。"""
    B, L = tok.shape
    ids = torch.zeros(B, L + 2, dtype=torch.long, device=tok.device)
    ids[:, 0] = CLS
    ids[:, 1 : L + 1] = tok
    ids[torch.arange(B, device=tok.device), m + 1] = SEP
    attn = (torch.arange(L + 2, device=tok.device)[None] <= (m + 1)[:, None]).long()
    return ids, attn


def make_views(body: torch.Tensor, lens: torch.Tensor, mask_ratio: float):
    """body: (B,L), lens: (B,)。ランダムな連続区間に切り出し(trim)、mask_ratio のトークンを [MASK] にする。

    仮定: 区間の長さは [0.5·n, n] から一様に選ぶ。論文では trim が最も効くと報告されている。
    """
    B, L = body.shape
    dev = body.device
    ar = torch.arange(L, device=dev)[None]
    n = lens.clamp(min=1)
    m = torch.ceil(n * (0.5 + 0.5 * torch.rand(B, device=dev))).long().clamp(min=1)
    m = torch.minimum(m, n)
    start = torch.floor(torch.rand(B, device=dev) * (n - m + 1)).long()
    tok = body.gather(1, (start[:, None] + ar).clamp(max=L - 1))
    valid = ar < m[:, None]
    tok = torch.where(valid, tok, torch.zeros_like(tok))
    masked = (torch.rand(B, L, device=dev) < mask_ratio) & valid
    tok = torch.where(masked, torch.full_like(tok, MASK), tok)
    return wrap_special(tok, m)


def full_caption_ids(body: torch.Tensor, lens: torch.Tensor):
    """評価用: trim も mask もしない完全なキャプション。"""
    return wrap_special(body, lens.clamp(min=1))


def ntp_batch(body_np: np.ndarray, lens_np: np.ndarray, idx: np.ndarray, rng, pairs: bool):
    """Next Token Prediction 用の系列 [CLS] cap [SEP] (pairs なら [CLS] capA [SEP] capB [SEP])。"""
    seqs = []
    for i in idx:
        a = rng.integers(N_CAPS)
        s = [CLS, *body_np[i, a, : lens_np[i, a]].tolist(), SEP]
        if pairs:
            b = (a + rng.integers(1, N_CAPS)) % N_CAPS  # a と異なるキャプション
            s += [*body_np[i, b, : lens_np[i, b]].tolist(), SEP]
        seqs.append(s)
    L = max(map(len, seqs))
    ids = np.zeros((len(seqs), L), dtype=np.int64)
    attn = np.zeros_like(ids)
    for j, s in enumerate(seqs):
        ids[j, : len(s)] = s
        attn[j, : len(s)] = 1
    return torch.from_numpy(ids), torch.from_numpy(attn)


# ---------------------------------------------------------------- SIGReg
def sigreg(z: torch.Tensor, num_slices: int = 256, n_knots: int = 17, t_max: float = 5.0):
    """LeJEPA の SIGReg: ランダムな 1 次元射影で Epps–Pulley 検定統計量を測り、N(0, I) からのずれを罰する。"""
    N, D = z.shape
    A = torch.randn(D, num_slices, device=z.device)
    A = A / A.norm(dim=0, keepdim=True)
    t = torch.linspace(-t_max, t_max, n_knots, device=z.device)
    target = torch.exp(-0.5 * t**2)  # N(0,1) の特性関数
    dt = 2 * t_max / (n_knots - 1)
    w = torch.full((n_knots,), 2 * dt, device=z.device)
    w[[0, -1]] = dt
    w = w * target  # 台形則 × ガウス窓
    xt = (z @ A).unsqueeze(-1) * t  # (N, S, T)
    err = (xt.cos().mean(0) - target).square() + xt.sin().mean(0).square()  # (S, T)
    return ((err @ w) * N).mean()


# ---------------------------------------------------------------- モデル
class TextEncoder(nn.Module):
    """BERT-S 相当(幅 384, 12 層, 6 ヘッド, 約 33M パラメータ)を最初から学習する。"""

    def __init__(self, causal: bool, hidden=384, layers=12, heads=6, proj_dim=128):
        super().__init__()
        cfg = BertConfig(
            vocab_size=30522, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
            intermediate_size=4 * hidden, max_position_embeddings=96, is_decoder=causal,
            attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0,
        )
        self.bert = BertModel(cfg, add_pooling_layer=False)
        self.proj = nn.Sequential(  # 仮定: 3 層 MLP + BatchNorm
            nn.Linear(hidden, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, proj_dim),
        )

    def hidden_states(self, ids, attn):
        return self.bert(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state

    def encode(self, ids, attn):
        """バックボーンの特徴量: トークンの平均(仮定: mean pooling)。"""
        h = self.hidden_states(ids, attn)
        m = attn.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1)

    def forward(self, ids, attn):
        return self.proj(self.encode(ids, attn))

    def lm_logits(self, h):
        return h @ self.bert.embeddings.word_embeddings.weight.T  # 入出力の埋め込みを共有(tied softmax)


# ---------------------------------------------------------------- 学習ステップ
def jepa_step(model, body, lens, good: bool, n_views: int, lam: float, mask_ratio: float):
    """body: (B,5,L), lens: (B,5)。good=True: 各ビューを同じ画像の別々のキャプションから作る。"""
    B, _, L = body.shape
    dev = body.device
    if good:
        perm = torch.rand(B, N_CAPS, device=dev).argsort(1)  # 5 つ全てを使い、余りは重複を許す
        cap = perm[:, torch.arange(n_views, device=dev) % N_CAPS]  # (B,V)
    else:
        cap = torch.randint(0, N_CAPS, (B, 1), device=dev).expand(B, n_views)  # 1 つのキャプションから 8 ビュー
    rows = torch.arange(B, device=dev)[:, None]
    ids, attn = make_views(body[rows, cap].reshape(B * n_views, L), lens[rows, cap].reshape(-1), mask_ratio)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = model(ids, attn)
    z = z.float().view(B, n_views, -1).transpose(0, 1)  # (V,B,D)
    inv = (z - z.mean(0, keepdim=True)).square().mean()
    sig = torch.stack([sigreg(z[v]) for v in range(n_views)]).mean()
    loss = (1 - lam) * inv + lam * sig
    return loss, {"inv": inv.item(), "sigreg": sig.item()}


def ntp_step(model, ids, attn):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.hidden_states(ids, attn)
    logits = model.lm_logits(h[:, :-1].float())
    tgt = ids[:, 1:]
    m = attn[:, 1:].bool()
    loss = F.cross_entropy(logits[m], tgt[m])
    return loss, {"nll": loss.item()}


@torch.no_grad()
def embed_first_captions(model, body, lens, bs=1000):
    """評価用の埋め込み: 各画像の 1 番目のキャプション(完全な文)のバックボーン特徴量。"""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(body), bs):
        ids, attn = full_caption_ids(body[i : i + bs, 0].to(dev), lens[i : i + bs, 0].to(dev))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model.encode(ids, attn).float().cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- 実行
def run(variant: str, seed: int, samples: int, feats: dict, captions: list, splits: np.ndarray,
        batch: int = 256, lr: float = 5e-4, wd: float = 0.05, lam: float = 0.02, n_views: int = 8,
        mask_ratio: float = 0.15, clip: float = 1.0, n_evals: int = 10, log=print, grad_checkpoint: bool = False):
    """variant: good | bad | ntp | ntp_pairs | random(未学習のベースライン)。

    samples: 学習で見る画像グループの総数(= バッチ数 × batch)。Flickr の 200 epoch は約 580 万。
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda"
    body, lens = tokenize_captions(captions)
    tr, va, te = (np.where(splits == s)[0] for s in ("train", "val", "test"))
    img = feats["img"]
    model = TextEncoder(causal=variant.startswith("ntp")).to(dev)
    if grad_checkpoint:  # 計算を増やして活性化のメモリを減らす(VRAM が 24GB 未満の GPU 向け。結果は変わらない)
        model.bert.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    total = max(1, samples // batch)
    log(f"[{variant}] seed={seed} samples={total * batch:,} ({total} step) 学習用 {len(tr)} 画像, "
        f"val {len(va)}, test {len(te)}, パラメータ {n_params:.1f}M")

    def evaluate(final: bool):
        if not final:  # 途中経過: 検証セットの mutual k-NN だけ(何も当てはめない)
            txt_va = embed_first_captions(model, body[va], lens[va])
            model.train()
            return {"val_mutual_knn": mutual_knn(txt_va, img[va])}
        txt = embed_first_captions(model, body, lens)
        return evaluate_alignment(txt[tr], img[tr], txt[va], img[va], txt[te], img[te])

    history = []
    step = 0
    if variant != "random":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        warm = max(1, int(0.05 * total))  # 仮定: 5% のウォームアップ + コサイン減衰
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
        )
        body_d, lens_d = body.to(dev), lens.to(dev)
        body_np, lens_np = body.numpy(), lens.numpy()
        eval_steps = {max(1, round(total * (i + 1) / n_evals)) for i in range(n_evals)}
        order, ptr, t0 = rng.permutation(tr), 0, time.time()
        model.train()
        while step < total:
            if ptr + batch > len(order):  # 1 周したら並べ直す
                order, ptr = rng.permutation(tr), 0
            idx = order[ptr : ptr + batch]
            ptr += batch
            if variant in ("good", "bad"):
                loss, stats = jepa_step(model, body_d[idx], lens_d[idx], variant == "good", n_views, lam, mask_ratio)
            else:
                ids, attn = ntp_batch(body_np, lens_np, idx, rng, pairs=variant == "ntp_pairs")
                loss, stats = ntp_step(model, ids.to(dev), attn.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 論文: クリッピングなしでは崩壊する run がある
            opt.step()
            sched.step()
            step += 1
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"loss が発散しました (step {step})")
            if step in eval_steps:
                rec = {"step": step, "samples": step * batch, "loss": loss.item(), **stats,
                       **evaluate(final=False), "sec": round(time.time() - t0)}
                history.append(rec)
                log(f"  {rec}")
    result = {"variant": variant, "seed": seed, "samples": step * batch, "params_M": n_params,
              "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)), **evaluate(final=True)}
    log(f"[{variant}] seed={seed} samples={step * batch:,} 最終: "
        f"{ {k: round(v, 4) for k, v in result.items() if isinstance(v, float)} }")
    result["history"] = history
    return result
