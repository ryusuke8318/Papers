"""学習しないベースラインの評価。

  # 凍結した Qwen2.5 の各層と DINOv2 の整合度(COCO)。層は検証セットで選ぶ
  python flickr_table1/baselines.py lm --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B

  # 未学習の BERT の整合度が、なぜ偶然の水準より高いのかの診断(Flickr30k)
  python flickr_table1/baselines.py diagnose

7B モデルは bf16 で約 15GB なので、VRAM が 16GB 未満の GPU には載らない。
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import load_features, save_json
from metrics import evaluate_alignment, mutual_knn


def lm_baseline(model_name: str, n_train: int = 20000):
    """凍結した LM(重みは更新しない)の各層のトークン平均を DINOv2 と比べる。層は検証セットで選ぶ。"""
    import torch
    from transformers import AutoModel, AutoTokenizer

    img, captions, splits = load_features("coco")
    tr, va, te = (np.where(splits == s)[0] for s in ("train", "val", "test"))
    tr = np.random.default_rng(0).choice(tr, size=n_train, replace=False)  # 白色化の推定用
    first = [c[0] for c in captions]

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda().eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9

    @torch.no_grad()
    def embed(idx, layers=None, bs=64):
        out = None
        for i in range(0, len(idx), bs):
            enc = tok([first[j] for j in idx[i : i + bs]], return_tensors="pt", padding=True, truncation=True,
                      max_length=48).to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states
            sel = range(len(hs)) if layers is None else layers
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = torch.stack([((hs[l].float() * m).sum(1) / m.sum(1)).cpu() for l in sel], 1)  # (b, L, d)
            out = pooled if out is None else torch.cat([out, pooled])
        return out.numpy().astype(np.float16)

    vt = embed(np.concatenate([va, te]))  # (n_val+n_test, L+1, d)
    v_all, t_all = vt[: len(va)], vt[len(va) :]
    val_knn = [mutual_knn(v_all[:, l].astype(np.float32), img[va]) for l in range(v_all.shape[1])]
    best = int(np.argmax(val_knn))
    last = v_all.shape[1] - 1
    tr_emb = embed(tr, layers=[best, last])
    res = {"model": model_name, "params_B": round(n_params, 2), "best_layer_by_val": best, "n_layers": last,
           "val_mutual_knn_by_layer": [round(x, 4) for x in val_knn]}
    for name, l, j in (("best_layer", best, 0), ("last_layer", last, 1)):
        res[name] = evaluate_alignment(
            tr_emb[:, j].astype(np.float32), img[tr], v_all[:, l].astype(np.float32), img[va],
            t_all[:, l].astype(np.float32), img[te],
        )
    print(json.dumps({k: v for k, v in res.items() if k != "val_mutual_knn_by_layer"}, indent=1))
    del model
    torch.cuda.empty_cache()
    return res


def diagnose_random():
    """未学習の BERT の整合度がなぜ偶然の水準より高いのかを調べる診断。"""
    import torch

    import jepa_text

    img, captions, splits = load_features("flickr")
    body, lens = jepa_text.tokenize_captions(captions)
    tr, va, te = (np.where(splits == s)[0] for s in ("train", "val", "test"))
    ids_all, attn_all = jepa_text.full_caption_ids(body[:, 0], lens[:, 0])

    def embed(model, mode):
        out = []
        with torch.no_grad():
            for i in range(0, len(ids_all), 1000):
                ids, attn = ids_all[i : i + 1000].cuda(), attn_all[i : i + 1000].cuda()
                if mode == "bow":  # Transformer を通さず、入力の単語埋め込みの平均だけ
                    h = model.bert.embeddings.word_embeddings(ids)
                    m = attn.unsqueeze(-1).float()
                    out.append(((h * m).sum(1) / m.sum(1)).cpu())
                    continue
                h = model.hidden_states(ids, attn)
                if mode == "cls":
                    out.append(h[:, 0].cpu())
                elif mode == "mean":
                    m = attn.unsqueeze(-1).float()
                    out.append(((h * m).sum(1) / m.sum(1)).cpu())
                elif mode == "mean_nospecial":  # [CLS] と [SEP] を除いて平均
                    m = attn.clone()
                    m[:, 0] = 0
                    m[torch.arange(len(m)), attn.sum(1) - 1] = 0
                    m = m.unsqueeze(-1).float()
                    out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
        return torch.cat(out).numpy()

    def ev(txt, eps=1e-2, shuffle=False):
        i_te = img[te]
        if shuffle:  # 対応をずらした対照実験: 偶然の水準に戻るはず
            i_te = i_te[np.random.default_rng(0).permutation(len(te))]
        return evaluate_alignment(txt[tr], img[tr], txt[va], img[va], txt[te], i_te, eps=eps)

    def fmt(r):
        return {k: round(v, 4) for k, v in r.items() if "chance" not in k}

    rows = []
    for seed in range(3):
        torch.manual_seed(seed)
        model = jepa_text.TextEncoder(causal=False).cuda().eval()
        for mode in ("mean", "cls", "mean_nospecial", "bow"):
            rows.append({"seed": seed, "pool": mode, "eps": 1e-2, **fmt(ev(embed(model, mode)))})
        txt = embed(model, "mean")
        for eps in (1e-3, 1e-1, 1.0):
            rows.append({"seed": seed, "pool": "mean", "eps": eps, **fmt(ev(txt, eps=eps))})
        rows.append({"seed": seed, "pool": "mean", "eps": 1e-2, "shuffled": True, **fmt(ev(txt, shuffle=True))})
    for r in rows:
        print(r)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lm")
    p.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"])
    p.add_argument("--n-train", type=int, default=20000)
    sub.add_parser("diagnose")
    a = ap.parse_args()
    if a.cmd == "lm":
        save_json("coco_lm_baselines_local.json", {m: lm_baseline(m, a.n_train) for m in a.models})
    else:
        save_json("diagnose_random_local.json", diagnose_random())
