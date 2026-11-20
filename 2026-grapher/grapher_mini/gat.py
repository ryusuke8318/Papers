"""GraphER-GAT (GATv2 5 層 + 全結合 2 層) と、メッセージパッシングなしの MLP 版を、素の torch で実装して評価する。

学習は HotpotQA の「評価とは別の質問」(train 分割。うち 200 問を検証に使う)だけ。評価は 3 データセットの eval 分割。
入力特徴 = [GCS の点数, 質問の埋め込み, 段落の埋め込み]。辺は実体の共有の有無(重みなし。重みは注意が学ぶ)。
"""
from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pipeline as pl
import grapher_core as gc
import grapher_metrics as gm

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def prepare(name, tag, alpha, mode="ner"):
    S = pl.Split(name, tag)
    n = len(S.Q)
    cand = np.zeros((n, pl.N_CAND), dtype=np.int64)
    feat = np.zeros((n, pl.N_CAND), dtype=np.float32)
    adj = np.zeros((n, pl.N_CAND, pl.N_CAND), dtype=bool)
    lab = np.zeros((n, pl.N_CAND), dtype=np.float32)
    for qi in range(n):
        c, s, _, _ = S.candidates(qi)
        A = S.graph(c, mode)
        cand[qi] = c
        feat[qi] = gc.gcs(A, s, alpha)
        adj[qi] = (A > 0) | (A.T > 0)
        lab[qi] = [float(int(x) in set(S.Q[qi]["gold"])) for x in c]
    return S, cand, feat, adj, lab


class GATv2Layer(nn.Module):
    def __init__(self, d, heads=4, mp=True):
        super().__init__()
        self.h, self.dh, self.mp = heads, d // heads, mp
        self.wl = nn.Linear(d, d)
        self.wr = nn.Linear(d, d)
        self.att = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, adj):           # x: B,n,d  adj: B,n,n (bool, 自己ループ込み)
        B, n, d = x.shape
        if not self.mp:
            return self.norm(x + F.elu(self.wl(x)))
        l = self.wl(x).view(B, n, self.h, self.dh)
        r = self.wr(x).view(B, n, self.h, self.dh)
        # e_ij = a^T LeakyReLU(W_l h_i + W_r h_j)
        e = F.leaky_relu(l.unsqueeze(2) + r.unsqueeze(1), 0.2)      # B,n,n,h,dh
        e = (e * self.att).sum(-1)                                   # B,n,n,h
        e = e.masked_fill(~adj.unsqueeze(-1), float("-inf"))
        a = torch.softmax(e, dim=2)                                  # i から見た近傍 j
        out = torch.einsum("bijh,bjhd->bihd", a, r).reshape(B, n, d)
        return self.norm(x + F.elu(out))


class Ranker(nn.Module):
    def __init__(self, in_dim, d=64, layers=5, mp=True):
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.layers = nn.ModuleList([GATv2Layer(d, mp=mp) for _ in range(layers)])
        self.fc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x, adj):
        h = F.elu(self.inp(x))
        for L in self.layers:
            h = L(h, adj)
        return self.fc(h).squeeze(-1)


def batch_tensors(S, cand, feat, adj, lab, idx):
    q = torch.from_numpy(S.Q_emb[idx]).to(DEV)
    d = torch.from_numpy(S.P_emb[cand[idx]]).to(DEV)
    f = torch.from_numpy(feat[idx]).to(DEV).unsqueeze(-1)
    x = torch.cat([f * 5, q.unsqueeze(1).expand(-1, d.shape[1], -1), d], -1)
    A = torch.from_numpy(adj[idx]).to(DEV)
    A = A | torch.eye(A.shape[1], dtype=torch.bool, device=DEV).unsqueeze(0)
    return x, A, torch.from_numpy(lab[idx]).to(DEV)


@torch.no_grad()
def predict(model, S, cand, feat, adj, lab, bs=8):
    model.eval()
    out = []
    for i in range(0, len(cand), bs):
        idx = np.arange(i, min(i + bs, len(cand)))
        x, A, _ = batch_tensors(S, cand, feat, adj, lab, idx)
        out.append(model(x, A).float().cpu().numpy())
    return np.concatenate(out)


def evaluate(S, cand, scores):
    pr5, pr10 = [], []
    for qi in range(len(cand)):
        order = np.argsort(-scores[qi], kind="stable")
        r = [int(cand[qi][i]) for i in order]
        g = set(S.Q[qi]["gold"])
        pr5.append(gm.perfect_recall_at_k(r, g, 5)); pr10.append(gm.perfect_recall_at_k(r, g, 10))
    return np.array(pr5), np.array(pr10)


def train(mp, tr, va, epochs=12, lr=5e-4, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    S, cand, feat, adj, lab = tr
    model = Ranker(1 + 2 * S.P_emb.shape[1], mp=mp).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best, best_state, log = -1, None, []
    n = len(cand)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        tot = 0
        for i in range(0, n, 8):
            idx = perm[i:i + 8]
            x, A, y = batch_tensors(S, cand, feat, adj, lab, idx)
            loss = F.binary_cross_entropy_with_logits(model(x, A), y, pos_weight=torch.tensor(10.0, device=DEV))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sc = predict(model, *va)
        _, pr10 = evaluate(va[0], va[1], sc)
        log.append({"epoch": ep, "train_loss": tot / n, "val_pr10": float(pr10.mean())})
        print(("GAT" if mp else "MLP"), log[-1], flush=True)
        if pr10.mean() > best:
            best, best_state = pr10.mean(), {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, log


def view(P, a, b):
    """質問の範囲 [a, b) だけを切り出した (S, cand, feat, adj, lab)。S は Q・Q_emb・P_emb だけ持つ。"""
    S, cand, feat, adj, lab = P
    V = SimpleNamespace(Q=S.Q[a:b], Q_emb=S.Q_emb[a:b], P_emb=S.P_emb)
    return V, cand[a:b], feat[a:b], adj[a:b], lab[a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", default=str(pl.HERE / "results" / "gat.json"))
    a = ap.parse_args()
    t0 = time.time()
    P = prepare("hotpot", "train", a.alpha)
    va, tr = view(P, 0, 200), view(P, 200, len(P[1]))
    models = {}
    logs = {}
    for name, mp in (("gat", True), ("mlp", False)):
        models[name], logs[name] = train(mp, tr, va)
    res = {"alpha": a.alpha, "logs": logs, "eval": {}}
    for ds in ("hotpot", "2wiki", "musique"):
        E = prepare(ds, "eval", a.alpha)
        res["eval"][ds] = {}
        for name, m in models.items():
            sc = predict(m, *E)
            pr5, pr10 = evaluate(E[0], E[1], sc)
            res["eval"][ds][name] = {"pr5": pr5.tolist(), "pr10": pr10.tolist()}
            print(ds, name, pr5.mean(), pr10.mean(), flush=True)
    res["seconds"] = time.time() - t0
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
