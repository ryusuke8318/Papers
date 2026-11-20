"""GRPO の学習ループ(PyTorch + transformers + peft)。DAVID-GRPO の 3 部品はフラグで切り替える。

  --warm_steps N   few-shot warm-start(専門家 1 本 + on-policy の混合グループ)を N ステップ。0 なら cold start
  --lam L          grounded retrieval reward の重み λ(0 なら outcome のみ)
  --expansion 0/1  grounded expansion
1 回の rollout につき更新は 1 回なので、重要度比 ρ は 1 で、クリップは働かない(論文の付録 D.2 の ρ*=1 と同じ扱い)。
KL 罰則(β = 1e-3)は省いた。
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

import common
from common import RESULTS, WORK, Peak, cap_memory, gb, load_model, log_line
from david_core import (accept_expansion, group_advantages, mixed_group_advantages, plan_expansion,
                        render_expert, trajectory_reward)
from rollout import State, advance, prompt_ids
from warmstart import EXPERTS


def build_expert_states(tok, retr, train_by_id, lam):
    out = []
    for e in EXPERTS:
        q = train_by_id[e["id"]]
        gens, traj = render_expert(e["steps"], e["final"], e["answer"],
                                   lambda query: retr.search([query], k=3)[0])
        s = State(-1, prompt_ids(tok, q["question"]))
        from david_core import format_information
        k = 0
        for g in gens:
            ids = tok(g, add_special_tokens=False)["input_ids"]
            s.resp_ids += ids
            s.mask += [1] * len(ids)
            if k < len(e["steps"]):
                docs = retr.search([e["steps"][k][1]], k=3)[0]
                info = tok(format_information(docs), add_special_tokens=False)["input_ids"]
                s.resp_ids += info
                s.mask += [0] * len(info)
                k += 1
        s.traj, s.expert, s.done = traj, True, True
        s.reward = trajectory_reward(traj, q["answers"], q["gold_titles"], lam)
        out.append((q, s, [t for st in traj.steps for t in st.retrieved]))
    return out


def seq_logprob_sum(model, batch: list[State], device="cuda"):
    """各 state の、モデルが生成したトークンの対数確率(合計)と、そのトークン数。"""
    inner = model.base_model.model.model
    head = model.base_model.model.lm_head
    seqs = [s.prompt_ids + s.resp_ids for s in batch]
    L = max(map(len, seqs))
    ids = torch.zeros((len(seqs), L), dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    sel_b, sel_p, tgt = [], [], []
    for b, (s, seq) in enumerate(zip(batch, seqs)):
        ids[b, :len(seq)] = torch.tensor(seq)
        att[b, :len(seq)] = 1
        P = len(s.prompt_ids)
        for j, m in enumerate(s.mask):
            if m:
                sel_b.append(b); sel_p.append(P + j - 1); tgt.append(seq[P + j])
    with torch.autocast("cuda", dtype=torch.float16):
        h = inner(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state
        hs = h[torch.tensor(sel_b), torch.tensor(sel_p)]
        logits = head(hs).float()
    tgt_t = torch.tensor(tgt, device=device)
    lp = logits.gather(-1, tgt_t[:, None])[:, 0] - torch.logsumexp(logits, -1)
    sums, cnts = [], []
    sel_b_t = torch.tensor(sel_b, device=device)
    for b in range(len(batch)):
        m = sel_b_t == b
        sums.append(lp[m].sum())
        cnts.append(int(m.sum()))
    return sums, cnts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--warm_steps", type=int, default=0)
    ap.add_argument("--main_steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--group", type=int, default=5)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--expansion", type=int, default=0)
    ap.add_argument("--n_expand", type=int, default=5)
    ap.add_argument("--lr_warm", type=float, default=2e-4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_n", type=int, default=300)
    ap.add_argument("--no_eval", action="store_true")
    a = ap.parse_args()

    frac = cap_memory()
    run = WORK / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    logf = run / "log.jsonl"
    (run / "args.json").write_text(json.dumps(vars(a)))
    state_f = run / "state.json"
    start = json.loads(state_f.read_text())["step"] if state_f.exists() else 0
    model, tok = load_model(str(run / "adapter") if start > 0 else None, trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    from retriever import Retriever
    retr = Retriever()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    if start > 0 and (run / "opt.pt").exists():
        opt.load_state_dict(torch.load(run / "opt.pt"))
    peak = Peak()
    print(f"[{a.name}] memory cap fraction={frac:.3f}, resume step={start}, trainable={sum(p.numel() for p in params)}", flush=True)

    train = json.loads((WORK / "train.json").read_text())
    by_id = {q["id"]: q for q in train}
    expert_ids = {e["id"] for e in EXPERTS}
    pool = [q for q in train if q["id"] not in expert_ids]
    random.Random(a.seed).shuffle(pool)
    experts = build_expert_states(tok, retr, by_id, a.lam) if a.warm_steps > 0 else []
    total = a.warm_steps + a.main_steps

    for step in range(start, total):
        warm = step < a.warm_steps
        t0 = time.time()
        torch.manual_seed(a.seed * 100003 + step)
        if warm:
            items = [(q, s) for q, s, _ in experts]  # 4 問すべて(論文: batch 4、データ 4 件)
            G = 5
        else:
            ms = step - a.warm_steps
            items = [(q, None) for q in pool[ms * a.batch:(ms + 1) * a.batch]]
            G = a.group
        n_on = G - 1 if warm else G
        states, groups = [], []
        for q, ex in items:
            g = [State(0, prompt_ids(tok, q["question"])) for _ in range(n_on)]
            states += g
            groups.append(g)
        model.eval()
        peak.start()
        advance(model, tok, retr, states, temperature=a.temperature)
        t_roll = time.time() - t0
        for (q, _), g in zip(items, groups):
            for s in g:
                s.reward = trajectory_reward(s.traj, q["answers"], q["gold_titles"], a.lam)
        # grounded expansion
        n_plan = n_acc = n_extra = 0
        if a.expansion and not warm:
            jobs = []
            for gi, ((q, _), g) in enumerate(zip(items, groups)):
                plan = plan_expansion([s.reward for s in g], [s.traj for s in g], q["gold_titles"])
                if plan:
                    m, w, t = plan
                    jobs.append((gi, m, w, [g[m].prefix_after_search(t) for _ in range(a.n_expand)]))
            n_plan = len(jobs)
            cands = [c for j in jobs for c in j[3]]
            if cands:
                advance(model, tok, retr, cands, temperature=a.temperature)
                n_extra = len(cands)
            for gi, m, w, cs in jobs:
                q = items[gi][0]
                for c in cs:
                    c.reward = trajectory_reward(c.traj, q["answers"], q["gold_titles"], a.lam)
                j = accept_expansion(groups[gi][m].reward, [c.reward for c in cs])
                if j is not None:
                    groups[gi][w] = cs[j]
                    n_acc += 1
        gen_peak = peak.stop("generation")
        torch.cuda.empty_cache()
        t_gen = time.time() - t0
        # 利点
        pairs = []
        for (q, ex), g in zip(items, groups):
            if warm:
                ea, adv = mixed_group_advantages(ex.reward, [s.reward for s in g])
                pairs.append((ex, ea))
            else:
                adv = group_advantages([s.reward for s in g])
            pairs += list(zip(g, adv))
        n_total = len(pairs)
        todo = [(s, adv) for s, adv in pairs if abs(adv) > 1e-8]
        todo.sort(key=lambda x: len(x[0].resp_ids))
        # 更新
        model.train()
        for pg in opt.param_groups:
            pg["lr"] = a.lr_warm if warm else a.lr
        peak.start()
        t1 = time.time()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        n_tok = 0
        for i in range(0, len(todo), a.micro):
            mb = todo[i:i + a.micro]
            sums, cnts = seq_logprob_sum(model, [s for s, _ in mb])
            # ρ = exp(logp - logp_old) = 1 なので、f_clip の勾配は A * ∇logp に等しい
            loss = sum(-(adv * sm / max(c, 1)) for (_, adv), sm, c in zip(mb, sums, cnts)) / n_total
            scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
            n_tok += sum(cnts)
        if todo:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
        train_peak = peak.stop("train")
        torch.cuda.empty_cache()
        t_train = time.time() - t1
        gen_tok = sum(sum(s.mask) for s, _ in pairs)
        on = [s for g in groups for s in g]
        rec = {"step": step + 1, "phase": "warm" if warm else "main", "reward": sum(s.reward for s in on) / len(on),
               "em": 0.0, "n_traj": len(on), "n_trained": len(todo), "loss": loss_sum, "n_plan": n_plan, "n_accept": n_acc,
               "n_extra_rollouts": n_extra, "t_gen": round(t_gen, 2), "t_train": round(t_train, 2),
               "t_step": round(time.time() - t0, 2), "gen_peak_gb": gen_peak, "train_peak_gb": train_peak,
               "gen_tokens": gen_tok, "tok_per_s": round(gen_tok / max(t_gen, 1e-6), 1),
               "searches": sum(s.traj.n_searches for s in on) / len(on),
               "answered": sum(s.traj.answer is not None for s in on) / len(on)}
        # outcome と recall の内訳
        o, r = [], []
        for (q, _), g in zip(items, groups):
            for s in g:
                o.append(trajectory_reward(s.traj, q["answers"], q["gold_titles"], 0.0))
                r.append(trajectory_reward(s.traj, q["answers"], q["gold_titles"], 1.0))
        rec["em"] = sum(o) / len(o)
        rec["recall"] = sum(r) / len(r)
        log_line(logf, rec)
        print(json.dumps(rec), flush=True)
        if (step + 1) % 5 == 0 or step + 1 == total:
            model.save_pretrained(run / "adapter")
            torch.save(opt.state_dict(), run / "opt.pt")
            state_f.write_text(json.dumps({"step": step + 1}))
    (run / "peak_mem.json").write_text(json.dumps(peak.max))
    if not a.no_eval:
        from evaluate import run_eval
        model.eval()
        run_eval(model, tok, retr, a.name, a.eval_n, peak=peak)
        (run / "peak_mem.json").write_text(json.dumps(peak.max))
    (run / "DONE").write_text("ok")


if __name__ == "__main__":
    main()
