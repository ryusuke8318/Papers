"""LoRA による Stage-II 学習と、評価(ranking・生成)。torch / transformers / peft が必要。

- 学習: (x, c, y) 形式の行を 1 epoch。損失は答えのトークンだけ。LoRA(全線形層)、bf16、勾配チェックポイント
- ranking: 「For the question ..., access whether ...」への True / False の logit 差を関連度スコアにする(論文 §4.3)
- 生成: top-k 件を Table 1 の形式で渡し、貪欲デコード
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankrag_core import (contains_answer, exact_match, f1_score, mrr, qa_prompt, rank_prompt,  # noqa: E402
                          recall_at_k, safe_mean, seeded_shuffle)

MAX_LEN = 1536
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------- モデルの読み込み
def load(model_name: str, adapter: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return tok, model.eval()


def chat_text(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------- 学習
def build_blend(data_dir: Path, cfg: dict) -> list[dict]:
    rows: list[dict] = []
    if cfg["rank_only"]:
        rows = read_jsonl(data_dir / cfg.get("rank_file", "rank.jsonl"))[: cfg["rank_rows"]]
    else:
        if cfg["use_qa"]:
            rows += read_jsonl(data_dir / "qa.jsonl")[: cfg["n_qa"]]
        if cfg["use_rqa"]:
            rows += read_jsonl(data_dir / "rqa.jsonl")[: cfg["n_rqa"]]
        if cfg["use_rar"]:
            rows += read_jsonl(data_dir / "rar.jsonl")[: cfg["n_rar"]]
        if cfg["rank_rows"]:
            rows += read_jsonl(data_dir / cfg.get("rank_file", "rank.jsonl"))[: cfg["rank_rows"]]
    return seeded_shuffle(rows, cfg["seed"])


def train(cfg: dict, data_dir: Path, adapter_dir: Path, log=print) -> dict:
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(cfg["seed"])
    tok, base = load(cfg["model"])
    model = get_peft_model(base, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules=TARGETS,
                                            task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA の重みは fp32 で持つ(bf16 のままだと更新が丸められる)
    inner = model.get_base_model()
    inner.gradient_checkpointing_enable()
    inner.enable_input_require_grads()
    inner.config.use_cache = False
    model.train()

    rows = build_blend(data_dir, cfg)
    end = tok("<|im_end|>", add_special_tokens=False)["input_ids"]
    ex = []
    for r in rows:
        p = tok(chat_text(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
        a = tok(r["answer"], add_special_tokens=False)["input_ids"] + end
        if len(p) + len(a) <= cfg.get("max_len", MAX_LEN):
            ex.append((p, a))
    n_task = {t: sum(1 for r in rows if r["task"] == t) for t in ("qa", "rqa", "rar", "rank")}
    bsz = cfg["batch"]
    steps = math.ceil(len(ex) / bsz)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.98), weight_decay=0.0)  # 論文: Adam, beta2=0.98
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    log(f"[train] {cfg['name']}: 行 {len(ex)} ({n_task})  step {steps}  学習可能 {sum(p.numel() for p in params) / 1e6:.1f}M")

    budget = 8192  # 1 micro-batch の(パディング込み)トークン数の上限
    t0, losses = time.time(), []
    for s in range(steps):
        batch = sorted(ex[s * bsz : (s + 1) * bsz], key=lambda e: len(e[0]) + len(e[1]))
        n_ans = sum(len(a) for _, a in batch)
        micro, cur = [], []
        for e in batch:
            if cur and (len(cur) + 1) * (len(e[0]) + len(e[1])) > budget:
                micro.append(cur)
                cur = []
            cur.append(e)
        micro.append(cur)
        tot = 0.0
        for mb in micro:
            L = max(len(p) + len(a) for p, a in mb)
            ids = torch.full((len(mb), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(mb), L), dtype=torch.long)
            lab = torch.full((len(mb), L), -100, dtype=torch.long)
            for i, (p, a) in enumerate(mb):
                n = len(p) + len(a)
                ids[i, :n] = torch.tensor(p + a)
                att[i, :n] = 1
                lab[i, len(p) : n] = torch.tensor(a)
            ids, att, lab = ids.cuda(), att.cuda(), lab.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
                sel = lab[:, 1:] != -100
                logits = inner.lm_head(hs[:, :-1][sel])  # 答えの位置だけ語彙に射影(メモリ節約)
            loss = F.cross_entropy(logits.float(), lab[:, 1:][sel], reduction="sum") / n_ans
            loss.backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if not math.isfinite(tot):
            raise RuntimeError(f"loss が発散しました (step {s})")
        losses.append(tot)
        if (s + 1) % max(1, steps // 10) == 0 or s == 0:
            log(f"  step {s + 1}/{steps} loss {np.mean(losses[-20:]):.4f} ({time.time() - t0:.0f}s)")
    model.save_pretrained(str(adapter_dir))
    del model, base, inner
    torch.cuda.empty_cache()
    return {"rows": len(ex), "tasks": n_task, "steps": steps, "sec": round(time.time() - t0),
            "loss_first": float(np.mean(losses[:10])), "loss_last": float(np.mean(losses[-10:]))}


# ---------------------------------------------------------------- 推論
def _batches(lengths: list[int], max_tokens: int, max_bs: int):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    cur = []
    for i in order:
        if cur and ((len(cur) + 1) * lengths[i] > max_tokens or len(cur) >= max_bs):
            yield cur
            cur = []
        cur.append(i)
    if cur:
        yield cur


@torch.no_grad()
def rank_scores(tok, model, prompts: list[str], max_tokens: int = 24000) -> np.ndarray:
    """True と False の logit 差。値が大きいほど関連が高い。"""
    t_id = tok("True", add_special_tokens=False)["input_ids"][0]
    f_id = tok("False", add_special_tokens=False)["input_ids"][0]
    texts = [chat_text(tok, p) for p in prompts]
    ids_all = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    out = np.zeros(len(prompts), dtype=np.float32)
    for b in _batches([len(x) for x in ids_all], max_tokens, 256):
        L = max(len(ids_all[i]) for i in b)
        ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for r, i in enumerate(b):
            ids[r, L - len(ids_all[i]) :] = torch.tensor(ids_all[i])  # 左詰めのパディング
            att[r, L - len(ids_all[i]) :] = 1
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        hs = model.model(input_ids=ids.cuda(), attention_mask=att.cuda(), position_ids=pos.cuda()).last_hidden_state[:, -1]
        lg = model.lm_head(hs).float()
        out[b] = (lg[:, t_id] - lg[:, f_id]).cpu().numpy()
    return out


@torch.no_grad()
def generate(tok, model, prompts: list[str], max_new: int = 24, max_tokens: int = 14000) -> list[str]:
    texts = [chat_text(tok, p) for p in prompts]
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    out = [""] * len(prompts)
    for b in _batches(lens, max_tokens, 32):
        enc = tok([texts[i] for i in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, i in enumerate(b):
            out[i] = tok.decode(gen[r, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip().split("\n")[0].strip()
    return out


def pool_auroc(scores: np.ndarray, pool: list[int], gold: set) -> float | None:
    """pool 内で、gold passage のスコアが非 gold より高い確率(同点は 0.5)。gold が pool に無い、または全部が gold なら None。"""
    pos = [scores[i] for i, p in enumerate(pool) if p in gold]
    neg = [scores[i] for i, p in enumerate(pool) if p not in gold]
    if not pos or not neg:
        return None
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def _gen_metrics(preds: list[str], qs: list[dict]) -> dict:
    return {"EM": safe_mean([exact_match(p, q["answers"]) for p, q in zip(preds, qs)]),
            "F1": safe_mean([f1_score(p, q["answers"]) for p, q in zip(preds, qs)]),
            "Acc": safe_mean([contains_answer(p, q["answers"]) for p, q in zip(preds, qs)])}


def evaluate(tok, model, data_dir: Path, n_q: int, do_gen: bool = True, ks=(1, 3, 5), n_sweep=(10, 20, 30),
             log=print) -> dict:
    res: dict = {}
    for name in ("squad", "hotpot"):
        D = json.loads((data_dir / f"eval_{name}.json").read_text())
        passages, qs = D["passages"], D["questions"][:n_q]
        r: dict = {"n_questions": len(qs), "ranking": {}, "generation": {}}
        for retr in ("dense", "bm25"):
            pools = [q["pool"][retr] for q in qs]
            n_pool = len(pools[0])
            prompts = [rank_prompt(q["q"], passages[p]) for q, pool in zip(qs, pools) for p in pool]
            t0 = time.time()
            sc = rank_scores(tok, model, prompts).reshape(len(qs), n_pool)
            sec = time.time() - t0
            gold = [set(q["gold"]) for q in qs]
            rr = {"retriever": {}, "reranked": {}, "sec_per_question": sec / len(qs)}
            for k in (1, 3, 5, 10):
                rr["retriever"][f"R@{k}"] = safe_mean([recall_at_k(pool, g, k) for pool, g in zip(pools, gold)])
                rr["reranked"][f"R@{k}"] = safe_mean([
                    recall_at_k([pool[j] for j in np.argsort(-s, kind="stable")], g, k) for pool, s, g in zip(pools, sc, gold)])
            rr["retriever"]["MRR"] = safe_mean([mrr(pool, g) for pool, g in zip(pools, gold)])
            rr["reranked"]["MRR"] = safe_mean([mrr([pool[j] for j in np.argsort(-s, kind="stable")], g) for pool, s, g in zip(pools, sc, gold)])
            for label, key in (("retriever", None), ("reranked", sc)):
                vals = [pool_auroc(-np.arange(n_pool, dtype=np.float32) if key is None else key[i], pool, g)
                        for i, (pool, g) in enumerate(zip(pools, gold))]
                rr[label]["AUROC"] = safe_mean([v for v in vals if v is not None])
            rr["rerank_by_N"] = {}
            for n in n_sweep:  # 上位 N 件だけを rerank した時の R@5(論文 Fig. 5 の N に対応)
                rr["rerank_by_N"][str(n)] = safe_mean([
                    recall_at_k([pool[j] for j in np.argsort(-s[:n], kind="stable")], g, 5) for pool, s, g in zip(pools, sc, gold)])
            r["ranking"][retr] = rr
            log(f"  [{name}/{retr}] R@5 retriever {rr['retriever']['R@5']:.3f} -> rerank {rr['reranked']['R@5']:.3f}  ({sec:.0f}s)")
            if retr == "dense":
                dense_pools, dense_sc = pools, sc
        if do_gen:
            def run(label, ctx_ids):
                t0 = time.time()
                preds = generate(tok, model, [qa_prompt(q["q"], [passages[p] for p in ids]) for q, ids in zip(qs, ctx_ids)])
                m = _gen_metrics(preds, qs)
                m["sec_per_question"] = (time.time() - t0) / len(qs)
                m["examples"] = [{"q": q["q"], "pred": p, "gold": q["answers"]} for q, p in zip(qs[:4], preds[:4])]
                r["generation"][label] = m
                log(f"  [{name}] {label}: EM {m['EM']:.3f} F1 {m['F1']:.3f} Acc {m['Acc']:.3f}")

            order = [[pool[j] for j in np.argsort(-s, kind="stable")] for pool, s in zip(dense_pools, dense_sc)]
            for k in ks:
                run(f"retriever_k{k}", [pool[:k] for pool in dense_pools])
                run(f"rerank_k{k}", [o[:k] for o in order])
            for n in n_sweep:
                if n != 30:
                    run(f"rerank_N{n}_k5", [[pool[j] for j in np.argsort(-s[:n], kind="stable")][:5] for pool, s in zip(dense_pools, dense_sc)])
            run("gold", [q["gold"] for q in qs])
        res[name] = r
    return res


# ---------------------------------------------------------------- 1 つの条件(学習 + 評価)
def run_job(cfg: dict, data_dir: str, work_dir: str, n_eval: int = 500, log=print) -> dict:
    data, work = Path(data_dir), Path(work_dir)
    out: dict = {"cfg": cfg}
    adapter = None
    if cfg.get("eval_only"):  # 保存済みの LoRA で評価だけをやり直す(AUROC の追加など)
        adapter = work / "adapters" / cfg["name"]
        if not adapter.exists():
            raise FileNotFoundError(f"保存済みの adapter がありません: {adapter}")
    elif cfg["train"]:
        adapter = work / "adapters" / cfg["name"]
        adapter.mkdir(parents=True, exist_ok=True)
        out["train"] = train(cfg, data, adapter, log)
    tok, model = load(cfg["model"], str(adapter) if adapter else None)
    t0 = time.time()
    out["eval"] = evaluate(tok, model, data, n_eval, do_gen=cfg["do_gen"], log=log)
    out["eval_sec"] = round(time.time() - t0)
    return out
