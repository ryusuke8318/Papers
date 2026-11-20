"""vLLM を使う 3 つの作業(コンテナ内で実行)。

  teacher : 教師モデル(思考モード)で、学習用の窓ごとに推論つきの順位を n 回サンプルする
  rollout : 方策モデルから、窓ごとに n 回(GRPO のグループ)サンプルして、報酬を付ける
  eval    : 窓 1 つ(eval20)またはスライディングウィンドウ(eval100)で並べ替え、NDCG@10 などを測る
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/pkg/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reasonrank_core import (build_prompt, ndcg_at_k, parse_ranking, rbo, recall_at_k, reward, split_think,  # noqa: E402
                             window_starts)


def make_llm(model: str, max_len: int = 8192, util: float = 0.85):
    from vllm import LLM

    return LLM(model=model, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=util, enforce_eager=False, seed=0)


def chat(tok, user: str, thinking: bool | None) -> str:
    kw = {} if thinking is None else {"enable_thinking": thinking}
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, **kw)


def generate(llm, prompts: list[str], n: int, temperature: float, max_tokens: int, top_p: float = 0.95, seed: int = 0):
    from vllm import SamplingParams

    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p if temperature > 0 else 1.0, max_tokens=max_tokens, seed=seed)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    return [[(c.text, len(c.token_ids)) for c in o.outputs] for o in outs]


# ---------------------------------------------------------------- 教師データ
def teacher(model: str, lists: list[dict], n: int, max_tokens: int, out: str):
    llm = make_llm(model, max_len=max_tokens + 5500)
    tok = llm.get_tokenizer()
    prompts = [chat(tok, build_prompt(l["query"], [c["text"] for c in l["cands"]], reasoning=True), thinking=True) for l in lists]
    t0 = time.time()
    res = generate(llm, prompts, n, 0.6, max_tokens)
    rows = []
    for l, samples in zip(lists, res):
        ids = [c["id"] for c in l["cands"]]
        for text, ntok in samples:
            # Qwen3 の思考は <think>…</think> の後に本文が来る。本文から <answer> を取る
            order, exact = parse_ranking(text.split("</think>")[-1], len(ids))
            ranked = [ids[i] for i in order]
            think = text.split("</think>")[0].replace("<think>", "").strip() if "</think>" in text else ""
            rows.append({"qid": l["qid"], "think": think, "order": order, "exact": exact, "ntok": ntok,
                         "ndcg10": ndcg_at_k(ranked, l["rels"], 10), "recall10": recall_at_k(ranked, l["rels"], 10)})
    Path(out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return {"n_prompts": len(lists), "n_samples": len(rows), "sec": round(time.time() - t0)}


# ---------------------------------------------------------------- GRPO の rollout
def rollout(model: str, lists: list[dict], g: int, max_tokens: int, kind: str, out: str, seed: int = 0):
    llm = make_llm(model, max_len=max_tokens + 5000, util=0.8)
    tok = llm.get_tokenizer()
    prompts = [chat(tok, build_prompt(l["query"], [c["text"] for c in l["cands"]], reasoning=True), thinking=None) for l in lists]
    res = generate(llm, prompts, g, 1.0, max_tokens, top_p=1.0, seed=seed)
    rows, stats = [], {"reward": 0.0, "exact": 0.0, "ndcg": 0.0, "ntok": 0.0}
    for l, p, samples in zip(lists, prompts, res):
        ids = [c["id"] for c in l["cands"]]
        grp = []
        for text, ntok in samples:
            order, exact = parse_ranking(text, len(ids))
            ranked = [ids[i] for i in order]
            r = reward(ranked, l["rels"], l["gold_list"], kind, exact)
            grp.append({"prompt": p, "completion": text, "reward": r, "exact": exact, "ntok": ntok, "ndcg10": ndcg_at_k(ranked, l["rels"], 10)})
        rows.append(grp)
        for s in grp:
            stats["reward"] += s["reward"]
            stats["exact"] += s["exact"]
            stats["ndcg"] += s["ndcg10"]
            stats["ntok"] += s["ntok"]
    n = sum(len(g_) for g_ in rows)
    Path(out).write_text("\n".join(json.dumps(g_, ensure_ascii=False) for g_ in rows), encoding="utf-8")
    return {k: v / n for k, v in stats.items()}


# ---------------------------------------------------------------- 評価
def metrics(lists: list[dict], final: list[list[str]]) -> dict:
    n = len(lists)
    return {"ndcg10": sum(ndcg_at_k(f, l["rels"], 10) for l, f in zip(lists, final)) / n,
            "recall10": sum(recall_at_k(f, l["rels"], 10) for l, f in zip(lists, final)) / n,
            "ndcg1": sum(ndcg_at_k(f, l["rels"], 1) for l, f in zip(lists, final)) / n,
            "n": n}


def evaluate(model: str, lists: list[dict], reasoning: bool, w: int, s: int, max_tokens: int, thinking: bool | None, out: str | None = None, dump: str | None = None,
             temperature: float = 0.0):
    llm = make_llm(model, max_len=max_tokens + 5000)
    tok = llm.get_tokenizer()
    cur = [[c["id"] for c in l["cands"]] for l in lists]
    text_of = [{c["id"]: c["text"] for c in l["cands"]} for l in lists]
    init = metrics(lists, cur)
    ntok_total, n_calls, n_bad, t0 = 0, 0, 0, time.time()
    dumped = []
    for step in range(len(window_starts(max(len(c) for c in cur), w, s))):
        jobs = []
        for qi, ids in enumerate(cur):
            starts = window_starts(len(ids), w, s)
            if step < len(starts):
                jobs.append((qi, starts[step]))
        prompts = [chat(tok, build_prompt(lists[qi]["query"], [text_of[qi][x] for x in cur[qi][st:st + w]], reasoning), thinking) for qi, st in jobs]
        res = generate(llm, prompts, 1, temperature, max_tokens)
        for (qi, st), samples in zip(jobs, res):
            text, ntok = samples[0]
            win = cur[qi][st:st + w]
            order, exact = parse_ranking(text, len(win))
            cur[qi][st:st + w] = [win[i] for i in order]
            ntok_total += ntok
            if dump:
                dumped.append({"qid": lists[qi]["qid"], "ntok": ntok, "exact": exact, "has_think_end": "</think>" in text, "has_answer": "<answer>" in text, "text": text})
            n_calls += 1
            n_bad += not exact
    res = metrics(lists, cur)
    res["per_query_ndcg10"] = [ndcg_at_k(f, l["rels"], 10) for l, f in zip(lists, cur)]   # ブートストラップ信頼区間のため、質問ごとの値を残す
    res["init_per_query_ndcg10"] = [ndcg_at_k([c["id"] for c in l["cands"]], l["rels"], 10) for l in lists]
    res.update({"init_ndcg10": init["ndcg10"], "calls": n_calls, "out_tokens_per_query": ntok_total / len(lists), "invalid_rate": n_bad / max(1, n_calls),
                "sec": round(time.time() - t0)})
    if dump:
        Path(dump).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in dumped), encoding="utf-8")
    if out:
        Path(out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return res


if __name__ == "__main__":
    a = json.loads(sys.argv[1])
    kind = a.pop("task")
    lists = [json.loads(l) for l in open(a.pop("lists"), encoding="utf-8")]
    if a.get("limit"):
        lists = lists[:a.pop("limit")]
    else:
        a.pop("limit", None)
    fn = {"teacher": teacher, "rollout": rollout, "eval": evaluate}[kind]
    print(json.dumps(fn(lists=lists, **a), ensure_ascii=False))
