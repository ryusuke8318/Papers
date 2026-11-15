"""評価(コンテナ内で `python evaluate.py '<json>'` として実行)。

質問ごとに、候補プール(検索で集めたチャンク)・答え・LLM 呼び出し数を記録する。手法:
  direct / cot            : 検索なし
  direct_r10 / r20 / r30  : 原問いで上位 K 件を検索して回答
  decompose_r             : LLM で分解し、副問いごとに上位 10 件を検索して回答
  iter_retgen2 / 3        : 検索 → 生成を 2, 3 回(次の検索の問いは 原問い + 直前の生成)。毎回上位 10 件
  selfask                 : Self-Ask(副問い → 検索 → 中間の答え)。副問いごとに上位 10 件
  efficientrag[_<variant>]: Labeler / Filter による反復検索(LLM は最後の回答の 1 回だけ)
手法は、質問のリストをまとめて処理する(精度の評価はバッチ、遅延の測定は質問 1 件ずつ)。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, "/root/pkg/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from config import ENCODER, INFER, MAX_CHUNK_WORDS  # noqa: E402
from efficientrag_core import (assemble_query, exact_match, extract_json_value, f1_score, parse_selfask, pool_size, recall_at_pool,  # noqa: E402
                               run_loop, split_words)

# ------------------------------------------------------------------ プロンプト
QA_DOCS = """As an assistant, your task is to answer the question using the documents below. Your answer should be in JSON format with key "answer" and its value should be a short string (an entity, a date, a number, or yes/no).

{docs}

<Question>: {question}
<Answer>:"""

QA_DIRECT = """As an assistant, your task is to answer the question directly after <Question>. Your answer should be after <Answer> in JSON format with key "answer" and its value should be a short string (an entity, a date, a number, or yes/no).

<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
<Answer>: {{"answer": "Little Richard"}}
<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
<Answer>: {{"answer": "Chinua Achebe"}}
<Question>: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
<Answer>: {{"answer": "The Mask Of Fu Manchu"}}

<Question>: {question}
<Answer>:"""

COT_DOCS = """As an assistant, your task is to answer the question using the documents below. First think step by step and give your thought, then answer. Your answer should be in JSON format with keys "thought" and "answer"; "answer" should be a short string (an entity, a date, a number, or yes/no).

{docs}

<Question>: {question}
<Answer>:"""

COT_DIRECT = """As an assistant, your task is to answer the question after <Question>. First think step by step and give your thought, then answer. Your answer should be after <Answer> in JSON format with keys "thought" and "answer"; "answer" should be a short string (an entity, a date, a number, or yes/no).

<Question>: Where was the director of film The Fascist born?
<Answer>: {{"thought": "The director of The Fascist is Luciano Salce, who was born in Rome. So the answer is Rome.", "answer": "Rome"}}

<Question>: {question}
<Answer>:"""

DECOMPOSE = """You are assigned a multi-hop question decomposition task.
You should decompose the given multi-hop question into multiple single-hop questions, such that you can answer each single-hop question independently.
You should answer in JSON format with the key "decomposed_questions": a list of strings, each string is a single-hop question.

<Multi-hop question>: Which film came out first, The Love Route or Engal Aasan?
Your response: {{"decomposed_questions": ["When does the film The Love Route come out?", "When does the film Engal Aasan come out?"]}}
<Multi-hop question>: Where did the spouse of Moderen's composer die?
Your response: {{"decomposed_questions": ["Who is Modern's composer?", "Who is the spouse of Carl Nielsen?", "In what place did Anne Marie Carl-Nielsen die?"]}}
<Multi-hop question>: Where was the director of film The Fascist born?
Your response: {{"decomposed_questions": ["Who is the director of film The Fascist?", "Where was Luciano Salce born?"]}}

<Multi-hop question>: {question}
Your response:"""

SELFASK = """Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Are both the directors of Jaws and Casino Royale from the same country?
Are follow up questions needed here: Yes.
Follow up: Who directed Jaws?
Intermediate answer: Jaws was directed by Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate answer: The United States.
Follow up: Who directed Casino Royale?
Intermediate answer: Casino Royale was directed by Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate answer: New Zealand.
So the final answer is: No

Question: {question}
Are follow up questions needed here:"""

SUB_ANSWER = """Answer the question in one short sentence, using only the documents below.

{docs}

Question: {question}
Answer:"""

JUDGE = """You are an experienced linguist who is responsible for evaluating the correctness of the generated responses. You are provided with question, the generated responses and the corresponding ground truth answer. Your task is to compare the generated responses with the ground truth responses and evaluate the correctness of the generated responses. Response in JSON format with key "response" and value "yes" or "no".
Question: {question}
Prediction: {prediction}
Ground-truth Answer: {answer}
Your response:"""


# ------------------------------------------------------------------ 実行環境
class Ctx:
    def __init__(self, llm_name: str, corpus: dict[str, str], models: dict | None):
        from vllm import LLM
        self.llm = LLM(model=llm_name, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.5, seed=0)
        self.tok = self.llm.get_tokenizer()
        self.text = corpus
        ids = list(corpus)
        from retriever import Index
        self.index = Index(ids, [corpus[i] for i in ids])
        self.cw = {}
        self.calls = 0
        self.out_tokens = 0
        self.models = models or {}

    def words(self, cid: str) -> list[str]:
        if cid not in self.cw:
            self.cw[cid] = split_words(self.text[cid])[:MAX_CHUNK_WORDS]
        return self.cw[cid]

    def gen(self, prompts: list[str], max_tokens: int, stop: list[str] | None = None) -> list[str]:
        from vllm import SamplingParams
        ps = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
        outs = self.llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=stop), use_tqdm=False)
        self.calls += len(prompts)
        self.out_tokens += sum(len(o.outputs[0].token_ids) for o in outs)
        return [o.outputs[0].text for o in outs]

    def docs(self, ids: list[str]) -> str:
        return "\n\n".join(f"[{i + 1}] {' '.join(self.text[c].split()[:200])}" for i, c in enumerate(ids))


def parse_answer(o: str) -> str:
    v = extract_json_value(o, "answer")
    return (v if v is not None else o.strip().split("\n")[0]).strip()



# ------------------------------------------------------------------ 基準手法(質問のリストをまとめて処理する)
def m_direct(ctx, qs, cot=False):
    c0 = ctx.calls
    outs = ctx.gen([(COT_DIRECT if cot else QA_DIRECT).format(question=q["question"]) for q in qs], 300 if cot else 60)
    return [{"id": q["id"], "pool": [], "answer": parse_answer(o), "llm_calls": 1} for q, o in zip(qs, outs)]


def m_direct_r(ctx, qs, k, qa=True):
    tops = ctx.index.search([q["question"] for q in qs], k)
    outs = ctx.gen([QA_DOCS.format(docs=ctx.docs(t), question=q["question"]) for q, t in zip(qs, tops)], 60) if qa else [""] * len(qs)
    return [{"id": q["id"], "pool": t, "answer": parse_answer(o) if qa else "", "llm_calls": 1 if qa else 0} for q, t, o in zip(qs, tops, outs)]


def m_decompose_r(ctx, qs, k=10):
    outs = ctx.gen([DECOMPOSE.format(question=q["question"]) for q in qs], 300)
    subs = []
    for q, o in zip(qs, outs):
        raw = o
        s, e = raw.find("["), raw.rfind("]")
        try:
            lst = [str(x) for x in json.loads(raw[s:e + 1])]
        except Exception:  # noqa: BLE001
            lst = []
        subs.append(lst[:4] or [q["question"]])
    flat = [(i, s) for i, ss in enumerate(subs) for s in ss]
    tops = ctx.index.search([s for _, s in flat], k)
    pools = [[] for _ in qs]
    for (i, _), t in zip(flat, tops):
        pools[i] += t
    pools = [list(dict.fromkeys(p)) for p in pools]
    ans = ctx.gen([QA_DOCS.format(docs=ctx.docs(p), question=q["question"]) for q, p in zip(qs, pools)], 60)
    return [{"id": q["id"], "pool": p, "answer": parse_answer(a), "llm_calls": 2} for q, p, a in zip(qs, pools, ans)]


def m_iter_retgen(ctx, qs, iters, k=10):
    pools = [[] for _ in qs]
    queries = [q["question"] for q in qs]
    last = [""] * len(qs)
    for _ in range(iters):
        tops = ctx.index.search(queries, k)
        outs = ctx.gen([COT_DOCS.format(docs=ctx.docs(t), question=q["question"]) for q, t in zip(qs, tops)], 300)
        for i, (t, o) in enumerate(zip(tops, outs)):
            pools[i] += t
            last[i] = o
            th = extract_json_value(o, "thought") or o
            queries[i] = f"{qs[i]['question']} {th}"[:1500]
    return [{"id": q["id"], "pool": list(dict.fromkeys(p)), "answer": parse_answer(o), "llm_calls": iters} for q, p, o in zip(qs, pools, last)]


def m_selfask(ctx, qs, k=10, max_hops=4):
    st = [{"text": SELFASK.format(question=q["question"]), "pool": [], "calls": 0, "answer": None} for q in qs]
    for r in range(max_hops + 1):
        act = [i for i, s in enumerate(st) if s["answer"] is None]
        if not act:
            break
        outs = ctx.gen([st[i]["text"] for i in act], 120, stop=["Intermediate answer:"])
        need = []
        for i, o in zip(act, outs):
            s = st[i]
            s["calls"] += 1
            kind, val = parse_selfask(o)
            if kind == "final":
                s["answer"] = val
            elif kind == "follow_up" and r < max_hops:
                s["text"] += o.rstrip() + "\nIntermediate answer:"
                need.append((i, val))
            else:
                s["answer"] = val or o.strip().split("\n")[-1]
        if need:
            tops = ctx.index.search([v for _, v in need], k)
            sub = ctx.gen([SUB_ANSWER.format(docs=ctx.docs(t), question=v) for (_, v), t in zip(need, tops)], 80)
            for (i, _), t, a in zip(need, tops, sub):
                st[i]["pool"] += t
                st[i]["calls"] += 1
                st[i]["text"] += " " + a.strip().split("\n")[0] + "\n"
    # 最後に、集めたチャンクで原問いに答える設定にはしない(Self-Ask は自分の最終答えを使う)
    return [{"id": q["id"], "pool": list(dict.fromkeys(s["pool"])), "answer": s["answer"] or "", "llm_calls": s["calls"]} for q, s in zip(qs, st)]


# ------------------------------------------------------------------ EfficientRAG
def load_tagger(path: str):
    from models import Tagger
    from transformers import AutoTokenizer
    m = Tagger(ENCODER)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to("cuda").eval(), AutoTokenizer.from_pretrained(ENCODER)


def make_fns(ctx, lab, fil, k, threshold, variant):
    from models import predict
    lm, lt = lab

    def retrieve(query, kk):
        return ctx.index.search([query], kk)[0]

    def label(query, ids):
        exs = [{"a": split_words(query), "b": ctx.words(c), "la": None, "lb": None, "tag": None} for c in ids]
        out = []
        for e, (p, _, pb) in zip(exs, predict(lm, lt, exs)):
            out.append((1.0 if variant == "no_tag" else p, [w for w, x in zip(e["b"], pb) if x >= threshold]))
        return out

    def filt(question, info):
        if not info:
            return ""
        if variant == "no_filter":     # Filter を使わず、原問いに Info の語をそのまま足す
            return question + " " + " ".join(info)
        fm, ft = fil
        e = {"a": split_words(question), "b": info, "la": None, "lb": None, "tag": None}
        _, pa, pb = predict(fm, ft, [e])[0]
        return assemble_query(e["a"], [int(x >= threshold) for x in pa], info, [int(x >= threshold) for x in pb])
    return retrieve, label, filt


def m_efficientrag(ctx, qs, lab, fil, variant="full", k=INFER["k"], max_iter=INFER["max_iter"], threshold=INFER["threshold"], qa=True):
    retrieve, label, filt = make_fns(ctx, lab, fil, k, threshold, variant)
    results, pools = [], []
    for q in qs:
        r = run_loop(q["question"], retrieve, label, filt, k=k, max_iter=max_iter, threshold=threshold)
        pools.append(r.pool)
        results.append({"id": q["id"], "pool": r.pool, "iters": r.iterations, "queries": r.queries[:8], "label_calls": r.n_label_calls, "llm_calls": 1 if qa else 0})
    if qa:
        outs = ctx.gen([QA_DOCS.format(docs=ctx.docs(p), question=q["question"]) for q, p in zip(qs, pools)], 60)
        for r, o in zip(results, outs):
            r["answer"] = parse_answer(o)
    else:
        for r in results:
            r["answer"] = ""
    return results


# ------------------------------------------------------------------ 採点
def score(ctx, qs, results):
    gold = {q["id"]: q for q in qs}
    judged = ctx.gen([JUDGE.format(question=gold[r["id"]]["question"], prediction=r["answer"], answer=gold[r["id"]]["answer"]) for r in results], 20) \
        if any(r["answer"] for r in results) else [""] * len(results)
    c = ctx.calls
    for r, j in zip(results, judged):
        q = gold[r["id"]]
        r["recall"] = recall_at_pool(r["pool"], q["gold"])
        r["K"] = pool_size(r["pool"])
        r["em"] = exact_match(r["answer"], q["answer"]) if r["answer"] else 0.0
        r["f1"] = f1_score(r["answer"], q["answer"]) if r["answer"] else 0.0
        r["acc"] = float("yes" in (extract_json_value(j, "response") or j).lower()) if r["answer"] else 0.0
    return results


def main(a: dict):
    root = Path(a["root"])
    corpus = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
    qs = [json.loads(l) for l in (root / "eval_q.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:a.get("limit")]
    ctx = Ctx(a["llm"], corpus, None)
    out = {"dataset": a["dataset"], "n": len(qs), "corpus": len(corpus), "methods": {}, "sec": {}}
    outp = Path(a["out"])
    outp.parent.mkdir(parents=True, exist_ok=True)

    def record(name, fn):
        t = time.time()
        c0, t0 = ctx.calls, ctx.out_tokens
        res = score(ctx, qs, fn())
        out["methods"][name] = res
        out["sec"][name] = round(time.time() - t, 1)
        outp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        mean = lambda k: sum(r[k] for r in res) / len(res)  # noqa: E731
        print(f"{name}: recall {mean('recall'):.3f} K {mean('K'):.2f} em {mean('em'):.3f} f1 {mean('f1'):.3f} acc {mean('acc'):.3f} llm {mean('llm_calls'):.2f} ({time.time() - t:.0f}s)", flush=True)

    base = a.get("baselines", True)
    if base:
        record("direct", lambda: m_direct(ctx, qs))
        record("cot", lambda: m_direct(ctx, qs, cot=True))
        for k in (10, 20, 30):
            record(f"direct_r{k}", lambda k=k: m_direct_r(ctx, qs, k))
        record("decompose_r", lambda: m_decompose_r(ctx, qs))
        record("iter_retgen2", lambda: m_iter_retgen(ctx, qs, 2))
        record("iter_retgen3", lambda: m_iter_retgen(ctx, qs, 3))
        record("selfask", lambda: m_selfask(ctx, qs))
    # EfficientRAG の各条件: "<名前>": {"labeler": パス, "filter": パス, "variant": ..., k, threshold}
    cache = {}
    for name, spec in a.get("efficientrag", {}).items():
        for p in (spec["labeler"], spec["filter"]):
            if p not in cache:
                cache[p] = load_tagger(p)
        kw = {x: spec[x] for x in ("k", "threshold", "max_iter") if x in spec}
        record(name, lambda spec=spec, kw=kw: m_efficientrag(ctx, qs, cache[spec["labeler"]], cache[spec["filter"]], spec.get("variant", "full"), qa=spec.get("qa", True), **kw))

    # ---- 遅延: 質問 1 件ずつ、検索・エンコーダ・LLM を通して測る
    lat = a.get("latency")
    if lat:
        n = lat["n"]
        sub = qs[:n]
        er = a["efficientrag"][lat["efficientrag"]]
        lab, fil = cache[er["labeler"]], cache[er["filter"]]
        fns = {"direct": lambda q: m_direct(ctx, q), "direct_r10": lambda q: m_direct_r(ctx, q, 10),
               "iter_retgen3": lambda q: m_iter_retgen(ctx, q, 3), "selfask": lambda q: m_selfask(ctx, q),
               "efficientrag": lambda q: m_efficientrag(ctx, q, lab, fil)}
        m_direct(ctx, sub[:3])     # ウォームアップ
        out["latency"] = {}
        for name, fn in fns.items():
            ts, calls, iters = [], [], []
            for q in sub:
                t = time.time()
                r = fn([q])[0]
                torch.cuda.synchronize()
                ts.append(time.time() - t)
                calls.append(r["llm_calls"])
                iters.append(r.get("iters", 0))
            out["latency"][name] = {"sec": ts, "llm_calls": calls, "iters": iters}
            print(f"latency {name}: {sum(ts) / len(ts):.2f}s calls {sum(calls) / len(calls):.2f}", flush=True)
            outp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"done": True, "methods": list(out["methods"])}))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
