"""3 つの経路(one-shot RAG、PRUNE、IRCOT*)の手順。LLM と検索は、関数として外から渡す(テストでは偽物を渡せる)。

  gen(prompts: list[str], max_tokens: int) -> list[(text, n_prompt_tokens, n_completion_tokens)]
  retrieve(queries: list[str], k: int) -> (ids: list[list[int]], scores: list[list[float]])   # 降順

質問は、バッチでまとめて処理する(経路の中の段階ごとに、全質問の呼び出しを一度に gen へ渡す)。
トークン数は、その経路が使った LLM 呼び出しの、プロンプト + 生成の合計。PRUNE と IRCOT* の合計には、1 回目(one-shot)の分も含める。
プロンプトは、論文の付録 B.1 の記述(役割と出力形式)を参考に、この実装用に書き直したもの。
"""
from __future__ import annotations

from raser_core import extract_features, jaccard, parse_bridges, verify_branch

P_ONE = ("Answer the following question concisely based on the given context. If you cannot find the answer, reply \"I don't know\".\n\n"
         "Context:\n{ctx}\n\nQuestion: {q}\nAnswer (be concise, just the answer):")

P_BRIDGE = ("You are decomposing a multi-hop question to find intermediate bridge entities. Identify up to 2 intermediate facts that could help answer the question.\n"
            "For each, output one JSON object on its own line with these fields: "
            "\"bridge_entity\" (the intermediate entity or fact, at most 5 words), \"bridge_relation\" (what role it plays), "
            "\"missing_slot\" (what is still missing after this bridge), \"confidence\" (a number from 0.0 to 1.0).\n\n"
            "Passages:\n{ctx}\n\nQuestion: {q}\nCurrent answer: {ans}\nBridge proposals (one JSON per line):")

P_SYNTH = ("Answer the following question concisely based on the given context. If you cannot find the answer, reply \"I don't know\".\n"
           "Intermediate facts already established by decomposition (treat them as reliable bridge entities):\n{facts}\n\n"
           "Context:\n{ctx}\n\nQuestion: {q}\nAnswer (be concise, just the answer):")

P_FACT = ("You are decomposing a multi-hop question. Read the passages and extract ONE intermediate fact that helps answer the question. "
          "The fact MUST be a short answer string (an entity name, date or place, at most 5 words), not a sentence, not an explanation, no reasoning. "
          "If the passages already contain everything needed to answer the question, reply DONE.\n"
          "Reply with only the fact itself (for example a person's name, a year, or a place name), or DONE.\n\n"
          "Passages:\n{ctx}\n\nFacts already found: {facts}\nQuestion: {q}\nFact:")


def fmt_ctx(ids: list[int], texts: dict[int, str]) -> str:
    return "\n\n".join(f"[{j + 1}] {texts[i]}" for j, i in enumerate(ids))


def clean_answer(t: str) -> str:
    t = t.strip().split("\n")[0].strip()
    for pre in ("Answer:", "answer:"):
        if t.startswith(pre):
            t = t[len(pre):].strip()
    return t.rstrip(".").strip() if len(t) > 1 else t


def merge_ids(*lists: list[int], cap: int = 20) -> list[int]:
    seen, out = set(), []
    for l in lists:
        for i in l:
            if i not in seen:
                seen.add(i)
                out.append(i)
    return out[:cap]


class Pipeline:
    def __init__(self, gen, retrieve, texts: dict[int, str], k: int = 10, branch_k: int = 10, max_rounds: int = 3, ans_tokens: int = 32,
                 max_ctx_chunks: int = 15, jaccard_stop: float = 0.6):
        self.gen, self.retrieve, self.texts = gen, retrieve, texts
        self.k, self.branch_k, self.max_rounds, self.ans_tokens, self.cap, self.jst = k, branch_k, max_rounds, ans_tokens, max_ctx_chunks, jaccard_stop

    # --- one-shot
    def one_shot(self, qs: list[str]) -> list[dict]:
        ids, scores = self.retrieve(qs, self.k)
        outs = self.gen([P_ONE.format(ctx=fmt_ctx(i, self.texts), q=q) for q, i in zip(qs, ids)], self.ans_tokens)
        res = []
        for q, i, s, (t, a, b) in zip(qs, ids, scores, outs):
            ans = clean_answer(t)
            res.append({"ids": i, "scores": s, "answer": ans, "tokens": a + b,
                        "features": extract_features(q, ans, s, [self.texts[j] for j in i])})
        return res

    # --- PRUNE: 提案 -> 各実体で再検索 -> 検証(規則)-> 最終の合成
    def prune(self, qs: list[str], one: list[dict]) -> list[dict]:
        outs = self.gen([P_BRIDGE.format(ctx=fmt_ctx(o["ids"], self.texts), q=q, ans=o["answer"]) for q, o in zip(qs, one)], 200)
        bridges = [parse_bridges(t) for t, _, _ in outs]
        tok = [o["tokens"] + a + b for o, (_, a, b) in zip(one, outs)]
        # 実体ごとの再検索(質問をまたいでまとめる)
        flat = [(qi, e) for qi, bs in enumerate(bridges) for e in bs]
        bids = self.retrieve([e for _, e in flat], self.branch_k)[0] if flat else []
        kept: list[list[str]] = [[] for _ in qs]
        extra: list[list[int]] = [[] for _ in qs]
        info: list[list[dict]] = [[] for _ in qs]
        for (qi, e), ids in zip(flat, bids):
            v = verify_branch(e, ids, one[qi]["ids"], self.texts)
            info[qi].append({"entity": e, **v})
            if v["keep"]:
                kept[qi].append(e)
                extra[qi] = merge_ids(extra[qi], ids, cap=10 ** 6)
        prompts = []
        for qi, q in enumerate(qs):
            ctx_ids = merge_ids(extra[qi], one[qi]["ids"], cap=self.cap)   # 枝の塊を先に、1 回目の塊を後に並べ、上限で切る
            facts = "\n".join(f"- {e}" for e in kept[qi]) or "- (none)"
            prompts.append(P_SYNTH.format(facts=facts, ctx=fmt_ctx(ctx_ids, self.texts), q=q))
        fin = self.gen(prompts, self.ans_tokens)
        return [{"answer": clean_answer(t), "tokens": tk + a + b, "raw": outs[qi][0][:300], "bridges": bridges[qi], "kept": kept[qi], "verifier": info[qi]}
                for qi, ((t, a, b), tk) in enumerate(zip(fin, tok))]

    # --- IRCOT*: 事実の抽出 -> 質問 + 事実で再検索 を最大 max_rounds 回、最後に合成
    def ircot(self, qs: list[str], one: list[dict]) -> list[dict]:
        n = len(qs)
        facts: list[list[str]] = [[] for _ in qs]
        cur = [list(o["ids"]) for o in one]
        pool = [list(o["ids"]) for o in one]
        tok = [o["tokens"] for o in one]
        active = list(range(n))
        rounds = [0] * n
        for _ in range(self.max_rounds):
            if not active:
                break
            outs = self.gen([P_FACT.format(ctx=fmt_ctx(cur[i], self.texts), facts="; ".join(facts[i]) or "(none)", q=qs[i]) for i in active], 24)
            nxt = []
            for i, (t, a, b) in zip(active, outs):
                tok[i] += a + b
                rounds[i] += 1
                f = t.strip().split("\n")[0].split(" / ")[0].strip().strip(".")
                if not f or f.upper().startswith("DONE"):
                    continue
                facts[i].append(f)
                nxt.append(i)
            if not nxt:
                break
            ids, _ = self.retrieve([qs[i] + " " + " ".join(facts[i]) for i in nxt], self.k)
            active = []
            for i, new in zip(nxt, ids):
                same = jaccard(new, cur[i]) > self.jst
                pool[i] = merge_ids(pool[i], new, cap=10 ** 6)
                cur[i] = new
                if not same:
                    active.append(i)
        prompts = [P_SYNTH.format(facts="\n".join(f"- {x}" for x in facts[i]) or "- (none)",
                                  ctx=fmt_ctx(merge_ids(cur[i], pool[i], cap=self.cap), self.texts), q=qs[i]) for i in range(n)]
        fin = self.gen(prompts, self.ans_tokens)
        return [{"answer": clean_answer(t), "tokens": tok[i] + a + b, "facts": facts[i], "rounds": rounds[i]} for i, (t, a, b) in enumerate(fin)]

    def run(self, qs: list[str]) -> list[dict]:
        one = self.one_shot(qs)
        pr = self.prune(qs, one)
        ir = self.ircot(qs, one)
        return [{"one_shot": o, "prune": p, "ircot": c} for o, p, c in zip(one, pr, ir)]
