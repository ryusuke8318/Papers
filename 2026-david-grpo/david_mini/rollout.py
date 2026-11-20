"""複数回の検索を挟む rollout(生成 → 検索 → 結果を文脈に足して続ける)。"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

import common  # noqa: F401  (src を sys.path に入れる)
from david_core import (MAX_SEARCHES, SYSTEM_PROMPT, Action, Step, Trajectory, build_prompt,
                        format_information, parse_segment)

STOPS = ["</search>", "</answer>"]


@dataclass
class State:
    qi: int
    prompt_ids: list[int]
    resp_ids: list[int] = field(default_factory=list)
    mask: list[int] = field(default_factory=list)       # 1 = モデルが生成したトークン、0 = 検索結果
    traj: Trajectory = field(default_factory=Trajectory)
    cuts: list[int] = field(default_factory=list)       # k 回目の検索結果を足した直後の resp_ids の長さ
    done: bool = False
    expert: bool = False
    reward: float = 0.0

    def prefix_after_search(self, k: int) -> "State":
        """k 回目の検索結果までを残した複製(grounded expansion 用)。"""
        n = self.cuts[k - 1]
        steps, seen = [], 0
        for s in self.traj.steps:
            steps.append(Step(s.gen, s.action, list(s.retrieved)))
            if s.action.kind == "search":
                seen += 1
                if seen == k:
                    break
        return State(self.qi, list(self.prompt_ids), self.resp_ids[:n], self.mask[:n], Trajectory(steps), self.cuts[:k])


def prompt_ids(tok, question: str, max_searches: int = MAX_SEARCHES) -> list[int]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_prompt(question, max_searches)}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def _gen(model, tok, ctxs, max_new, temperature, greedy):
    pad = tok.pad_token_id
    L = max(len(c) for c in ctxs)
    ids = torch.full((len(ctxs), L), pad, dtype=torch.long)
    att = torch.zeros((len(ctxs), L), dtype=torch.long)
    for i, c in enumerate(ctxs):
        ids[i, L - len(c):] = torch.tensor(c)
        att[i, L - len(c):] = 1
    kw = dict(do_sample=not greedy, max_new_tokens=max_new, pad_token_id=pad, stop_strings=STOPS, tokenizer=tok,
              repetition_penalty=1.0, top_p=1.0)
    if greedy:
        kw.update(temperature=None, top_k=None)
    else:
        kw.update(temperature=temperature, top_k=0)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), **kw)
    res = []
    for row in out[:, L:].tolist():
        cut = len(row)
        for j, t in enumerate(row):
            if t == pad or t == tok.eos_token_id or t == 151645:
                cut = j
                break
        res.append(row[:cut])
    return res


def advance(model, tok, retriever, states: list[State], *, max_searches=MAX_SEARCHES, turn_tokens=160, max_ctx=2048,
            temperature=0.8, greedy=False, bs=96, topk=3) -> None:
    """未完了の state を、answer / invalid / 検索回数の上限まで進める。"""
    for _ in range(max_searches + 2):
        active = [s for s in states if not s.done]
        if not active:
            return
        for s in active:
            if len(s.prompt_ids) + len(s.resp_ids) + turn_tokens > max_ctx:
                s.done = True
        active = [s for s in active if not s.done]
        for i in range(0, len(active), bs):
            chunk = active[i:i + bs]
            outs = _gen(model, tok, [s.prompt_ids + s.resp_ids for s in chunk], turn_tokens, temperature, greedy)
            queries, who = [], []
            for s, ids in zip(chunk, outs):
                text = tok.decode(ids, skip_special_tokens=False)
                act = parse_segment(text)
                if act.kind == "search" and s.traj.n_searches >= max_searches:
                    act = Action("invalid")
                s.resp_ids += ids
                s.mask += [1] * len(ids)
                s.traj.steps.append(Step(text, act))
                if act.kind == "search":
                    queries.append(act.text)
                    who.append(s)
                else:
                    s.done = True
            if queries:
                for s, docs in zip(who, retriever.search(queries, k=topk)):
                    s.traj.steps[-1].retrieved = [t for t, _ in docs]
                    info = tok(format_information(docs), add_special_tokens=False)["input_ids"]
                    s.resp_ids += info
                    s.mask += [0] * len(info)
                    s.cuts.append(len(s.resp_ids))
