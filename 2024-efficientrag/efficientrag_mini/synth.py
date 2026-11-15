"""合成データの生成(論文 3.2 の前半。vLLM を使う。コンテナ内で `python synth.py '<json>'` として実行)。

1. 多段の問いを、正解チャンクごとの単一ホップの副問いに分解し、依存関係を出す(Table 8 の分解プロンプトに準拠)
2. 副問いと対応するチャンクについて、答えに必要な語を抜き出す(トークンラベル)
3. 依存のある副問いについて、原問いと依存先の Info から、次の問いの語を選ぶ(Filter 用)
負例の選択(4)は、検索が要るので train.py 側で行う。出力は 1 行 1 問の jsonl。
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/pkg/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MAX_CHUNK_WORDS  # noqa: E402
from efficientrag_core import align_labels, info_chain, select_words, split_words, valid_decomposition  # noqa: E402

DECOMP = """You are assigned a multi-hop question decomposition task.
Decompose the original multi-hop question into EXACTLY {n} single-hop sub-questions: one sub-question for each of the {n} supporting documents, so that each sub-question can be answered from its own document and each document is used exactly once. Do not write extra sub-questions.
Each document is labeled #1, #2, ... and states facts as relations between entities.

Respond only with JSON containing the key "decomposed_questions": a dict whose keys are sub-question numbers (strings) and whose values are dicts with:
- "sub_question": a string, the single-hop sub-question. It MUST NOT contain more information than the original question and the answers of the sub-questions it depends on. NEVER introduce information that appears only in the documents.
- "answer": a string, the answer of the sub-question.
- "dependency": a list of sub-question numbers (strings) whose answers this sub-question needs. Use [] if it is independent (for example when comparing two entities).
- "document": the document number (e.g. "1") that supports the sub-question. Use every document exactly once.

Example (2 documents, bridge question): "Where was the director of Film X born?" gives
{{"decomposed_questions": {{"1": {{"sub_question": "Who directed Film X?", "answer": "Alice Smith", "dependency": [], "document": "1"}}, "2": {{"sub_question": "Where was Alice Smith born?", "answer": "Paris", "dependency": ["1"], "document": "2"}}}}}}
Example (2 documents, comparison question): "Which is older, A or B?" gives
{{"decomposed_questions": {{"1": {{"sub_question": "When was A founded?", "answer": "1990", "dependency": [], "document": "1"}}, "2": {{"sub_question": "When was B founded?", "answer": "1985", "dependency": [], "document": "2"}}}}}}

Question: {question}
Final answer: {answer}

{docs}

JSON:"""

LABEL = """You have been assigned an information extraction task.
Extract words from the paragraph so that others can answer the question using only your extracted words.
Your extracted words should cover information from both the question and the answer, including entities (e.g. people, locations, films) and core relations.
Respond only with JSON containing the key "extracted_words": a string of words extracted from the paragraph, separated by a space.
Guidelines:
- Do not reorder, change, or add words. Keep them exactly as in the paragraph.
- Extract ONLY words explicitly mentioned in the question or its answer, or strongly related to them.
- NEVER extract words that do not contribute meaningful information.

Question: {question}
Answer: {answer}
Paragraph: {paragraph}

JSON:"""

FILTER = """You are assigned a multi-hop question refactoring task.
Given a complex question and some known information, refactor the question: remove the parts that are already known, and keep the parts that are still unknown, together with the new knowledge needed to ask about them.
You may only pick words: first pick words from the question, then pick words from the known information, and concatenate them. Do not add, change, or reorder words. The known information starts with "Info: ".
Respond only with JSON containing the key "filtered_query": a string of the picked words, separated by a space.

Question: {question}
Info: {info}

JSON:"""


def parse_json(text: str) -> dict | None:
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        v = json.loads(text[s:e + 1])
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None


def chunk_words(text: str) -> list[str]:
    return split_words(text)[:MAX_CHUNK_WORDS]


def run(model: str, questions: list[dict], out: str, seed: int = 0) -> dict:
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=0.85, seed=seed)
    tok = llm.get_tokenizer()

    def gen(prompts: list[str], n: int, temperature: float = 0.0) -> list[str]:
        ps = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
        params = SamplingParams(temperature=temperature, top_p=0.95 if temperature else 1.0, max_tokens=n, seed=seed)
        return [o.outputs[0].text for o in llm.generate(ps, params, use_tqdm=False)]

    rng = random.Random(seed)
    stats = {"questions": len(questions), "decomp_parsed": 0, "decomp_valid": 0, "labels_prompts": 0, "filter_prompts": 0}
    t_load = time.time() - t0
    t1 = time.time()
    # ---- 1. 分解
    prompts, meta = [], []
    for q in questions:
        gold = list(q["gold"])
        rng.shuffle(gold)
        text = {p["id"]: p["text"] for p in q["paras"]}
        docs = "\n\n".join(f"#{i + 1}: {' '.join(text[g].split()[:MAX_CHUNK_WORDS])}" for i, g in enumerate(gold))
        prompts.append(DECOMP.format(question=q["question"], answer=q["answer"], docs=docs, n=len(gold)))
        meta.append((q, gold))
    recs, fails = [], []
    stats |= {"fail_json": 0, "fail_fields": 0, "fail_invalid": 0, "retried": 0}

    def parse_decomp(q, gold, o):
        j = parse_json(o)
        d = (j or {}).get("decomposed_questions")
        if not isinstance(d, dict):
            return None, "json"
        subs = {}
        try:
            for sid, v in d.items():
                di = int(str(v["document"]).strip("#")) - 1
                subs[str(sid)] = {"doc": gold[di], "deps": [str(x) for x in v.get("dependency", [])], "sub_question": str(v["sub_question"]),
                                  "answer": str(v["answer"])}
        except (KeyError, ValueError, IndexError, TypeError, AttributeError):
            return None, "fields"
        return (subs, "ok") if valid_decomposition(subs, gold) else (None, "invalid")

    pending = list(range(len(meta)))
    parsed: dict[int, dict] = {}
    for attempt in range(3):      # 失敗した問いは、温度 0.8 で最大 2 回、作り直す
        if not pending:
            break
        outs = gen([prompts[i] for i in pending], 700, 0.0 if attempt == 0 else 0.8)
        nxt = []
        for i, o in zip(pending, outs):
            subs, why = parse_decomp(meta[i][0], meta[i][1], o)
            if subs is not None:
                parsed[i] = subs
                stats["retried"] += attempt > 0
            else:
                nxt.append(i)
                if attempt == 2:
                    stats["fail_" + why] += 1
                    fails.append({"why": why, "out": o[:600], "n_gold": len(meta[i][1])})
        pending = nxt
    stats["decomp_parsed"] = len(parsed)
    for i in sorted(parsed):
        q, gold = meta[i]
        stats["decomp_valid"] += 1
        recs.append({"qid": q["id"], "question": q["question"], "answer": q["answer"], "subs": parsed[i],
                     "text": {p["id"]: p["text"] for p in q["paras"] if p["id"] in gold}})
    # ---- 2. トークンラベル
    prompts, keys = [], []
    for r in recs:
        for sid, s in r["subs"].items():
            prompts.append(LABEL.format(question=s["sub_question"], answer=s["answer"], paragraph=" ".join(chunk_words(r["text"][s["doc"]]))))
            keys.append((r, sid))
    stats["labels_prompts"] = len(prompts)
    for (r, sid), o in zip(keys, gen(prompts, 400)):
        j = parse_json(o)
        w = (j or {}).get("extracted_words")
        r["subs"][sid]["labeled"] = split_words(w) if isinstance(w, str) else []
    # ---- 3. 次の問い(依存のある副問い)
    prompts, keys = [], []
    for r in recs:
        info_of = {}
        for sid, s in r["subs"].items():
            words = chunk_words(r["text"][s["doc"]])
            lab, _ = align_labels(words, s["labeled"])
            info_of[sid] = select_words(words, lab)
        for sid, s in r["subs"].items():
            if s["deps"]:
                info = info_chain(r["subs"], info_of, sid)
                if info:
                    prompts.append(FILTER.format(question=r["question"], info=" ".join(info)))
                    keys.append((r, sid))
    stats["filter_prompts"] = len(prompts)
    for (r, sid), o in zip(keys, gen(prompts, 200)):
        j = parse_json(o)
        w = (j or {}).get("filtered_query")
        r["subs"][sid]["filtered"] = split_words(w) if isinstance(w, str) else []
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out + ".fails.json").write_text(json.dumps(fails[:30], ensure_ascii=False, indent=1), encoding="utf-8")
    Path(out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    stats |= {"records": len(recs), "sec_load": round(t_load), "sec_generate": round(time.time() - t1), "llm_prompts": stats["questions"] + stats["labels_prompts"] + stats["filter_prompts"]}
    return stats


if __name__ == "__main__":
    a = json.loads(sys.argv[1])
    qs = [json.loads(l) for l in Path(a["questions"]).read_text(encoding="utf-8").splitlines() if l.strip()][:a.get("limit")]
    print(json.dumps(run(a["model"], qs, a["out"])))
