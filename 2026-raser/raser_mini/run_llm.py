"""1 つの LLM で、データセット × 3 経路を実行し、質問ごとの結果を JSONL に保存する(コンテナ内で実行)。

  python run_llm.py '{"model": "Qwen/Qwen2.5-7B-Instruct", "tag": "q7b", "datasets": ["musique", ...], "n": 400, "data": "/cache/raser/data", "out": "/cache/raser/results"}'

検索は、データセットごとに、質問群の候補段落を全部集めたコーパスに対する dense 検索(bge-small-en-v1.5、コサイン類似度、上位 10)。
結果は、1 つのデータセットが終わるごとに <out>/<tag>_<dataset>.jsonl に書く。最後の行に、データセットごとの所要時間とトークン数の JSON を出す。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/pkg/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from raser_core import ROUTES, chunk_passage, f1_max  # noqa: E402
from raser_pipeline import Pipeline  # noqa: E402

EMB = "BAAI/bge-small-en-v1.5"
QPREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(EMB)
        self.model = AutoModel.from_pretrained(EMB).to("cuda").eval().half()

    def count(self, s: str) -> int:
        return len(self.tok.encode(s, add_special_tokens=False))

    def encode(self, texts: list[str], bs: int = 128, max_len: int = 256):
        torch = self.torch
        outs = []
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
            with torch.no_grad():
                h = self.model(**b).last_hidden_state[:, 0]      # bge は CLS ベクトルを使う
            outs.append(torch.nn.functional.normalize(h.float(), dim=-1))
        return torch.cat(outs)


def build_corpus(rows: list[dict], emb: Embedder):
    texts: list[str] = []
    seen: dict[str, int] = {}
    for r in rows:
        for p in r["paragraphs"]:
            for c in chunk_passage(p["text"], emb.count):
                t = f"{p['title']}: {c}"
                if t not in seen:
                    seen[t] = len(texts)
                    texts.append(t)
    return texts


def make_retrieve(emb: Embedder, texts: list[str]):
    mat = emb.encode(texts)
    torch = emb.torch

    def retrieve(queries: list[str], k: int):
        q = emb.encode([QPREFIX + x for x in queries])
        s, i = torch.topk(q @ mat.T, k, dim=1)
        return i.tolist(), s.tolist()
    return retrieve


def main(args: dict):
    from vllm import LLM, SamplingParams

    emb = Embedder()
    llm = LLM(model=args["model"], dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.80, seed=0, enable_prefix_caching=True)
    tok = llm.get_tokenizer()

    def gen(prompts: list[str], max_tokens: int):
        ps = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
        outs = llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False)
        return [(o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids)) for o in outs]

    summary = {}
    Path(args["out"]).mkdir(parents=True, exist_ok=True)
    for ds in args["datasets"]:
        t0 = time.time()
        rows = [json.loads(l) for l in open(Path(args["data"]) / f"{ds}.jsonl", encoding="utf-8")][: args["n"]]
        texts = build_corpus(rows, emb)
        print(f"[{ds}] 質問 {len(rows)}、コーパスの塊 {len(texts)}", flush=True)
        pipe = Pipeline(gen, make_retrieve(emb, texts), dict(enumerate(texts)))
        res = pipe.run([r["question"] for r in rows])
        out = Path(args["out"]) / f"{args['tag']}_{ds}.jsonl"
        tot = {r: 0 for r in ROUTES}
        with open(out, "w", encoding="utf-8") as f:
            for r, x in zip(rows, res):
                golds = [r["gold"]] + list(r.get("aliases") or [])
                rec = {"id": r["id"], "dataset": ds, "model": args["model"], "question": r["question"], "gold": r["gold"], "hops": r.get("hops"),
                       "features": x["one_shot"]["features"], "top_scores": x["one_shot"]["scores"]}
                for name in ROUTES:
                    y = x[name]
                    rec[name] = {"answer": y["answer"], "f1": f1_max(y["answer"], golds), "tokens": y["tokens"]}
                    tot[name] += y["tokens"]
                rec["prune"]["bridges"], rec["prune"]["kept"], rec["prune"]["raw"] = x["prune"]["bridges"], x["prune"]["kept"], x["prune"]["raw"]
                rec["ircot"]["facts"], rec["ircot"]["rounds"] = x["ircot"]["facts"], x["ircot"]["rounds"]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary[ds] = {"n": len(rows), "sec": round(time.time() - t0), "tokens": tot,
                       "mean_f1": {r: sum(json.loads(l)[r]["f1"] for l in open(out)) / len(rows) for r in ROUTES}}
        print(f"[{ds}] 完了", json.dumps(summary[ds]), flush=True)
    print(json.dumps(summary))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
