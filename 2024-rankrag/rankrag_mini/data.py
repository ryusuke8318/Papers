"""学習ブレンドと評価プールの作成(CPU でも動く。dense retriever だけは GPU があれば使う)。

学習ブレンド(論文 Stage-II の縮小版)は、部品ごとに jsonl で保存し、条件ごとに組み合わせる。
  qa.jsonl    Context-rich QA        SQuAD、gold passage 1 件
  rqa.jsonl   Retrieval-augmented QA SQuAD、gold + BM25 の hard negative で計 5 件
  rar.jsonl   Retrieval-augmented ranking  同上の 5 件から、答えを含む passage の番号を出力
  rank.jsonl  Context ranking        MS MARCO、(正例, 負例) を交互に並べる(先頭 R 行が常に均衡)

評価プール: 各質問について、retriever(dense / BM25)が返した上位 N=30 件の passage 番号を保存する。
  squad    SQuAD 開発セット 500 問、コーパスは開発セットの全段落(約 2,000)
  hotpot   HotpotQA distractor の開発セット 500 問、コーパスは 500 問の全段落(約 5,000)。学習に使わない(zero-shot)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rankrag_core import BM25, MAX_WORDS, answer_in_text, qa_prompt, rank_prompt, rar_prompt, truncate_words  # noqa: E402

N_POOL = 30
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _answer_reachable(context: str, answers: dict) -> bool:
    """切り詰めた passage の中に答えが残っているか。"""
    s, t = answers["answer_start"][0], answers["text"][0]
    return len(context[: s + len(t)].split()) <= MAX_WORDS


def _squad_passages(ds):
    """SQuAD の (title, context) を重複なしの passage に直し、各問の gold 番号を返す。"""
    index: dict[tuple, int] = {}
    passages: list[str] = []
    gold: list[int] = []
    for ex in ds:
        key = (ex["title"], ex["context"])
        if key not in index:
            index[key] = len(passages)
            passages.append(f"{ex['title'].replace('_', ' ')}: {truncate_words(ex['context'])}")
        gold.append(index[key])
    return passages, gold


# ---------------------------------------------------------------- dense retriever
class Dense:
    def __init__(self, model_name: str = DENSE_MODEL):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.dev).eval()
        if self.dev == "cuda":
            self.model.half()

    def encode(self, texts: list[str], prefix: str = "", bs: int = 128) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = self.tok([prefix + t for t in texts[i : i + bs]], padding=True, truncation=True, max_length=256,
                               return_tensors="pt").to(self.dev)
                h = self.model(**enc).last_hidden_state[:, 0]  # CLS
                out.append(self.torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)


def _pools(dense: Dense, passages: list[str], questions: list[str], n: int = N_POOL) -> dict:
    p_emb = dense.encode(passages)
    q_emb = dense.encode(questions, prefix=QUERY_PREFIX)
    sims = q_emb @ p_emb.T
    dense_pool = [[int(i) for i in np.argsort(-s)[:n]] for s in sims]
    bm = BM25(passages)
    bm25_pool = [bm.topk(q, n) for q in questions]
    return {"dense": dense_pool, "bm25": bm25_pool}


# ---------------------------------------------------------------- 学習ブレンド
def build_train(out_dir: Path, n_qa=4000, n_rqa=4000, n_rar=2000, n_rank_queries=10000, seed=0):
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    ds = load_dataset("rajpurkar/squad", split="train")
    passages, gold = _squad_passages(ds)
    keep = [i for i, ex in enumerate(ds) if _answer_reachable(ex["context"], ex["answers"])]
    order = [keep[i] for i in rng.permutation(len(keep))]
    qa_idx, rqa_idx, rar_idx = order[:n_qa], order[n_qa : n_qa + n_rqa], order[n_qa + n_rqa : n_qa + n_rqa + n_rar]
    bm = BM25(passages)

    def five(i):
        ex = ds[i]
        negs = [p for p in bm.topk(ex["question"], 10) if p != gold[i]][:4]
        ids = [gold[i]] + negs
        ids = [ids[j] for j in rng.permutation(len(ids))]
        return ex, ids

    qa = []
    for i in qa_idx:
        ex = ds[i]
        qa.append({"task": "qa", "prompt": qa_prompt(ex["question"], [passages[gold[i]]]), "answer": ex["answers"]["text"][0]})
    rqa, rar = [], []
    for i in rqa_idx:
        ex, ids = five(i)
        rqa.append({"task": "rqa", "prompt": qa_prompt(ex["question"], [passages[p] for p in ids]), "answer": ex["answers"]["text"][0]})
    for i in rar_idx:
        ex, ids = five(i)
        rel = [j + 1 for j, p in enumerate(ids) if answer_in_text(ex["answers"]["text"], passages[p])]
        rar.append({"task": "rar", "prompt": rar_prompt(ex["question"], [passages[p] for p in ids]),
                    "answer": ", ".join(str(j) for j in rel)})

    # Context ranking(MS MARCO)。負例は検索結果の中で選ばれなかった passage(論文は BM25 の hard negative。仮定)
    mm = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True).shuffle(seed=seed, buffer_size=5000)
    rank = []
    nq = 0
    for ex in mm:
        texts, sel = ex["passages"]["passage_text"], ex["passages"]["is_selected"]
        pos = [t for t, s in zip(texts, sel) if s == 1]
        neg = [t for t, s in zip(texts, sel) if s == 0]
        if not pos or not neg:
            continue
        q = ex["query"]
        rank.append({"task": "rank", "prompt": rank_prompt(q, truncate_words(pos[0])), "answer": "True"})
        rank.append({"task": "rank", "prompt": rank_prompt(q, truncate_words(neg[int(rng.integers(len(neg)))])), "answer": "False"})
        nq += 1
        if nq >= n_rank_queries:
            break

    for name, rows in (("qa", qa), ("rqa", rqa), ("rar", rar), ("rank", rank)):
        write_jsonl(out_dir / f"{name}.jsonl", rows)
    meta = {"qa": len(qa), "rqa": len(rqa), "rar": len(rar), "rank": len(rank), "squad_passages": len(passages)}
    (out_dir / "train_meta.json").write_text(json.dumps(meta))
    return meta


# ---------------------------------------------------------------- 評価プール
def _load_hotpot():
    from datasets import load_dataset

    errs = []
    for kw in ({}, {"revision": "refs/convert/parquet"}):
        try:
            return load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", **kw)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{kw}: {type(e).__name__}: {str(e)[:150]}")
    raise RuntimeError("HotpotQA を読み込めません: " + " | ".join(errs))


def build_eval(out_dir: Path, n_eval=500, seed=0):
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    dense = Dense()

    # ---- SQuAD 開発セット
    ds = load_dataset("rajpurkar/squad", split="validation")
    passages, gold = _squad_passages(ds)
    keep = [i for i, ex in enumerate(ds) if _answer_reachable(ex["context"], ex["answers"])]
    pick = sorted(rng.choice(keep, size=min(n_eval, len(keep)), replace=False).tolist())
    qs = [ds[i]["question"] for i in pick]
    pools = _pools(dense, passages, qs)
    squad = {"passages": passages, "questions": [
        {"q": ds[i]["question"], "answers": sorted(set(ds[i]["answers"]["text"])), "gold": [gold[i]],
         "pool": {r: pools[r][k] for r in pools}} for k, i in enumerate(pick)]}
    (out_dir / "eval_squad.json").write_text(json.dumps(squad, ensure_ascii=False))

    # ---- HotpotQA(zero-shot。学習には使わない)
    hp = _load_hotpot()
    pick = sorted(rng.choice(len(hp), size=min(n_eval, len(hp)), replace=False).tolist())
    title_id: dict[str, int] = {}
    hp_passages: list[str] = []
    questions = []
    for i in pick:
        ex = hp[i]
        for t, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            if t not in title_id:
                title_id[t] = len(hp_passages)
                hp_passages.append(f"{t}: {truncate_words(' '.join(s.strip() for s in sents))}")
        sup = sorted({title_id[t] for t in ex["supporting_facts"]["title"] if t in title_id})
        questions.append({"q": ex["question"], "answers": [ex["answer"]], "gold": sup})
    pools = _pools(dense, hp_passages, [q["q"] for q in questions])
    for k, q in enumerate(questions):
        q["pool"] = {r: pools[r][k] for r in pools}
    (out_dir / "eval_hotpot.json").write_text(json.dumps({"passages": hp_passages, "questions": questions}, ensure_ascii=False))
    return {"squad": [len(squad["passages"]), len(squad["questions"])], "hotpot": [len(hp_passages), len(questions)]}


def prepare(out_dir: str, smoke: bool = False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if smoke:
        meta = build_train(out, n_qa=48, n_rqa=48, n_rar=32, n_rank_queries=60)
        ev = build_eval(out, n_eval=24)
    else:
        meta = build_train(out)
        ev = build_eval(out)
    print({"train": meta, "eval": ev})
    return {"train": meta, "eval": ev}
