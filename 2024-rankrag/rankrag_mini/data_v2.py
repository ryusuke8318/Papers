"""第 2 版のデータ。第 1 版(data.py)で論文と合わなかった点を直した構成。

第 1 版からの変更点
  1. 評価コーパスを広げる(SQuAD 約 2 万段落、HotpotQA 約 3.5 万段落)。検索の R@5 に伸びしろを作る
  2. context ranking の負例を、MS MARCO の passage 全体に対する BM25 の hard negative にする(論文と同じ)
  3. RQA / RAR の候補を 5 件から 10 件にする
  4. 学習の QA / RQA / RAR に HotpotQA の train(多段)を半分混ぜる
  5. 評価は各 1,000 問

学習ブレンドの部品(jsonl)は第 1 版と同じ名前。評価は eval_squad.json / eval_hotpot.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from data import Dense, _answer_reachable, _pools, _squad_passages, write_jsonl  # noqa: E402
from rankrag_core import BM25, answer_in_text, qa_prompt, rank_prompt, rar_prompt, truncate_words  # noqa: E402

N_CAND = 10


def _hotpot(split: str):
    from datasets import load_dataset

    errs = []
    for kw in ({}, {"revision": "refs/convert/parquet"}):
        try:
            return load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, **kw)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{kw}: {type(e).__name__}: {str(e)[:150]}")
    raise RuntimeError("HotpotQA を読み込めません: " + " | ".join(errs))


def _hp_paragraphs(ex) -> tuple[list[str], list[int]]:
    """HotpotQA の 1 問から(10 段落, 根拠段落の番号)。"""
    paras = [f"{t}: {truncate_words(' '.join(s.strip() for s in sents))}"
             for t, sents in zip(ex["context"]["title"], ex["context"]["sentences"])]
    sup = set(ex["supporting_facts"]["title"])
    return paras, [i for i, t in enumerate(ex["context"]["title"]) if t in sup]


def build_train(out_dir: Path, n_qa=4000, n_rqa=4000, n_rar=2000, n_rank_queries=10000, seed=0):
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    half = lambda n: (n // 2, n - n // 2)  # noqa: E731  (SQuAD, HotpotQA)

    ds = load_dataset("rajpurkar/squad", split="train")
    passages, gold = _squad_passages(ds)
    keep = [i for i, ex in enumerate(ds) if _answer_reachable(ex["context"], ex["answers"])]
    order = [keep[i] for i in rng.permutation(len(keep))]
    s_qa, h_qa = half(n_qa)
    s_rqa, h_rqa = half(n_rqa)
    s_rar, h_rar = half(n_rar)
    qa_idx = order[:s_qa]
    rqa_idx = order[s_qa : s_qa + s_rqa]
    rar_idx = order[s_qa + s_rqa : s_qa + s_rqa + s_rar]
    bm = BM25(passages)

    def cands(i):
        ex = ds[i]
        negs = [p for p in bm.topk(ex["question"], N_CAND + 5) if p != gold[i]][: N_CAND - 1]
        ids = [gold[i]] + negs
        return ex, [ids[j] for j in rng.permutation(len(ids))]

    qa, rqa, rar = [], [], []
    for i in qa_idx:
        ex = ds[i]
        qa.append({"task": "qa", "src": "squad", "prompt": qa_prompt(ex["question"], [passages[gold[i]]]),
                   "answer": ex["answers"]["text"][0]})
    for i in rqa_idx:
        ex, ids = cands(i)
        rqa.append({"task": "rqa", "src": "squad", "prompt": qa_prompt(ex["question"], [passages[p] for p in ids]),
                    "answer": ex["answers"]["text"][0]})
    for i in rar_idx:
        ex, ids = cands(i)
        rel = [j + 1 for j, p in enumerate(ids) if answer_in_text(ex["answers"]["text"], passages[p])]
        rar.append({"task": "rar", "src": "squad", "prompt": rar_prompt(ex["question"], [passages[p] for p in ids]),
                    "answer": ", ".join(str(j) for j in rel)})

    # HotpotQA train(多段。根拠 2 段落 + distractor 8 段落 = 10 候補)
    hp = _hotpot("train")
    hp_order = rng.permutation(len(hp))[: h_qa + h_rqa + h_rar + 500].tolist()
    k = 0
    for i in hp_order:
        ex = hp[i]
        paras, sup = _hp_paragraphs(ex)
        if len(paras) < 10 or len(sup) < 2:
            continue
        if len(qa) - s_qa < h_qa:
            qa.append({"task": "qa", "src": "hotpot", "prompt": qa_prompt(ex["question"], [paras[j] for j in sup]),
                       "answer": ex["answer"]})
            continue
        perm = rng.permutation(len(paras)).tolist()
        shown = [paras[j] for j in perm]
        if len(rqa) - s_rqa < h_rqa:
            rqa.append({"task": "rqa", "src": "hotpot", "prompt": qa_prompt(ex["question"], shown), "answer": ex["answer"]})
        elif len(rar) - s_rar < h_rar:
            rel = [r + 1 for r, j in enumerate(perm) if j in sup]
            rar.append({"task": "rar", "src": "hotpot", "prompt": rar_prompt(ex["question"], shown),
                        "answer": ", ".join(str(j) for j in rel)})
        else:
            break
    for rows in (qa, rqa, rar):  # 先頭 n 行を取っても SQuAD / HotpotQA が半々になるように混ぜる
        perm = rng.permutation(len(rows)).tolist()
        rows[:] = [rows[j] for j in perm]

    # Context ranking(MS MARCO)。負例は MS MARCO の passage 全体に対する BM25 の上位のうち、正例でないもの
    mm = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True).shuffle(seed=seed, buffer_size=5000)
    items, texts_all, seen = [], [], {}
    for ex in mm:
        texts, sel = ex["passages"]["passage_text"], ex["passages"]["is_selected"]
        pos = [t for t, s in zip(texts, sel) if s == 1]
        if not pos:
            continue
        items.append((ex["query"], pos))
        for t in texts:
            if t not in seen:
                seen[t] = len(texts_all)
                texts_all.append(t)
        if len(items) >= n_rank_queries:
            break
    mbm = BM25(texts_all)
    rank, n_noneg = [], 0
    for q, pos in items:
        pos_set = set(pos)
        neg = next((texts_all[j] for j in mbm.topk(q, 20) if texts_all[j] not in pos_set), None)
        if neg is None:
            n_noneg += 1
            continue
        rank.append({"task": "rank", "prompt": rank_prompt(q, truncate_words(pos[0])), "answer": "True"})
        rank.append({"task": "rank", "prompt": rank_prompt(q, truncate_words(neg)), "answer": "False"})

    for name, rows in (("qa", qa), ("rqa", rqa), ("rar", rar), ("rank", rank)):
        write_jsonl(out_dir / f"{name}.jsonl", rows)
    meta = {"qa": len(qa), "rqa": len(rqa), "rar": len(rar), "rank": len(rank), "squad_passages": len(passages),
            "msmarco_passages": len(texts_all), "msmarco_no_neg": n_noneg}
    (out_dir / "train_meta.json").write_text(json.dumps(meta))
    return meta


def build_rank_variants(out_dir: Path, n_rank_queries=10000, seed=0):
    """context ranking の負例の取り方を変えた 2 種類(診断用)。build_train と同じ MS MARCO の事例列を使う。

      rank_rand.jsonl   負例 = 同じ質問の候補のうち選ばれなかった passage(第 1 版と同じ。ランダム負例)
      rank_bm25x.jsonl  負例 = 全 passage に対する BM25 上位のうち、その質問自身の候補ではない passage
                        (同じ質問の未ラベルの関連 passage が負例に混ざる誤りを避ける)
    """
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    mm = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True).shuffle(seed=seed, buffer_size=5000)
    items, texts_all, seen = [], [], {}
    for ex in mm:
        texts, sel = ex["passages"]["passage_text"], ex["passages"]["is_selected"]
        pos = [t for t, s in zip(texts, sel) if s == 1]
        if not pos:
            continue
        items.append((ex["query"], pos, [t for t, s in zip(texts, sel) if s == 0], set(texts)))
        for t in texts:
            if t not in seen:
                seen[t] = len(texts_all)
                texts_all.append(t)
        if len(items) >= n_rank_queries:
            break
    mbm = BM25(texts_all)
    rand, bm25x = [], []
    for q, pos, own_neg, own in items:
        if own_neg:
            neg = own_neg[int(rng.integers(len(own_neg)))]
            rand += [{"task": "rank", "prompt": rank_prompt(q, truncate_words(pos[0])), "answer": "True"},
                     {"task": "rank", "prompt": rank_prompt(q, truncate_words(neg)), "answer": "False"}]
        nx = next((texts_all[j] for j in mbm.topk(q, 60) if texts_all[j] not in own), None)
        if nx is not None:
            bm25x += [{"task": "rank", "prompt": rank_prompt(q, truncate_words(pos[0])), "answer": "True"},
                      {"task": "rank", "prompt": rank_prompt(q, truncate_words(nx)), "answer": "False"}]
    write_jsonl(out_dir / "rank_rand.jsonl", rand)
    write_jsonl(out_dir / "rank_bm25x.jsonl", bm25x)
    return {"rank_rand": len(rand), "rank_bm25x": len(bm25x)}


def build_eval(out_dir: Path, n_eval=1000, n_hp_corpus=4000, seed=0):
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    dense = Dense()

    # ---- SQuAD: コーパス = 開発セットの全段落 + 学習セットの全段落
    dev = load_dataset("rajpurkar/squad", split="validation")
    tr = load_dataset("rajpurkar/squad", split="train")
    d_pass, d_gold = _squad_passages(dev)
    t_pass, _ = _squad_passages(tr)
    have = set(d_pass)
    passages = d_pass + [p for p in t_pass if p not in have]
    keep = [i for i, ex in enumerate(dev) if _answer_reachable(ex["context"], ex["answers"])]
    pick = sorted(rng.choice(keep, size=min(n_eval, len(keep)), replace=False).tolist())
    pools = _pools(dense, passages, [dev[i]["question"] for i in pick])
    squad = {"passages": passages, "questions": [
        {"q": dev[i]["question"], "answers": sorted(set(dev[i]["answers"]["text"])), "gold": [d_gold[i]],
         "pool": {r: pools[r][k] for r in pools}} for k, i in enumerate(pick)]}
    (out_dir / "eval_squad.json").write_text(json.dumps(squad, ensure_ascii=False))

    # ---- HotpotQA 検証セット(学習に使わない): コーパス = n_hp_corpus 問分の全段落、評価はそのうち n_eval 問
    hp = _hotpot("validation")
    corpus_q = sorted(rng.choice(len(hp), size=min(n_hp_corpus, len(hp)), replace=False).tolist())
    eval_q = set(rng.choice(corpus_q, size=min(n_eval, len(corpus_q)), replace=False).tolist())
    title_id: dict[str, int] = {}
    hp_passages: list[str] = []
    questions = []
    for i in corpus_q:
        ex = hp[i]
        for t, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            if t not in title_id:
                title_id[t] = len(hp_passages)
                hp_passages.append(f"{t}: {truncate_words(' '.join(s.strip() for s in sents))}")
        if i in eval_q:
            sup = sorted({title_id[t] for t in ex["supporting_facts"]["title"] if t in title_id})
            questions.append({"q": ex["question"], "answers": [ex["answer"]], "gold": sup})
    pools = _pools(dense, hp_passages, [q["q"] for q in questions])
    for k, q in enumerate(questions):
        q["pool"] = {r: pools[r][k] for r in pools}
    (out_dir / "eval_hotpot.json").write_text(json.dumps({"passages": hp_passages, "questions": questions}, ensure_ascii=False))
    return {"squad": [len(passages), len(squad["questions"])], "hotpot": [len(hp_passages), len(questions)]}


def prepare(out_dir: str, smoke: bool = False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if smoke:
        meta = build_train(out, n_qa=48, n_rqa=48, n_rar=32, n_rank_queries=60)
        ev = build_eval(out, n_eval=24, n_hp_corpus=100)
    else:
        meta = build_train(out)
        ev = build_eval(out)
    print({"train": meta, "eval": ev})
    return {"train": meta, "eval": ev}
