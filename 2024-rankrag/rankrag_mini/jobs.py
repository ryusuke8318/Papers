"""実験条件の一覧。Modal からもローカルからも同じ設定を使う(torch は読み込まない)。

学習ブレンドの記号(論文 Stage-II の 5 部品のうち、縮小版で使うもの)
  QA   Context-rich QA            RQA  Retrieval-augmented QA
  RAR  Retrieval-augmented ranking     R    Context ranking の行数(MS MARCO)
条件
  base        学習なしの Qwen2.5-Instruct(論文の「Stage-I のみ」に相当)
  chatqa      QA + RQA、ranking データなし(論文の ChatQA-1.5 に相当。推論時に rerank しない)
  rr_r<R>     QA + RQA + RAR + ranking R 行(RankRAG。R=2000 が基準)
  rr_r2000_noRAR / _noRQA   Tab. 3 の ablation
  specialist  ranking データ 20,000 行だけで学習(基準の RankRAG の 10 倍。論文 Tab. 6 の専門 ranker に相当)
"""
from __future__ import annotations

BASE = dict(model="Qwen/Qwen2.5-0.5B-Instruct", train=True, n_qa=4000, n_rqa=4000, n_rar=2000,
            use_qa=True, use_rqa=True, use_rar=True, rank_rows=0, rank_only=False,
            lr=2e-4, batch=32, seed=0, do_gen=True)


def _mk(name: str, **kw) -> dict:
    cfg = {**BASE, **kw}
    cfg["name"] = name
    return cfg


JOBS: dict[str, dict] = {}
for tag, model, lr in (("05b", "Qwen/Qwen2.5-0.5B-Instruct", 2e-4), ("15b", "Qwen/Qwen2.5-1.5B-Instruct", 1e-4)):
    common = dict(model=model, lr=lr)
    JOBS[f"{tag}_base"] = _mk(f"{tag}_base", train=False, **common)
    JOBS[f"{tag}_chatqa"] = _mk(f"{tag}_chatqa", use_rar=False, **common)
    JOBS[f"{tag}_rr_r2000"] = _mk(f"{tag}_rr_r2000", rank_rows=2000, **common)
    JOBS[f"{tag}_specialist"] = _mk(f"{tag}_specialist", rank_only=True, rank_rows=20000, do_gen=False, **common)

# 0.5B だけの追加条件: ranking データ量の掃引(Fig. 4)と ablation(Tab. 3)
_c = dict(model="Qwen/Qwen2.5-0.5B-Instruct", lr=2e-4)
JOBS["05b_rr_r500"] = _mk("05b_rr_r500", rank_rows=500, **_c)
JOBS["05b_rr_r8000"] = _mk("05b_rr_r8000", rank_rows=8000, **_c)
JOBS["05b_rr_r2000_noRAR"] = _mk("05b_rr_r2000_noRAR", rank_rows=2000, use_rar=False, **_c)
JOBS["05b_rr_r2000_noRQA"] = _mk("05b_rr_r2000_noRQA", rank_rows=2000, use_rqa=False, **_c)

# seed を変えた再実行(0.5B の主要 3 条件)
for s in (1, 2):
    for base in ("chatqa", "rr_r2000", "specialist"):
        src = JOBS[f"05b_{base}"]
        JOBS[f"05b_{base}_s{s}"] = {**src, "name": f"05b_{base}_s{s}", "seed": s}

# ---- 第 2 版(data_v2.py のデータ。候補 10 件、BM25 hard negative、広いコーパス、1,000 問で評価)
V2 = dict(data="data_v2", n_eval=1000, max_len=3072)
_c = dict(model="Qwen/Qwen2.5-0.5B-Instruct", lr=2e-4, **V2)
JOBS["v2_base"] = _mk("v2_base", train=False, **_c)
JOBS["v2_chatqa"] = _mk("v2_chatqa", use_rar=False, **_c)
for _r in (500, 1000, 2000, 4000, 8000, 16000):
    JOBS[f"v2_rr_r{_r}"] = _mk(f"v2_rr_r{_r}", rank_rows=_r, **_c)
JOBS["v2_rr_r2000_noRAR"] = _mk("v2_rr_r2000_noRAR", rank_rows=2000, use_rar=False, **_c)
JOBS["v2_rr_r2000_noRQA"] = _mk("v2_rr_r2000_noRQA", rank_rows=2000, use_rqa=False, **_c)
JOBS["v2_specialist"] = _mk("v2_specialist", rank_only=True, rank_rows=20000, do_gen=False, **_c)
for _s in (1, 2):
    for _b in ("chatqa", "rr_r2000", "rr_r8000", "specialist"):
        JOBS[f"v2_{_b}_s{_s}"] = {**JOBS[f"v2_{_b}"], "name": f"v2_{_b}_s{_s}", "seed": _s}
JOBS["v2_smoke_rr"] = _mk("v2_smoke_rr", n_qa=32, n_rqa=32, n_rar=16, rank_rows=32, batch=16, **V2)
JOBS["v2_smoke_base"] = _mk("v2_smoke_base", train=False, **V2)
V2_ORDER = ["v2_base", "v2_chatqa", "v2_rr_r2000", "v2_specialist", "v2_rr_r500", "v2_rr_r1000", "v2_rr_r4000",
            "v2_rr_r8000", "v2_rr_r16000", "v2_rr_r2000_noRAR", "v2_rr_r2000_noRQA"]
V2_SEED_ORDER = [f"v2_{b}_s{s}" for s in (1, 2) for b in ("chatqa", "rr_r2000", "rr_r8000", "specialist")]

# ---- 第 2 版の追加: 総量を合わせた対照、ranking のみの対照、1.5B、seed 違い、AUROC 付きの評価やり直し
JOBS["v2_rr_matched8k"] = _mk("v2_rr_matched8k", n_qa=3000, n_rqa=2000, n_rar=1000, rank_rows=2000, **_c)  # chatqa と総量 8,000 行
JOBS["v2_rankonly_r2000"] = _mk("v2_rankonly_r2000", rank_only=True, rank_rows=2000, do_gen=False, **_c)  # ranking 量を RankRAG と同じに
_c15 = dict(model="Qwen/Qwen2.5-1.5B-Instruct", lr=1e-4, **V2)
JOBS["v2_15b_base"] = _mk("v2_15b_base", train=False, **_c15)
JOBS["v2_15b_chatqa"] = _mk("v2_15b_chatqa", use_rar=False, **_c15)
JOBS["v2_15b_rr_r2000"] = _mk("v2_15b_rr_r2000", rank_rows=2000, **_c15)
JOBS["v2_15b_specialist"] = _mk("v2_15b_specialist", rank_only=True, rank_rows=20000, do_gen=False, **_c15)
for _s in (1, 2):
    for _b in ("rr_matched8k", "rankonly_r2000"):
        JOBS[f"v2_{_b}_s{_s}"] = {**JOBS[f"v2_{_b}"], "name": f"v2_{_b}_s{_s}", "seed": _s}
V2B_ORDER = ["v2_base", "v2_rr_matched8k", "v2_rankonly_r2000", "v2_15b_base", "v2_15b_chatqa", "v2_15b_rr_r2000", "v2_15b_specialist"]
V2B_SEED_ORDER = [f"v2_{b}_s{s}" for s in (1, 2) for b in ("chatqa", "rr_r2000", "specialist", "rr_matched8k", "rankonly_r2000")]
V2_EVAL_AGAIN = ["v2_chatqa", "v2_rr_r500", "v2_rr_r1000", "v2_rr_r2000", "v2_rr_r4000", "v2_rr_r8000", "v2_rr_r16000",
                 "v2_rr_r2000_noRAR", "v2_rr_r2000_noRQA", "v2_specialist"]
for _n in V2_EVAL_AGAIN:  # 学習済みの adapter を使い、AUROC を足して評価だけやり直す
    JOBS[f"{_n}_eval"] = {**JOBS[_n], "name": _n, "eval_only": True}
V2_EVAL_ORDER = [f"{n}_eval" for n in V2_EVAL_AGAIN]

# ---- 診断: context ranking の負例の取り方を変える(rank_rand: ランダム負例、rank_bm25x: 自分の候補を除いた BM25 負例)
for _neg in ("rand", "bm25x"):
    for _r in (2000, 8000):
        JOBS[f"v2_rr_r{_r}_{_neg}"] = _mk(f"v2_rr_r{_r}_{_neg}", rank_rows=_r, rank_file=f"rank_{_neg}.jsonl", **_c)
    JOBS[f"v2_specialist_{_neg}"] = _mk(f"v2_specialist_{_neg}", rank_only=True, rank_rows=20000, rank_file=f"rank_{_neg}.jsonl",
                                        do_gen=False, **_c)
V2C_ORDER = [f"v2_{k}_{n}" for n in ("rand", "bm25x") for k in ("rr_r2000", "rr_r8000", "specialist")]

# 動作確認用(小さな学習と評価)
JOBS["smoke_rr"] = _mk("smoke_rr", n_qa=32, n_rqa=32, n_rar=16, rank_rows=32, batch=16)
JOBS["smoke_specialist"] = _mk("smoke_specialist", rank_only=True, rank_rows=32, batch=16, do_gen=False)
JOBS["smoke_base"] = _mk("smoke_base", train=False)

MAIN_ORDER = ["05b_base", "05b_chatqa", "05b_rr_r2000", "05b_specialist", "05b_rr_r500", "05b_rr_r8000",
              "05b_rr_r2000_noRAR", "05b_rr_r2000_noRQA", "15b_base", "15b_chatqa", "15b_rr_r2000", "15b_specialist"]
SEED_ORDER = [f"05b_{b}_s{s}" for s in (1, 2) for b in ("chatqa", "rr_r2000", "specialist")]
