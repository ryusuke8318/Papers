"""評価: 貪欲(温度 0)で rollout し、EM と証拠の再現率を質問ごとに保存する。"""
from __future__ import annotations

import argparse
import json
import time

import common
from common import RESULTS, WORK, Peak, cap_memory, load_model
from david_core import exact_match, grounded_reward
from rollout import State, advance, prompt_ids

DATASETS = ["hotpot", "2wiki", "musique"]


def run_eval(model, tok, retr, name: str, n: int = 300, bs: int = 100, peak: Peak | None = None, max_searches=3) -> dict:
    model.eval()
    out = {}
    for ds in DATASETS:
        qs = json.loads((WORK / f"dev_{ds}.json").read_text())[:n]
        t0 = time.time()
        rows = []
        for i in range(0, len(qs), bs):
            chunk = qs[i:i + bs]
            states = [State(i + j, prompt_ids(tok, q["question"], max_searches)) for j, q in enumerate(chunk)]
            if peak:
                peak.start()
            advance(model, tok, retr, states, greedy=True, bs=bs, max_searches=max_searches)
            if peak:
                peak.stop("eval_generation")
            for q, s in zip(chunk, states):
                rows.append({"id": q["id"], "pred": s.traj.answer, "gold": q["answer"],
                             "em": exact_match(s.traj.answer, q["answers"]),
                             "recall": grounded_reward(s.traj, q["gold_titles"]),
                             "n_search": s.traj.n_searches, "answered": s.traj.answer is not None})
        out[ds] = rows
        d = RESULTS / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"eval_{ds}.json").write_text(json.dumps(rows, ensure_ascii=False))
        em = sum(r["em"] for r in rows) / len(rows)
        print(f"[eval {name}] {ds} n={len(rows)} EM={em:.3f} time={time.time() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA のディレクトリ(なしなら未学習のモデル)")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    cap_memory()
    from retriever import Retriever
    model, tok = load_model(a.adapter)
    retr = Retriever()
    pk = Peak()
    run_eval(model, tok, retr, a.name, a.n, peak=pk)
    (RESULTS / a.name / "peak_mem.json").write_text(json.dumps(pk.max))
