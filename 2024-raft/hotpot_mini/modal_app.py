"""Modal で RAFT の縮小再現 (HotpotQA + Qwen2.5-0.5B + LoRA)。

  modal run hotpot_mini/modal_app.py --stage prepare   # データ作成 (CPU)
  modal run hotpot_mini/modal_app.py --stage train     # 学習+評価 (A10G x2 並列)

計算の本体は `core.py`。ローカル試運転は `run_local.py` (同コード)。
variants: dsf (Q->A 学習、Q のみで評価), dsf_rag (dsf の重みを RAG で評価),
          raft (Q+文書->引用つき CoT で学習、RAG で評価)
結果は hotpot_mini/results/ に JSON で保存される。GPU 料金は Modal のアカウントにかかる。
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP_NAME = "raft-hotpot-mini"
VOL_NAME = "raft-hotpot-cache"
GPU = "A10G"

from core import EPOCHS  # noqa: E402  (定数のみ。remote 本体は core を参照)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "transformers==4.46.3", "datasets==3.1.0",
        "peft==0.13.2", "accelerate==1.1.1", "numpy",
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "core.py", "/root/core.py")
)


@app.function(image=image, cpu=4, memory=16384, volumes={"/cache": vol}, timeout=3600)
def prepare_data(n_train: int = 3000, n_test: int = 300,
                 n_dist: int = 4, seed: int = 0):
    """HotpotQA distractor から RAFT 式の学習・評価データを作り volume に保存する。"""
    import core

    out = Path("/cache/raft")
    out.mkdir(parents=True, exist_ok=True)
    train = core.build_rows("train", n_train, out, n_dist, seed)
    test = core.build_rows("validation", n_test, out, n_dist, seed)
    core.save_rows(train, out / "train.json")
    core.save_rows(test, out / "test.json")
    vol.commit()
    peek = train[0]
    return {"n_train": len(train), "n_test": len(test),
            "peek_q": peek["question"][:120], "peek_cot": peek["cot"][:200]}


@app.function(image=image, gpu=GPU, cpu=8, memory=32768, volumes={"/cache": vol},
              timeout=2 * 3600)
def train_eval(kind: str = "raft", epochs: int = EPOCHS, seed: int = 0):
    """LoRA 学習して RAG 評価。kind: raft (文書+CoTで学習) / dsf (Q->Aで学習し、dsf と dsf_rag を評価)。"""
    import core

    base = Path("/cache/raft")
    train = core.load_rows(base / "train.json")
    test = core.load_rows(base / "test.json")
    results = core.train_and_eval(kind, train, test, epochs=epochs, seed=seed,
                                  device="cuda")

    out = Path("/cache/raft")
    for tag, res in results.items():
        (out / f"result_{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    vol.commit()
    return results


# ---------------------------------------------------------------- 入口
@app.local_entrypoint()
def main(stage: str = "prepare", epochs: int = EPOCHS):
    if stage in ("prepare", "all"):
        info = prepare_data.remote()
        print(info)
    if stage in ("train", "all"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(train_eval.remote, k, epochs) for k in ("dsf", "raft")]
            merged = {}
            for f in futs:
                merged.update(f.result())
        out_dir = HERE / "results"
        out_dir.mkdir(exist_ok=True)
        for tag, res in merged.items():
            (out_dir / f"hotpot_{tag}_mini.json").write_text(
                json.dumps(res, indent=1, ensure_ascii=False))
            print(f"保存: {out_dir / f'hotpot_{tag}_mini.json'} "
                  f"EM={res['em']:.3f} F1={res['f1']:.3f}")
