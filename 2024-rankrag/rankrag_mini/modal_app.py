"""Modal で RankRAG の縮小再現 (HotpotQA + Qwen2.5-0.5B + LoRA)。

  modal run rankrag_mini/modal_app.py --stage prepare   # データ作成 (CPU)
  modal run rankrag_mini/modal_app.py --stage train     # 学習+評価 (複数条件を並列)

計算の本体は `core.py`。ローカル試運転は `run_local.py` (同コード)。
kinds: qa_only (生成のみ学習), rank_small (生成+ranking 400件), rank_full (生成+ranking 4000件),
       specialist (ranking のみ 8000件)。
結果は rankrag_mini/results/ に JSON で保存される。GPU 料金は Modal のアカウントにかかる。

注意: `modal run` のローカルプロセスを kill するとリモートの実行もキャンセルされる。
完了までクライアントを生かしておくこと (フォアグラウンド実行か、kill されないバックグラウンド実行)。
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP_NAME = "rankrag-mini"
VOL_NAME = "rankrag-mini-cache"
GPU = "A10G"

from core import EPOCHS, MODEL_ID  # noqa: E402  (定数のみ。remote 本体は core を参照)

DATA_VER = "v2"  # データセット版。test 250 件 (v1 は 150 件)。seed 別 dir に保存。

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
    .add_local_file(HERE / "data_seed.json", "/root/data_seed.json")
)


@app.function(image=image, cpu=4, memory=16384, volumes={"/cache": vol}, timeout=3600)
def prepare_data(n_train: int = 4000, n_test: int = 150, seed: int = 0):
    """HotpotQA distractor から学習・評価データを作り volume に保存する。"""
    import core

    out = Path(f"/cache/rankrag/{DATA_VER}_seed_{seed}")
    out.mkdir(parents=True, exist_ok=True)
    seed_data = core.load_seed(Path("/root/data_seed.json"))
    train = core.build_rows("train", n_train, seed_data, seed)
    test = core.build_rows("validation", n_test, seed_data, seed)
    rank_pool = core.build_rank_rows(train, 10 * n_train, seed)
    core.save_rows(train, out / "train.json")
    core.save_rows(test, out / "test.json")
    core.save_rows(rank_pool, out / "rank_pool.json")
    vol.commit()
    return {"n_train": len(train), "n_rank_pool": len(rank_pool), "n_test": len(test),
            "peek_q": train[0]["question"][:120]}


@app.function(image=image, gpu=GPU, cpu=8, memory=32768, volumes={"/cache": vol},
              timeout=3 * 3600)
def train_eval(kind: str = "rank_full", epochs: int = EPOCHS, seed: int = 0,
               dtype: str = "bfloat16", model_id: str = MODEL_ID, tag: str = "05b"):
    """LoRA 学習して評価。kind: qa_only / rank_small / rank_full。"""
    import core

    base = Path(f"/cache/rankrag/{DATA_VER}_seed_{seed}")
    train = core.load_rows(base / "train.json")
    rank_pool = core.load_rows(base / "rank_pool.json")
    test = core.load_rows(base / "test.json")
    results = core.train_and_eval(kind, train, rank_pool, test, epochs=epochs,
                                  seed=seed, device="cuda", dtype=dtype,
                                  model_id=model_id)

    out = base
    for k, res in results.items():
        (out / f"result_{k}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    vol.commit()
    return results


# ---------------------------------------------------------------- 入口
@app.local_entrypoint()
def main(stage: str = "prepare", epochs: int = EPOCHS,
         kinds: str = "qa_only,rank_small,rank_full,specialist,rankonly4k,mix_matched10,mix_half",
         seeds: str = "0,1,2", dtype: str = "bfloat16",
         model_id: str = MODEL_ID, tag: str = "05b", n_test: int = 250):
    seed_list = [int(s) for s in seeds.split(",") if s.strip() != ""]
    kind_list = [k.strip() for k in kinds.split(",") if k.strip()]
    if stage in ("prepare", "all"):
        with ThreadPoolExecutor(max_workers=len(seed_list)) as pool:
            futs = [pool.submit(prepare_data.remote, 4000, n_test, s)
                    for s in seed_list]
            for f in futs:
                print(f.result())
    if stage in ("train", "all"):
        pairs = [(s, k) for s in seed_list for k in kind_list]
        out_dir = HERE / "results"
        out_dir.mkdir(exist_ok=True)
        for i in range(0, len(pairs), 6):
            chunk = pairs[i:i + 6]
            with ThreadPoolExecutor(max_workers=6) as pool:
                futs = [pool.submit(train_eval.remote, k, epochs, s, dtype,
                                    model_id, tag)
                        for s, k in chunk]
                for (s, k), f in zip(chunk, futs):
                    res = f.result()[k]
                    (out_dir / f"rankrag_{k}_seed{s}_{tag}_mini.json").write_text(
                        json.dumps(res, indent=1, ensure_ascii=False))
                    print(f"保存: rankrag_{k}_seed{s}_{tag}_mini.json "
                          f"norerank EM={res['norerank']['em']:.3f} "
                          f"reranked EM={res['reranked']['em']:.3f} "
                          f"AUROC={res['reranked'].get('auroc', float('nan')):.3f}")
