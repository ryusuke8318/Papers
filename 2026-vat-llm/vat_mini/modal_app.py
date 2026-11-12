"""Modal で VAT 縮小再現 (合成ドイツ語事例 + Qwen2.5-0.5B + LoRA)。

  modal run vat_mini/modal_app.py --stage prepare   # データ作成 (CPU)
  modal run vat_mini/modal_app.py --stage train     # 学習+評価 (並列は 5 以内)
  modal run vat_mini/modal_app.py --stage pull      # 結果回収

計算の本体は `core.py`。ローカル試運転は `run_local.py` (同コード)。
kinds: plain (学習なし・文脈なし), rag (学習なし+検索), mocked (学習なし+正解 chunk),
       ft (LoRA・chain 除外), rag_ft (LoRA+検索)。
結果は volume と vat_mini/results/ に JSON で保存される。GPU 料金は Modal のアカウントにかかる。

注意: `modal run` のローカルプロセスを kill するとリモートの実行もキャンセルされる。
完了までクライアントを生かしておくこと。
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP_NAME = "vat-mini"
VOL_NAME = "vat-mini-cache"
GPU = "A10G"
MAX_PARALLEL = 5  # 同時コンテナ上限

from core import EPOCHS, MODEL_ID  # noqa: E402  (定数のみ。remote 本体は core を参照)

DATA_VER = "v1"
KINDS_ALL = "plain,rag,mocked,ft,rag_ft"

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
def prepare_data(seed: int = 0):
    """data_seed.json から seed 別の train/test を作り volume に保存する。"""
    import core

    seed_data = core.load_seed(Path("/root/data_seed.json"))
    rng = random.Random(seed)
    train_pool = seed_data["train_pool"][:]
    rng.shuffle(train_pool)
    test_all = seed_data["test_pool"]
    test = test_all[seed * 150:(seed + 1) * 150] or test_all[:150]
    out = Path(f"/cache/vat/{DATA_VER}_seed_{seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.json").write_text(json.dumps(train_pool, ensure_ascii=False))
    (out / "test.json").write_text(json.dumps(test, ensure_ascii=False))
    (out / "rules.json").write_text(json.dumps(seed_data["rules"], ensure_ascii=False))
    vol.commit()
    return {"n_train": len(train_pool), "n_test": len(test)}


@app.function(image=image, gpu=GPU, cpu=8, memory=32768, volumes={"/cache": vol},
              timeout=3 * 3600)
def train_eval(kind: str = "ft", epochs: int = EPOCHS, seed: int = 0,
               dtype: str = "bfloat16", model_id: str = MODEL_ID, tag: str = "05b",
               lr: float = 2e-4):
    """学習 (ft 系のみ) して評価。結果は volume に保存+commit する。"""
    import core

    base = Path(f"/cache/vat/{DATA_VER}_seed_{seed}")
    train = core.load_seed(base / "train.json")
    test = core.load_seed(base / "test.json")
    rules = core.load_seed(base / "rules.json")
    results = core.train_and_eval(kind, train, test, rules, epochs=epochs,
                                  seed=seed, device="cuda", dtype=dtype,
                                  model_id=model_id, lr=lr)
    for k, res in results.items():
        (base / f"final_{k}_{tag}.json").write_text(
            json.dumps(res, indent=1, ensure_ascii=False))
    vol.commit()
    return results


@app.function(image=image, cpu=1, memory=1024, volumes={"/cache": vol},
              timeout=600)
def dump_finals():
    """volume 上の final_*.json を返す。回収用。"""
    base = Path("/cache/vat")
    out = {}
    for p in sorted(base.glob(f"{DATA_VER}_seed_*/final_*.json")):
        out[f"{p.parent.name}/{p.name}"] = json.loads(p.read_text())
    return out


@app.function(image=image, cpu=2, memory=4096, volumes={"/cache": vol},
              timeout=8 * 3600)
def sweep_all(epochs: int = EPOCHS, seeds: str = "0", kinds: str = KINDS_ALL,
              dtype: str = "bfloat16", model_id: str = MODEL_ID, tag: str = "05b",
              lr: float = 2e-4):
    """種・条件をサーバ側で逐次実行する。結果は volume に保存+commit するので、
    起動後にローカルを落としても失われない (逐次=同時 GPU 1 本)。"""
    seed_list = [int(s) for s in seeds.split(",") if s.strip() != ""]
    kind_list = [k.strip() for k in kinds.split(",") if k.strip()]
    summary = {}
    for s in seed_list:
        for k in kind_list:
            res = train_eval.remote(k, epochs, s, dtype, model_id, tag, lr)[k]
            summary[f"seed{s}/{k}"] = {"acc": round(res["acc"], 3),
                                       "strict": round(res["acc_strict"], 3)}
    return summary


# ---------------------------------------------------------------- 入口
@app.local_entrypoint()
def main(stage: str = "prepare", epochs: int = EPOCHS, kinds: str = KINDS_ALL,
         seeds: str = "0", dtype: str = "bfloat16",
         model_id: str = MODEL_ID, tag: str = "05b", lr: float = 2e-4):
    seed_list = [int(s) for s in seeds.split(",") if s.strip() != ""]
    kind_list = [k.strip() for k in kinds.split(",") if k.strip()]
    if stage in ("prepare", "all"):
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(seed_list))) as pool:
            futs = [pool.submit(prepare_data.remote, s) for s in seed_list]
            for f in futs:
                print(f.result())
    if stage in ("train", "all"):
        pairs = [(s, k) for s in seed_list for k in kind_list]
        out_dir = HERE / "results"
        out_dir.mkdir(exist_ok=True)
        for i in range(0, len(pairs), MAX_PARALLEL):
            chunk = pairs[i:i + MAX_PARALLEL]
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
                futs = [pool.submit(train_eval.remote, k, epochs, s, dtype,
                                    model_id, tag, lr)
                        for s, k in chunk]
                for (s, k), f in zip(chunk, futs):
                    res = f.result()[k]
                    (out_dir / f"vat_{k}_seed{s}_{tag}_mini.json").write_text(
                        json.dumps(res, indent=1, ensure_ascii=False))
                    print(f"保存: vat_{k}_seed{s}_{tag}_mini.json acc={res['acc']:.3f} "
                          f"hit={res['hit_rate']}")
    if stage == "pull":
        out_dir = HERE / "results"
        out_dir.mkdir(exist_ok=True)
        for key, res in dump_finals.remote().items():
            parent, name = key.split("/")
            s = parent.split("_seed_")[1]
            k, tag = name[len("final_"):-len(".json")].rsplit("_", 1)
            (out_dir / f"vat_{k}_seed{s}_{tag}_mini.json").write_text(
                json.dumps(res, indent=1, ensure_ascii=False))
            print(f"回収: vat_{k}_seed{s}_{tag}_mini.json acc={res['acc']:.3f}")
