"""Modal で text JEPA の整合度実験を回す(複数の run を別々の GPU で並列実行したいとき用)。

計算の本体は run.py / extract_features.py / baselines.py にあり、ローカルでも同じコードが動く。
ここは、それらを Modal の GPU と Volume(特徴量のキャッシュ)につなぐだけの薄いラッパー。

  # --- Flickr30k (Table 1 の縮小版) ---
  modal run flickr_table1/modal_app.py --stage features
  modal run flickr_table1/modal_app.py --stage all --seed 0 --epochs 200
  modal run flickr_table1/modal_app.py --stage diagnose            # 未学習モデルの診断

  # --- COCO (縮小スケーリング実験) ---
  modal run flickr_table1/modal_app.py --stage coco_features       # 画像を取得して DINOv2 特徴量を保存(1 回だけ)
  modal run flickr_table1/modal_app.py --stage scaling --seed 0    # good/bad/ntp × {0.6M, 1.7M, 5.8M} サンプル
  modal run flickr_table1/modal_app.py --stage lm                  # 小型 LM(Qwen2.5)の整合度
  modal run flickr_table1/modal_app.py --stage train --dataset coco --variant good --samples 100000 --seed 1

variant: good | bad | ntp | ntp_pairs | random
結果は flickr_table1/results/ に JSON で保存される。GPU 料金は Modal のアカウントにかかる。
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
FLICKR_TRAIN = 29000  # 1 epoch = 29,000 画像
GPU = "A10G"
SCALING_SAMPLES = [600_000, 1_700_000, 5_800_000]  # Plan A: 見るサンプル数
LM_MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]
LOCAL_MODULES = ["common", "metrics", "jepa_text", "extract_features", "run", "baselines"]

app = modal.App("platonic-flickr-table1")
vol = modal.Volume.from_name("platonic-flickr-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "datasets==3.1.0", "pillow", "numpy", "requests")
    .env({"PLATONIC_DATA": "/cache", "HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
)
for _m in LOCAL_MODULES:
    image = image.add_local_file(HERE / f"{_m}.py", f"/root/{_m}.py")


@app.function(image=image, gpu=GPU, cpu=8, memory=32768, volumes={"/cache": vol}, timeout=3 * 3600)
def extract_features(limit: int = 0):
    import extract_features as ef

    return ef.extract_flickr(limit, batch=128, workers=8, on_done=vol.commit)


@app.function(image=image, gpu=GPU, cpu=16, memory=32768, volumes={"/cache": vol}, timeout=5 * 3600)
def extract_coco_features(limit: int = 0):
    import extract_features as ef

    return ef.extract_coco(limit, batch=64, workers=8, on_done=vol.commit)  # チャンクごとに commit し、中断しても再開できる


@app.function(image=image, gpu=GPU, volumes={"/cache": vol}, timeout=8 * 3600)
def train_eval(variant: str, seed: int, samples: int, dataset: str = "flickr", n_evals: int = 10):
    import run

    vol.reload()
    return run.train_one(dataset, variant, seed, samples, n_evals)


@app.function(image=image, gpu=GPU, memory=65536, volumes={"/cache": vol}, timeout=3 * 3600)
def lm_baseline(model_name: str, n_train: int = 20000):
    import baselines

    vol.reload()
    return baselines.lm_baseline(model_name, n_train)


@app.function(image=image, gpu=GPU, volumes={"/cache": vol}, timeout=3600)
def diagnose_random():
    import baselines

    vol.reload()
    return baselines.diagnose_random()


# ================================================================ 入口
@app.local_entrypoint()
def main(stage: str = "train", variant: str = "good", seed: int = 0, epochs: int = 2, samples: int = 0,
         dataset: str = "flickr", limit: int = 0, n_evals: int = 10):
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)

    def save(name, obj):
        (out_dir / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False))
        print(f"保存: {out_dir / name}")

    if stage == "features":
        print(extract_features.remote(limit))
    elif stage == "coco_features":
        print(extract_coco_features.remote(limit))
    elif stage == "diagnose":
        save("diagnose_random.json", diagnose_random.remote())
    elif stage == "lm":
        calls = {m: lm_baseline.spawn(m) for m in LM_MODELS}
        save("coco_lm_baselines.json", {m: c.get() for m, c in calls.items()})
    elif stage in ("all", "scaling", "train"):
        if stage == "all":  # Flickr: 5 variant × epochs
            jobs = [(v, epochs * FLICKR_TRAIN, "flickr") for v in ["random", "good", "bad", "ntp", "ntp_pairs"]]
        elif stage == "scaling":  # COCO: good/bad/ntp × サンプル数 + random
            jobs = [("random", SCALING_SAMPLES[0], "coco")] + [
                (v, s, "coco") for v in ("good", "bad", "ntp") for s in SCALING_SAMPLES
            ]
        else:
            jobs = [(variant, samples or epochs * FLICKR_TRAIN, dataset)]
        calls = [(j, train_eval.spawn(j[0], seed, j[1], j[2], n_evals)) for j in jobs]  # 別々の GPU で並列実行
        for (v, s, d), c in calls:
            save(f"{d}_{v}_seed{seed}_s{s}.json", c.get())
    else:
        raise SystemExit(f"不明な stage: {stage}")
