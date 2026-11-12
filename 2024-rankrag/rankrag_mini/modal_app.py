"""Modal で縮小再現を回す(GPU コンテナは同時に最大 4 つ)。

  modal run rankrag_mini/modal_app.py --stage prepare                    # 学習ブレンドと評価プールの作成(CPU、1 回だけ)
  modal run rankrag_mini/modal_app.py --stage smoke                      # 小さな学習と評価で全体の動作確認
  modal run rankrag_mini/modal_app.py --stage run --jobs 05b_base,05b_chatqa,05b_rr_r2000,05b_specialist
  modal run rankrag_mini/modal_app.py --stage run --jobs main            # 主要 12 条件
  modal run rankrag_mini/modal_app.py --stage run --jobs seeds           # seed 違いの 6 条件
  modal run rankrag_mini/modal_app.py --stage prepare --v2 / --stage smoke --v2 / --stage run --jobs v2 / v2seeds / v2b / v2bseeds / v2eval   # 第 2 版
  modal run rankrag_mini/modal_app.py --stage prepare-rankvar; --stage run --jobs v2c         # 負例の診断

条件の定義は jobs.py。結果は rankrag_mini/results/<条件名>.json に保存される。GPU 料金は Modal のアカウントにかかる。
`modal run` のローカルプロセスを kill するとリモートの実行もキャンセルされるので、完了まで生かしておくこと。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/root/pkg/rankrag_mini")  # コンテナ内では add_local_dir でここに置かれる
from jobs import JOBS, MAIN_ORDER, SEED_ORDER, V2_ORDER, V2_SEED_ORDER, V2B_ORDER, V2B_SEED_ORDER, V2_EVAL_ORDER, V2C_ORDER  # noqa: E402

MAX_GPU_CONTAINERS = 4

app = modal.App("rankrag-reimpl")
vol = modal.Volume.from_name("rankrag-reimpl-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "peft==0.13.2", "accelerate==1.1.1", "datasets==3.1.0",
                 "numpy", "scipy")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(HERE.parent / "src", "/root/pkg/src", ignore=["__pycache__"])
    .add_local_dir(HERE, "/root/pkg/rankrag_mini", ignore=["__pycache__", "results", "data", "work"])
)


@app.function(image=image, cpu=8, memory=32768, volumes={"/cache": vol}, timeout=3 * 3600)
def prepare(smoke: bool = False, v2: bool = False):
    sys.path.insert(0, "/root/pkg/rankrag_mini")
    if v2:
        import data_v2 as data
    else:
        import data

    name = "data_v2" if v2 else "data"
    r = data.prepare(f"/cache/{name}_smoke" if smoke else f"/cache/{name}", smoke)
    vol.commit()
    return r


@app.function(image=image, cpu=8, memory=32768, volumes={"/cache": vol}, timeout=3 * 3600)
def prepare_rank_variants():
    sys.path.insert(0, "/root/pkg/rankrag_mini")
    import data_v2
    from pathlib import Path

    r = data_v2.build_rank_variants(Path("/cache/data_v2"))
    vol.commit()
    return r


@app.function(image=image, gpu="A10G", cpu=4, memory=32768, volumes={"/cache": vol}, timeout=8 * 3600,
              max_containers=MAX_GPU_CONTAINERS)
def gpu_job(cfg: dict, smoke: bool = False):
    sys.path.insert(0, "/root/pkg/rankrag_mini")
    import rr_model

    vol.reload()
    d = cfg.get("data", "data")
    out = rr_model.run_job(cfg, f"/cache/{d}_smoke" if smoke else f"/cache/{d}", "/cache/work",
                           n_eval=16 if smoke else cfg.get("n_eval", 500), log=print)
    vol.commit()
    return out


@app.local_entrypoint()
def main(stage: str = "run", jobs: str = "", smoke: bool = False, v2: bool = False):
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    if stage == "prepare-rankvar":
        print(prepare_rank_variants.remote())
        return
    if stage == "prepare":
        print(prepare.remote(smoke, v2))
        return
    if stage == "smoke":
        print(prepare.remote(True, v2))
        names, smoke = (["v2_smoke_base", "v2_smoke_rr"] if v2 else ["smoke_base", "smoke_rr", "smoke_specialist"]), True
    elif stage == "run":
        names = (MAIN_ORDER if jobs == "main" else SEED_ORDER if jobs == "seeds" else V2_ORDER if jobs == "v2"
                 else V2_SEED_ORDER if jobs == "v2seeds"
                 else V2B_ORDER if jobs == "v2b" else V2B_SEED_ORDER if jobs == "v2bseeds" else V2_EVAL_ORDER if jobs == "v2eval"
                 else V2C_ORDER if jobs == "v2c" else [j for j in jobs.split(",") if j])
    else:
        raise SystemExit(f"不明な stage: {stage}")
    bad = [n for n in names if n not in JOBS]
    if bad:
        raise SystemExit(f"不明な条件: {bad}")
    print(f"{len(names)} 条件を実行(GPU コンテナは同時に最大 {MAX_GPU_CONTAINERS}): {names}")
    for res in gpu_job.map([JOBS[n] for n in names], kwargs={"smoke": smoke}, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print("失敗:", repr(res)[:300])
            continue
        name = res["cfg"]["name"]
        old = out_dir / f"{name}.json"
        if res["cfg"].get("eval_only") and old.exists():  # 学習の記録は前回の結果から引き継ぐ
            prev = json.loads(old.read_text())
            res["train"] = prev.get("train")
            res["cfg"] = prev["cfg"]
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print("保存:", out_dir / f"{name}.json")
