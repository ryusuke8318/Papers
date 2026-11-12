"""Modal で RankRAG の縮小再現 (HotpotQA + Qwen2.5-0.5B + LoRA)。

  modal run gold_context_mini/modal_app.py --stage prepare   # データ作成 (CPU)
  modal run gold_context_mini/modal_app.py --stage train     # 学習+評価 (複数条件を並列)

計算の本体は `core.py`。ローカル試運転は `run_local.py` (同コード)。
kinds: qa_only (生成のみ学習), rank_small (生成+ranking 400件), rank_full (生成+ranking 4000件),
       specialist (ranking のみ 8000件)。
結果は gold_context_mini/results/ に JSON で保存される。GPU 料金は Modal のアカウントにかかる。

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
               dtype: str = "bfloat16", model_id: str = MODEL_ID, tag: str = "05b",
               lr: float = 2e-4):
    """LoRA 学習して評価。kind: qa_only / rank_small / rank_full。"""
    import core

    base = Path(f"/cache/rankrag/{DATA_VER}_seed_{seed}")
    train = core.load_rows(base / "train.json")
    rank_pool = core.load_rows(base / "rank_pool.json")
    test = core.load_rows(base / "test.json")
    results = core.train_and_eval(kind, train, rank_pool, test, epochs=epochs,
                                  seed=seed, device="cuda", dtype=dtype,
                                  model_id=model_id, lr=lr)

    out = base
    for k, res in results.items():
        (out / f"result_{k}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        (out / f"result_{k}_{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    vol.commit()
    return results


# 電源断でも完結する全量 sweep (orchestrator 1 + GPU 5 以内)。
FULL_SWEEP = [
    {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "tag": "15b",
     "kinds": ["rank_small", "mix_half", "rankonly4k", "specialist", "mix_matched10"]},
    {"model_id": "Qwen/Qwen2.5-3B-Instruct", "tag": "30b",
     "kinds": ["qa_only", "rank_full"]},
]


@app.function(image=image, cpu=2, memory=4096, volumes={"/cache": vol},
              timeout=8 * 3600)
def sweep_all(epochs: int = EPOCHS, seeds: str = "0,1,2",
              dtype: str = "bfloat16"):
    """全 sweep をサーバ側で完結させる。結果は volume に保存+commit するので、
    `modal run --detach` 後にローカルを落としても失われない。"""
    import core
    from concurrent.futures import ThreadPoolExecutor

    seed_list = [int(s) for s in seeds.split(",") if s.strip() != ""]
    summary = {}
    for spec in FULL_SWEEP:
        pairs = [(s, k) for s in seed_list for k in spec["kinds"]]
        with ThreadPoolExecutor(max_workers=5) as pool:
            futs = [pool.submit(train_eval.remote, k, epochs, s, dtype,
                                spec["model_id"], spec["tag"])
                    for s, k in pairs]
            for (s, k), f in zip(pairs, futs):
                res = f.result()[k]
                base = Path(f"/cache/rankrag/{DATA_VER}_seed_{s}")
                (base / f"final_{k}_{spec['tag']}.json").write_text(
                    json.dumps(res, indent=1, ensure_ascii=False))
                summary[f"{spec['tag']}/seed{s}/{k}"] = {
                    "norerank_em": round(res["norerank"]["em"], 3),
                    "reranked_em": round(res["reranked"]["em"], 3),
                    "auroc": round(res["reranked"].get("auroc", float("nan")), 3)}
        vol.commit()  # フェーズごとに commit (後段が失敗しても前段は残る)
    return summary


@app.function(image=image, cpu=2, memory=4096, volumes={"/cache": vol},
              timeout=8 * 3600)
def sweep_30b_lr(lr: float = 1e-5, tag: str = "30b15", seeds: str = "0,1,2",
                 epochs: int = EPOCHS, dtype: str = "bfloat16"):
    """30b の qa_only/rank_full を低 lr で走らせ直す (2 並列=同時 GPU 2 本)。
    結果は volume に final_{kind}_{tag}.json で保存+commit するので、
    起動後にローカルを落としても失われない。"""
    import core
    from concurrent.futures import ThreadPoolExecutor

    seed_list = [int(s) for s in seeds.split(",") if s.strip() != ""]
    pairs = [(s, k) for s in seed_list for k in ("qa_only", "rank_full")]
    summary = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(train_eval.remote, k, epochs, s, dtype,
                            "Qwen/Qwen2.5-3B-Instruct", tag, lr)
                for s, k in pairs]
        for (s, k), f in zip(pairs, futs):
            res = f.result()[k]
            base = Path(f"/cache/rankrag/{DATA_VER}_seed_{s}")
            (base / f"final_{k}_{tag}.json").write_text(
                json.dumps(res, indent=1, ensure_ascii=False))
            vol.commit()
            summary[f"seed{s}/{k}"] = {
                "norerank_em": round(res["norerank"]["em"], 3),
                "reranked_em": round(res["reranked"]["em"], 3),
                "auroc": round(res["reranked"].get("auroc", float("nan")), 3)}
    return summary


@app.function(image=image, cpu=1, memory=1024, volumes={"/cache": vol},
              timeout=600)
def dump_finals():
    """volume 上の final_*.json を返す。次セッションでの回収用。"""
    base = Path("/cache/rankrag")
    out = {}
    for p in sorted(base.glob(f"{DATA_VER}_seed_*/final_*.json")):
        out[f"{p.parent.name}/{p.name}"] = json.loads(p.read_text())
    return out


@app.function(image=image, cpu=1, memory=1024, volumes={"/cache": vol},
              timeout=600)
def dump_glob(pattern: str = "result_qa_only_diag*.json"):
    """volume 上の pattern に合う JSON を返す。診断 run の回収・ポーリング用。"""
    base = Path("/cache/rankrag")
    out = {}
    for p in sorted(base.glob(f"{DATA_VER}_seed_*/{pattern}")):
        out[f"{p.parent.name}/{p.name}"] = json.loads(p.read_text())
    return out


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
        for i in range(0, len(pairs), 5):
            chunk = pairs[i:i + 5]
            with ThreadPoolExecutor(max_workers=5) as pool:
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
    if stage == "sweep":
        for k in sorted(summary := sweep_all.remote(epochs, seeds, dtype)):
            print(k, summary[k])
    if stage == "pull":
        out_dir = HERE / "results"
        out_dir.mkdir(exist_ok=True)
        for key, res in dump_finals.remote().items():
            parent, name = key.split("/")
            s = parent.split("_seed_")[1]
            k, tag = name[len("final_"):-len(".json")].rsplit("_", 1)
            (out_dir / f"rankrag_{k}_seed{s}_{tag}_mini.json").write_text(
                json.dumps(res, indent=1, ensure_ascii=False))
            print(f"回収: rankrag_{k}_seed{s}_{tag}_mini.json "
                  f"norerank EM={res['norerank']['em']:.3f} "
                  f"reranked EM={res['reranked']['em']:.3f}")
