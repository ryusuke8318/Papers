"""Modal で内外判定の検証を回す(GPU コンテナは同時に最大 4 つ。API の評価は CPU)。

  modal run jp_vat/modal_app.py --stage api-smoke                 # API を 3 件だけ呼んで動作と Secret を確認
  modal run jp_vat/modal_app.py --stage api                       # Claude(haiku, sonnet)で plain / rag / mocked / long
  modal run jp_vat/modal_app.py --stage smoke                     # GPU の動作確認(小さな学習と評価)
  modal run jp_vat/modal_app.py --stage run --jobs main           # Qwen 1.5B / 3B: 学習なし 2 条件 + LoRA 3 seed
  modal run jp_vat/modal_app.py --stage run --jobs q7_base,q7_ft_s0

結果は jp_vat/results/<名前>.json。API キーは Modal の Secret `anthropic-secret` を使い、値はコードにもログにも出さない。
`modal run` のローカルプロセスを kill するとリモートの実行もキャンセルされるので、完了まで生かしておくこと。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/root/jp_vat")
from jobs import API_CONDS, API_MODELS, CTX_ORDER, JOBS, MAIN_ORDER  # noqa: E402

MAX_GPU_CONTAINERS = 4
IGNORE = ["__pycache__", "results", "*.pyc"]
app = modal.App("jp-vat")
vol = modal.Volume.from_name("jp-vat-cache", create_if_missing=True)
gpu_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("torch==2.5.1", "transformers==4.46.3", "peft==0.13.2", "accelerate==1.1.1", "numpy")
             .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
             .add_local_dir(HERE, "/root/jp_vat", ignore=IGNORE))
api_image = (modal.Image.debian_slim(python_version="3.11").pip_install("anthropic")
             .add_local_dir(HERE, "/root/jp_vat", ignore=IGNORE))


@app.function(image=gpu_image, gpu="A10G", cpu=4, memory=32768, volumes={"/cache": vol}, timeout=6 * 3600,
              max_containers=MAX_GPU_CONTAINERS)
def gpu_job(cfg: dict):
    sys.path.insert(0, "/root/jp_vat")
    import jp_model

    vol.reload()
    out = jp_model.run_job(cfg, "/root/jp_vat", "/cache/work", log=print)
    try:
        vol.commit()  # adapter は結果の再現には不要。HF のログが開いていると失敗するので、結果は捨てない
    except Exception as e:  # noqa: BLE001
        print(f"commit 失敗(無視): {str(e)[:120]}")
    return out


@app.function(image=api_image, secrets=[modal.Secret.from_name("anthropic-secret")], cpu=2, timeout=4 * 3600)
def api_job(model: str, conds: list, limit: int | None = None):
    sys.path.insert(0, "/root/jp_vat")
    import jp_api

    try:
        return jp_api.run(model, conds, "/root/jp_vat", limit=limit, log=print)
    except Exception as e:  # noqa: BLE001  手元に anthropic が無くても原因が読めるよう、文字列にして返す
        return {"error": f"{type(e).__name__}: {str(e)[:500]}", "cfg": {"name": f"api_{model}", "model": model}}


@app.function(image=gpu_image, gpu="A10G", cpu=4, memory=32768, volumes={"/cache": vol}, timeout=2 * 3600, max_containers=3)
def embft_job(seed: int):
    sys.path.insert(0, "/root/jp_vat")
    import jp_embft

    return jp_embft.run("/root/jp_vat", seed=seed, log=print)


@app.function(image=api_image, secrets=[modal.Secret.from_name("anthropic-secret")], cpu=2, timeout=2 * 3600)
def retr_llm_job(task: str, model: str, cands: dict | None = None):
    sys.path.insert(0, "/root/jp_vat")
    import jp_retr_llm

    if task == "rewrite":
        return jp_retr_llm.rewrite(model, "/root/jp_vat")
    return jp_retr_llm.rerank(model, "/root/jp_vat", cands)


@app.local_entrypoint()
def main(stage: str = "run", jobs: str = "", models: str = "", conds: str = ""):
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    if stage == "api-smoke":
        r = api_job.remote(models.split(",")[0] if models else API_MODELS[0], ["plain", "rag"], 3)
        if "error" in r:
            raise SystemExit("失敗: " + r["error"])
        print(json.dumps({k: v for k, v in r.items() if k != "eval"}, ensure_ascii=False), "tokens:", r["tokens"])
        for cond, sp in r["eval"].items():
            e = sp["test_seen"]["examples"][0]
            print(f"--- {cond} 例 ---\n{e['out']}\n(正解: {'国内' if e['label'] else '国外'})")
        return
    if stage == "embft":
        for res in embft_job.map([0, 1, 2], order_outputs=False, return_exceptions=True):
            if isinstance(res, Exception):
                print("失敗:", repr(res)[:300])
                continue
            name = res["cfg"]["name"]
            (out_dir / f"ft_scores_{name}.json").write_text(json.dumps(res))
            print("保存:", out_dir / f"ft_scores_{name}.json", res["sec"], "秒")
        return
    if stage in ("rewrite", "rerank"):  # 検索の LLM 補助(結果は results/ に保存)
        model = models or "claude-haiku-5-5"
        cands = json.loads((out_dir / "rerank_in.json").read_text()) if stage == "rerank" else None
        res = retr_llm_job.remote(stage, model, cands)
        (out_dir / f"{stage}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print("保存:", out_dir / f"{stage}.json", len(res), "件")
        return
    if stage == "api":
        ms = [m for m in models.split(",") if m] or API_MODELS
        cs = [c for c in conds.split(",") if c] or API_CONDS
        suffix = "_" + "_".join(cs) if conds else ""
        for res in api_job.starmap([(m, cs) for m in ms], return_exceptions=True):
            if isinstance(res, Exception) or "error" in res:
                print("失敗:", repr(res)[:300] if isinstance(res, Exception) else res["error"])
                continue
            name = res["cfg"]["name"] + suffix
            (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
            print("保存:", out_dir / f"{name}.json", "tokens:", res["tokens"])
        return
    names = ["smoke_base", "smoke_ft"] if stage == "smoke" else MAIN_ORDER if jobs == "main" else CTX_ORDER if jobs == "ctx" else [j for j in jobs.split(",") if j]
    bad = [n for n in names if n not in JOBS]
    if bad:
        raise SystemExit(f"不明な条件: {bad}")
    print(f"{len(names)} 条件を実行(GPU コンテナは同時に最大 {MAX_GPU_CONTAINERS}): {names}")
    for res in gpu_job.map([JOBS[n] for n in names], order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print("失敗:", repr(res)[:300])
            continue
        name = res["cfg"]["name"]
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print("保存:", out_dir / f"{name}.json")
