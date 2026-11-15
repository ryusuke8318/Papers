"""Modal で縮小再現を回す(コンテナは同時に最大 3 つ。GPU は VRAM 40〜80GB のもの)。

  modal run --profile modal-lab efficientrag_mini/modal_app.py --stage prepare --datasets hotpotqa,2wiki,musique
  modal run --profile modal-lab efficientrag_mini/modal_app.py --stage synth   --datasets hotpotqa
  modal run --profile modal-lab efficientrag_mini/modal_app.py --stage fit     --datasets hotpotqa --sizes 250,1000,0   # 0 = 全部
  modal run --profile modal-lab efficientrag_mini/modal_app.py --stage eval    --datasets hotpotqa
  modal run --profile modal-lab efficientrag_mini/modal_app.py --stage smoke                                            # 少数の質問で全体の動作確認

結果は efficientrag_mini/results/ に保存する。`modal run` のローカルプロセスは完了まで生かしておくこと。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/root/pkg/efficientrag_mini")
from config import DATASETS, INFER, N_EVAL, N_TRAIN, CORPUS_QUESTIONS, SYNTH_LLM  # noqa: E402

NO_SECRET = bool(os.environ.get("ER_NO_SECRET"))
SECRETS = [] if NO_SECRET else [modal.Secret.from_name("huggingface-secret")]

app = modal.App("efficientrag-mini")
vol = modal.Volume.from_name("efficientrag-mini-cache", create_if_missing=True)
VLLM_WHEEL = "https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
image = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
         .run_commands("printf 'torch==2.11.0\\ntorchvision==0.26.0\\n' > /tmp/constraints.txt",
                       f"pip install -q '{VLLM_WHEEL}' -c /tmp/constraints.txt datasets sentencepiece protobuf tiktoken accelerate numpy",
                       "pip uninstall -y -q torchcodec || true",
                       "pip install -q -U 'protobuf>=7'")   # sentencepiece(deberta の tokenizer 変換)が protobuf 7 系で生成されているため
         .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", **({"ER_NO_SECRET": "1"} if NO_SECRET else {})})
         .add_local_dir(HERE.parent / "src", "/root/pkg/src", ignore=["__pycache__"])
         .add_local_dir(HERE, "/root/pkg/efficientrag_mini", ignore=["__pycache__", "results"]))

C = Path("/cache/er")
PKG = "/root/pkg/efficientrag_mini"
GPU_GEN, GPU_TRAIN = "A100-80GB", "A100-40GB"


def sh(script: str, args: dict) -> dict:
    """スクリプトを別プロセスで実行する(GPU メモリを毎回解放するため)。出力は、その場でログに流し、最後の行の JSON を返す。"""
    p = subprocess.Popen([sys.executable, "-u", f"{PKG}/{script}", json.dumps(args)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: list[str] = []
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if p.wait() != 0:
        raise RuntimeError(f"{script} が失敗した\n" + "".join(lines[-40:]))
    for line in reversed(lines):       # 終了時の警告が JSON の後に出ることがあるので、最後の JSON 行を探す
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def droot(ds: str, small: bool) -> Path:
    return C / (ds + ("_small" if small else ""))


@app.function(image=image, cpu=4, memory=32768, volumes={"/cache": vol}, secrets=SECRETS, timeout=3600)
def prepare(ds: str, small: bool = False):
    sys.path.insert(0, PKG)
    from data import build_splits, save_jsonl

    n_train, n_eval, nc = (N_TRAIN[ds], N_EVAL, CORPUS_QUESTIONS) if not small else (200, 12, 100)
    sp = build_splits(ds, n_train, n_eval, nc)
    r = droot(ds, small)
    save_jsonl(sp["train"], r / "train_q.jsonl")
    save_jsonl(sp["eval"], r / "eval_q.jsonl")
    (r / "corpus.json").write_text(json.dumps(sp["corpus"]), encoding="utf-8")
    (r / "train_corpus.json").write_text(json.dumps(sp["train_corpus"]), encoding="utf-8")
    vol.commit()
    return {"dataset": ds, "train": len(sp["train"]), "eval": len(sp["eval"]), "corpus": len(sp["corpus"]), "train_corpus": len(sp["train_corpus"])}


@app.function(image=image, gpu=GPU_GEN, cpu=8, memory=65536, volumes={"/cache": vol}, secrets=SECRETS, timeout=3 * 3600)
def synth(ds: str, small: bool = False):
    vol.reload()
    r = droot(ds, small)
    res = sh("synth.py", dict(model=SYNTH_LLM, questions=str(r / "train_q.jsonl"), out=str(r / "synth.jsonl")))
    vol.commit()
    return res


@app.function(image=image, gpu=GPU_TRAIN, cpu=8, memory=65536, volumes={"/cache": vol}, secrets=SECRETS, timeout=6 * 3600)
def fit(ds: str, sizes: list[int], small: bool = False, epochs: int | None = None):
    vol.reload()
    r = droot(ds, small)
    out = {}
    if small or not (r / "examples.jsonl").exists():
        out["build"] = sh("train.py", dict(task="build", root=str(r)))
        vol.commit()
    n_rec = sum(1 for _ in open(r / "examples.jsonl", encoding="utf-8"))
    for n in sizes:
        n = n_rec if n == 0 or n > n_rec else n
        tag = f"n{n}"
        extra = {"epochs": epochs} if epochs else {}
        out[tag] = sh("train.py", dict(task="fit", root=str(r), n=n, tag=tag, **extra))
        vol.commit()
    return out


@app.function(image=image, gpu=GPU_GEN, cpu=8, memory=98304, volumes={"/cache": vol}, secrets=SECRETS, timeout=4 * 3600)
def evaluate(ds: str, spec: dict, small: bool = False, out_name: str = "eval"):
    vol.reload()
    r = droot(ds, small)
    a = dict(spec, dataset=ds, root=str(r), llm=SYNTH_LLM, out=str(r / f"{out_name}.json"))
    er = {}
    for name, e in spec.get("efficientrag", {}).items():
        tag = e["tag"]
        er[name] = dict(e, labeler=str(r / "models" / tag / "labeler.pt"), filter=str(r / "models" / tag / "filter.pt"))
    a["efficientrag"] = er
    res = sh("evaluate.py", a)
    vol.commit()
    return {"summary": res, "json": (r / f"{out_name}.json").read_text(encoding="utf-8")}


def default_spec(tag: str, full_ablation: bool, small: bool, latency: bool) -> dict:
    er = {"efficientrag": {"tag": tag}}
    if full_ablation is not False:
        er |= {"efficientrag_no_filter": {"tag": tag, "variant": "no_filter"}, "efficientrag_no_tag": {"tag": tag, "variant": "no_tag"}}
    if full_ablation is True:
        er |= {"efficientrag_k3": {"tag": tag, "k": 3}, "efficientrag_k10": {"tag": tag, "k": 10},
               "efficientrag_thr03": {"tag": tag, "threshold": 0.3}, "efficientrag_thr07": {"tag": tag, "threshold": 0.7}}
    spec = {"efficientrag": er, "limit": 12 if small else None}
    if latency:
        spec["latency"] = {"n": 3 if small else 30, "efficientrag": "efficientrag"}
    return spec


@app.local_entrypoint()
def main(stage: str = "smoke", datasets: str = "hotpotqa", sizes: str = "0", small: bool = False, epochs: int = 0, tag: str = "", name: str = "eval"):
    res_dir = HERE / "results"
    res_dir.mkdir(exist_ok=True)
    dss = [d for d in datasets.split(",") if d]
    szs = [int(s) for s in sizes.split(",") if s != ""]
    suf = "_small" if small else ""
    if stage == "prepare":
        for r in prepare.map(dss, kwargs={"small": small}):
            print(r)
    elif stage == "synth":
        for ds, r in zip(dss, synth.map(dss, kwargs={"small": small})):
            print(ds, r)
            (res_dir / f"synth_{ds}{suf}.json").write_text(json.dumps(r, indent=1))
    elif stage == "fit":
        for ds, r in zip(dss, fit.map(dss, kwargs={"sizes": szs, "small": small, "epochs": epochs or None})):
            print(ds, r)
            (res_dir / f"fit_{ds}{suf}.json").write_text(json.dumps(r, indent=1))
    elif stage == "eval":
        # tag 未指定なら、全部のサイズを使った n<N> は、fit の結果から決める
        def spec_for(ds):
            t = tag or "n0"
            fit_res = res_dir / f"fit_{ds}{suf}.json"
            if not tag and fit_res.exists():
                t = [k for k in json.loads(fit_res.read_text()) if k.startswith("n")][-1]
            sizes_done = [k for k in (json.loads(fit_res.read_text()) if fit_res.exists() else {}) if k.startswith("n")]
            sp = default_spec(t, full_ablation=(True if ds == "hotpotqa" else "min"), small=small, latency=(ds == "hotpotqa"))
            for k in sizes_done:      # データ量の条件(質問数の違い)
                if k != t:
                    sp["efficientrag"][f"efficientrag_{k}"] = {"tag": k}
            return sp
        outs = evaluate.map(dss, [spec_for(d) for d in dss], kwargs={"small": small, "out_name": name})
        for ds, r in zip(dss, outs):
            (res_dir / f"{name}_{ds}{suf}.json").write_text(r["json"], encoding="utf-8")
            print(ds, r["summary"])
    elif stage == "smoke":
        ds = dss[0]
        print(prepare.remote(ds, True))
        print(synth.remote(ds, True))
        f = fit.remote(ds, [0], True, 3)
        print(f)
        t = [k for k in f if k.startswith("n")][-1]
        r = evaluate.remote(ds, default_spec(t, False, True, True), True, "eval")
        (res_dir / f"smoke_eval_{ds}.json").write_text(r["json"], encoding="utf-8")
        print(r["summary"])
    else:
        raise SystemExit("unknown stage")
