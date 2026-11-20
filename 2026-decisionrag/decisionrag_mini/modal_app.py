"""Modal で縮小再現を回す(コンテナは同時に最大 10 まで)。GPU は環境変数 DR_GPU に Modal の GPU 名を入れて選ぶ(既定は VRAM 80GB の GPU)。

  modal run decisionrag_mini/modal_app.py --stage prepare                 # SQuAD から木・問い・学習例を作る(CPU)
  modal run decisionrag_mini/modal_app.py --stage smoke                   # 小さな件数で、学習と評価が通るか確認する
  modal run decisionrag_mini/modal_app.py --stage train                   # 判定モデルの学習(LoRA)
  modal run decisionrag_mini/modal_app.py --stage eval                    # 全実験
  DR_GPU=<Modal の GPU 名> modal run decisionrag_mini/modal_app.py --stage train   # GPU を変える

別のワークスペース(Secret が無い)で動かすときは RR_NO_SECRET=1 を付ける。結果は decisionrag_mini/results/*.json に保存される。
`modal run` のローカルプロセスは、完了まで生かしておくこと。Colab や手元の GPU では、run_local.py を直接使える。
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
sys.path.insert(0, "/root/pkg/decisionrag_mini")  # コンテナ内では add_local_dir でここに置かれる
from jobs import EVAL, MODEL, SMOKE, TRAIN  # noqa: E402

GPU = os.environ.get("DR_GPU", "A100-80GB")
NO_SECRET = bool(os.environ.get("RR_NO_SECRET"))
SECRETS = [] if NO_SECRET else [modal.Secret.from_name("huggingface-secret")]

app = modal.App("decisionrag-mini")
vol = modal.Volume.from_name("decisionrag-mini-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==5.22.2", "peft==0.21.2", "accelerate", "datasets", "scipy", "numpy")
         .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "DR_GPU": GPU, **({"RR_NO_SECRET": "1"} if NO_SECRET else {})})
         .add_local_dir(HERE.parent / "src", "/root/pkg/src", ignore=["__pycache__"])
         .add_local_dir(HERE, "/root/pkg/decisionrag_mini", ignore=["__pycache__", "results", "work"]))
PKG = "/root/pkg/decisionrag_mini"
WORK = "/cache/dr"


def sh(args: list[str]) -> str:
    """run_local.py を別プロセスで実行する。出力は、その場でログに流しながら、最後に全文を返す(途中経過が見えるように)。"""
    p = subprocess.Popen([sys.executable, "-u", f"{PKG}/run_local.py", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: list[str] = []
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if p.wait() != 0:
        raise RuntimeError("run_local.py が失敗した\n" + "".join(lines[-40:]))
    return "".join(lines)


@app.function(image=image, cpu=8, memory=32768, volumes={"/cache": vol}, secrets=SECRETS, timeout=2 * 3600)
def prepare(smoke: bool = False):
    work = f"{WORK}/work{'_smoke' if smoke else ''}"
    n = (SMOKE if smoke else TRAIN)["n_train_questions"]
    out = sh(["prepare", "--work", work, "--n-train-questions", str(n)])
    vol.commit()
    return out.strip().splitlines()[-1]


@app.function(image=image, gpu=GPU, cpu=8, memory=65536, volumes={"/cache": vol}, secrets=SECRETS, timeout=12 * 3600)
def train(smoke: bool = False, full: bool = False):
    vol.reload()
    work = f"{WORK}/work{'_smoke' if smoke else ''}"
    cfg = SMOKE if smoke else TRAIN
    args = ["train", "--work", work, "--out", f"{work}/judge{'_full' if full else ''}", "--model", MODEL, "--epochs", str(cfg.get("epochs", 1)), "--batch", str(cfg.get("batch", 8))]
    if cfg.get("limit"):
        args += ["--limit", str(cfg["limit"])]
    if full:
        args.append("--full")
    out = sh(args)
    vol.commit()
    return out.strip().splitlines()[-1]


@app.function(image=image, gpu=GPU, cpu=8, memory=65536, volumes={"/cache": vol}, secrets=SECRETS, timeout=6 * 3600)
def evaluate(smoke: bool = False, full: bool = False, judge: str = "", tag: str = "judge"):
    vol.reload()
    work = f"{WORK}/work{'_smoke' if smoke else ''}"
    cfg = SMOKE if smoke else EVAL
    res_path = f"{work}/results_{tag}.json"
    sh(["eval", "--work", work, "--judge", judge or f"{work}/judge{'_full' if full else ''}", "--exps", cfg["exps"], "--n-eval", str(cfg["n_eval"]), "--result", res_path])
    vol.commit()
    return json.loads(Path(res_path).read_text())


@app.local_entrypoint()
def main(stage: str = "smoke", full: bool = False, tag: str = "judge"):
    res_dir = HERE / "results"
    res_dir.mkdir(exist_ok=True)
    if stage == "prepare":
        print(prepare.remote())
    elif stage == "train":
        print(train.remote(False, full))
    elif stage == "eval":
        res = evaluate.remote(False, full, "", tag)
        (res_dir / f"{tag}.json").write_text(json.dumps(res, ensure_ascii=False))
        print("保存:", res_dir / f"{tag}.json")
    elif stage == "smoke":
        print(prepare.remote(True))
        print(train.remote(True, False))
        res = evaluate.remote(True, False, "", "smoke")
        print(json.dumps({k: v.get("summary") if isinstance(v, dict) and "summary" in v else "…" for k, v in res.items() if k != "setup"}, ensure_ascii=False)[:600])
