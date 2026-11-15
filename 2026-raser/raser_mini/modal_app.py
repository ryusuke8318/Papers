"""Modal で LLM の実行を回す(GPU は VRAM 80GB のもの)。

  modal run raser_mini/modal_app.py --stage upload              # データを Volume に置く(CPU)
  modal run raser_mini/modal_app.py --stage smoke               # 各データセット 12 問で、動作と 1 問あたりの時間・トークンを実測
  modal run raser_mini/modal_app.py --stage run --models q7b,q3b --n 400

結果は Volume の /cache/raser/results/<タグ>_<データセット>.jsonl に保存される。取り出し:
  modal volume get raser-mini-cache raser/results <既存のディレクトリ>
`modal run` のローカルプロセスは、完了まで生かしておくこと。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
MODELS = {"q7b": "Qwen/Qwen2.5-7B-Instruct", "q3b": "Qwen/Qwen2.5-3B-Instruct"}
DATASETS = ["musique", "2wiki", "hotpotqa"]

app = modal.App("raser-mini")
vol = modal.Volume.from_name("raser-mini-cache", create_if_missing=True)
VLLM_WHEEL = "https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
image = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
         .run_commands("printf 'torch==2.11.0\\ntorchvision==0.26.0\\n' > /tmp/constraints.txt",
                       f"pip install -q '{VLLM_WHEEL}' -c /tmp/constraints.txt numpy accelerate", "pip uninstall -y -q torchcodec || true")
         .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir(HERE.parent / "src", "/root/pkg/src", ignore=["__pycache__"])
         .add_local_file(HERE / "run_llm.py", "/root/pkg/raser_mini/run_llm.py"))


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=65536, volumes={"/cache": vol}, timeout=3 * 3600)
def run(tag: str, n: int, datasets: list[str], sub: str):
    vol.reload()
    args = dict(model=MODELS[tag], tag=tag, datasets=datasets, n=n, data="/cache/raser/data", out=f"/cache/raser/{sub}")
    p = subprocess.Popen([sys.executable, "-u", "/root/pkg/raser_mini/run_llm.py", json.dumps(args)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if p.wait() != 0:
        raise RuntimeError("run_llm.py が失敗した\n" + "".join(lines[-40:]))
    vol.commit()
    js = [l for l in lines if l.startswith("{")]
    return json.loads(js[-1]) if js else {}


@app.local_entrypoint()
def main(stage: str = "smoke", models: str = "q7b,q3b", n: int = 400):
    if stage == "upload":
        with vol.batch_upload(force=True) as up:
            for ds in DATASETS:
                up.put_file(str(HERE / "data" / f"{ds}.jsonl"), f"raser/data/{ds}.jsonl")
        print("アップロード完了")
        return
    tags = models.split(",")
    sub, nn = ("smoke", 12) if stage == "smoke" else ("results", n)
    handles = [run.spawn(t, nn, DATASETS, sub) for t in tags]
    for t, h in zip(tags, handles):
        print(t, json.dumps(h.get(), ensure_ascii=False))
