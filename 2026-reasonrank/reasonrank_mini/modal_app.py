"""Modal で縮小再現を回す(コンテナは同時に最大 10 まで。GPU は VRAM 80GB のもの)。

  modal run reasonrank_mini/modal_app.py --stage prepare              # データ作成(CPU)
  modal run reasonrank_mini/modal_app.py --stage teacher              # 教師の推論つき順位をサンプルし、SFT データを作る
  modal run reasonrank_mini/modal_app.py --stage run --jobs zs_direct,zs_cot,zs_teacher
  modal run reasonrank_mini/modal_app.py --stage run --jobs sft_direct,sft_reason,sft_reason_nofilter
  modal run reasonrank_mini/modal_app.py --stage run --jobs grpo_multi,grpo_ndcg,rl_only
  modal run reasonrank_mini/modal_app.py --stage smoke                # 小さな件数で全体の動作確認

条件の定義は jobs.py。結果は reasonrank_mini/results/<条件名>.json に保存される。`modal run` のローカルプロセスは完了まで生かしておくこと。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/root/pkg/reasonrank_mini")  # コンテナ内では add_local_dir でここに置かれる
from jobs import DATA, GRPO, JOBS, SFT_MAXLEN, STUDENT, TEACHER  # noqa: E402

import os

# 別のワークスペースで動かすとき(Secret が無い)は、環境変数 RR_NO_SECRET=1 で外す。公開モデルの取得だけなら、トークンは要らない。
# コンテナ側でも同じ分岐になるよう、同じ値を image の環境変数に入れる(ローカルとリモートで依存の数が食い違うと、Modal が拒否する)
NO_SECRET = bool(os.environ.get("RR_NO_SECRET"))
SECRETS = [] if NO_SECRET else [modal.Secret.from_name("huggingface-secret")]

app = modal.App("reasonrank-mini")
vol = modal.Volume.from_name("reasonrank-mini-cache", create_if_missing=True)
VLLM_WHEEL = "https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
image = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
         .run_commands("printf 'torch==2.11.0\\ntorchvision==0.26.0\\n' > /tmp/constraints.txt",
                       f"pip install -q '{VLLM_WHEEL}' -c /tmp/constraints.txt datasets scipy numpy accelerate", "pip uninstall -y -q torchcodec || true")
         .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", **({"RR_NO_SECRET": "1"} if NO_SECRET else {})})
         .add_local_dir(HERE.parent / "src", "/root/pkg/src", ignore=["__pycache__"])
         .add_local_dir(HERE, "/root/pkg/reasonrank_mini", ignore=["__pycache__", "results", "work"]))

C = Path("/cache/rr")
PKG = "/root/pkg/reasonrank_mini"


def sh(script: str, args: dict) -> dict:
    """スクリプトを別プロセスで実行する(GPU メモリを毎回解放するため)。出力は、その場でログに流し、最後の行の JSON を返す。"""
    p = subprocess.Popen([sys.executable, "-u", f"{PKG}/{script}", json.dumps(args)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: list[str] = []
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if p.wait() != 0:
        raise RuntimeError(f"{script} が失敗した\n" + "".join(lines[-40:]))
    return json.loads(lines[-1].strip()) if lines else {}


@app.function(image=image, cpu=8, memory=65536, volumes={"/cache": vol}, secrets=SECRETS, timeout=2 * 3600)
def prepare(small: bool = False):
    sys.path.insert(0, PKG)
    from data import load_hotpot, make_lists, save_jsonl

    d = dict(DATA)
    if small:
        d.update(train_pool=600, n_train=60, n_eval20=20, n_eval100=10)
    train_rows = load_hotpot("train", limit=d["train_pool"])
    dev_rows = load_hotpot("validation")
    out = C / ("data_small" if small else "data")
    save_jsonl(make_lists(train_rows, train_rows, "train", seed=0, n_queries=d["n_train"]), out / "train.jsonl")
    save_jsonl(make_lists(dev_rows, dev_rows, "eval20", seed=1, n_queries=d["n_eval20"]), out / "eval20.jsonl")
    save_jsonl(make_lists(dev_rows, dev_rows, "eval100", seed=2, n_queries=d["n_eval100"]), out / "eval100.jsonl")
    vol.commit()
    return {k: sum(1 for _ in open(out / f"{k}.jsonl")) for k in ("train", "eval20", "eval100")} | {"train_rows": len(train_rows), "dev_rows": len(dev_rows)}


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=98304, volumes={"/cache": vol}, secrets=SECRETS, timeout=4 * 3600)
def teacher(small: bool = False):
    """教師で推論つきの順位をサンプルし、SFT データ 3 種(direct / reason / nofilter)を作る。"""
    sys.path.insert(0, PKG)
    sys.path.insert(0, "/root/pkg/src")
    from data import load_jsonl
    from reasonrank_core import build_prompt

    vol.reload()
    root = C / ("data_small" if small else "data")
    lists = load_jsonl(root / "train.jsonl")
    res = sh("vllm_run.py", dict(task="teacher", lists=str(root / "train.jsonl"), model=TEACHER, n=DATA["teacher_n"], max_tokens=DATA["teacher_max_tokens"],
                                 out=str(root / "teacher.jsonl")))
    samples = load_jsonl(root / "teacher.jsonl")
    by_q: dict[str, list[dict]] = {}
    for s in samples:
        by_q.setdefault(s["qid"], []).append(s)
    thr = DATA["teacher_threshold"]
    out = {"direct": [], "reason": [], "nofilter": []}
    stats = {"queries": len(lists), "exact": sum(s["exact"] for s in samples) / len(samples),
             "mean_ndcg": sum(s["ndcg10"] for s in samples) / len(samples), "pass": 0}
    for l in lists:
        ids = [c["id"] for c in l["cands"]]
        texts = [c["text"] for c in l["cands"]]
        rid = {x: i + 1 for i, x in enumerate(ids)}
        gold_order = " > ".join(f"[{rid[x]}]" for x in l["gold_list"])
        out["direct"].append({"prompt": None, "qid": l["qid"], "completion": f"<answer>{gold_order}</answer>", "reasoning": False})
        ok = [s for s in by_q.get(l["qid"], []) if s["exact"] and s["think"]]

        def comp(s):
            return "<think>\n" + s["think"] + "\n</think>\n<answer>" + " > ".join(f"[{i + 1}]" for i in s["order"]) + "</answer>"
        good = sorted([s for s in ok if s["ndcg10"] >= thr], key=lambda s: (-s["ndcg10"], s["ntok"]))
        if good:    # 自己整合性フィルタ: 正解との NDCG@10 が閾値以上の出力だけを残す(1 質問に 1 つ)
            stats["pass"] += 1
            out["reason"].append({"qid": l["qid"], "completion": comp(good[0]), "reasoning": True})
        if ok:      # フィルタなし: 形式が正しい出力から 1 つ(NDCG にかかわらず)
            out["nofilter"].append({"qid": l["qid"], "completion": comp(ok[0]), "reasoning": True})
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT)
    by_id = {l["qid"]: l for l in lists}
    for name, rows in out.items():
        fin = []
        for r in rows:
            l = by_id[r["qid"]]
            u = build_prompt(l["query"], [c["text"] for c in l["cands"]], reasoning=r["reasoning"])
            fin.append({"prompt": tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True), "completion": r["completion"]})
        (root / f"sft_{name}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in fin), encoding="utf-8")
        stats[f"n_{name}"] = len(fin)
    vol.commit()
    return stats | res


def _eval(model: str, spec: dict, root: Path, tag: str) -> dict:
    res = {}
    for name, w in (("eval20", dict(w=20, s=10)), ("eval100", dict(w=20, s=10))):
        res[name] = sh("vllm_run.py", dict(task="eval", lists=str(root / f"{name}.jsonl"), model=model, reasoning=spec["reasoning"], thinking=spec.get("thinking"),
                                           max_tokens=spec["max_tokens"], temperature=spec.get("temperature", 0.0), **w))
    return res


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=98304, volumes={"/cache": vol}, secrets=SECRETS, timeout=12 * 3600)
def run_job(name: str, small: bool = False, skip_eval: bool = False):
    import threading
    import time as _t

    vol.reload()
    stop = threading.Event()

    def _commit_loop():   # 途中で止まっても、保存したチェックポイントが Volume に残るよう、5 分ごとに commit する
        while not stop.wait(300):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print("commit 失敗:", str(e)[:100], flush=True)
    threading.Thread(target=_commit_loop, daemon=True).start()
    spec = JOBS[name]
    root = C / ("data_small" if small else "data")
    models = C / ("models_small" if small else "models")
    out: dict = {"name": name, "spec": spec}
    model = spec.get("model")
    if spec["kind"] == "sft":
        model = str(models / name)
        out["train"] = sh("train.py", dict(task="sft", model=spec.get("student", STUDENT), data=str(root / f"sft_{spec['data']}.jsonl"), out=model,
                                           epochs=1 if small else spec["epochs"], lr=spec["lr"], batch=8, max_len=SFT_MAXLEN, ckpt=True))
        vol.commit()
    elif spec["kind"] == "grpo":
        cur = spec["init"] if spec["init"] == STUDENT else str(models / spec["init"])
        g = dict(GRPO)
        if small:
            g.update(rounds=2, prompts_per_round=8, group=4)
        logs = []
        lists = [json.loads(l) for l in open(root / "train.jsonl", encoding="utf-8")]
        for r in range(g["rounds"]):
            sub = root / "work" / f"{name}_r{r}_lists.jsonl"
            sub.parent.mkdir(parents=True, exist_ok=True)
            seed = spec.get("seed", 0)
            chunk = lists[(seed * 1500 + r * g["prompts_per_round"]) % len(lists):][:g["prompts_per_round"]]
            sub.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in chunk), encoding="utf-8")
            ro = root / "work" / f"{name}_r{r}_rollout.jsonl"
            st = sh("vllm_run.py", dict(task="rollout", lists=str(sub), model=cur, g=g["group"], max_tokens=spec["max_tokens"], kind=spec["reward"], out=str(ro), seed=spec.get("seed", 0) * 100 + r))
            nxt = str(models / f"{name}_r{r + 1}")
            tr = sh("train.py", dict(task="grpo", model=cur, rollouts=str(ro), out=nxt, lr=g["lr"], minibatch=g["minibatch"], max_len=g["max_len"]))
            logs.append({"round": r, "rollout": st, "train": tr})
            print(name, "round", r, st, tr, flush=True)
            cur = nxt
            vol.commit()
        out["rounds"] = logs
        model = str(models / name)
        subprocess.run(["bash", "-c", f"rm -rf {model} && cp -r {cur} {model}"], check=True)
        vol.commit()
    if not skip_eval:
        out["eval"] = _eval(model, spec, root, name)
    stop.set()
    vol.commit()
    return out


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=98304, volumes={"/cache": vol}, secrets=SECRETS, timeout=3600)
def diag(name: str, limit: int = 100):
    """eval20 を実行して、出力の全文を保存し、無効な出力の内訳(打ち切り・形式崩れ)を返す。"""
    vol.reload()
    spec = JOBS[name]
    model = str(C / "models" / name) if spec["kind"] != "eval" else spec["model"]
    dump = str(C / "work" / f"diag_{name}.jsonl")
    Path(dump).parent.mkdir(parents=True, exist_ok=True)
    res = sh("vllm_run.py", dict(task="eval", lists=str(C / "data" / "eval20.jsonl"), limit=limit, model=model, reasoning=spec["reasoning"],
                                 thinking=spec.get("thinking"), max_tokens=spec["max_tokens"], w=20, s=10, dump=dump))
    rows = [json.loads(l) for l in open(dump, encoding="utf-8")]
    bad = [r for r in rows if not r["exact"]]
    cap = spec["max_tokens"]
    info = {"n": len(rows), "invalid": len(bad), "invalid_hit_cap": sum(r["ntok"] >= cap - 1 for r in bad), "invalid_no_think_end": sum(not r["has_think_end"] for r in bad),
            "invalid_no_answer": sum(not r["has_answer"] for r in bad)}
    ex = [r["text"][-700:] for r in bad[:3]]
    return {"eval": res, "info": info, "bad_tails": ex, "ok_sample": next((r["text"][:900] for r in rows if r["exact"]), "")}


@app.local_entrypoint()
def main(stage: str = "run", jobs: str = "", small: bool = False, skip_eval: bool = False):
    res_dir = HERE / "results"
    res_dir.mkdir(exist_ok=True)
    if stage == "prepare":
        print(prepare.remote(small))
    elif stage == "teacher":
        res = teacher.remote(small)
        print(res)
        (res_dir / ("teacher_small.json" if small else "teacher.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    elif stage == "diag":
        for n in [j for j in jobs.split(",") if j]:
            r = diag.remote(n)
            (res_dir / f"diag_{n}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
            print(n, r["info"])
    elif stage == "smoke":
        print(prepare.remote(True))
        print(teacher.remote(True))
        for n in ("zs_direct", "sft_reason", "grpo_multi"):
            print(run_job.remote(n, True))
    else:
        names = [j for j in jobs.split(",") if j]
        for n, r in zip(names, run_job.map(names, kwargs={"small": small, "skip_eval": skip_eval})):
            (res_dir / f"{n}{'_small' if small else ''}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
            print(n, json.dumps(r.get("eval", r.get("train")), ensure_ascii=False)[:600])
