"""ローカル試運転: core.py と同じコードで小規模に学習・評価する。

  pip install -r rankrag_mini/requirements-local.txt
  python rankrag_mini/run_local.py --smoke        # 試運転 (train 32 / test 8 / 1 epoch)
  python rankrag_mini/run_local.py                # 本番規模 (train 4000 / test 150 / 2 epoch)

結果は rankrag_mini/results/rankrag_*_local.json に保存される。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import core  # noqa: E402

DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="試運転: train 32 / test 8 / 1 epoch")
    ap.add_argument("--n-train", type=int, default=core.N_TRAIN)
    ap.add_argument("--n-test", type=int, default=core.N_TEST)
    ap.add_argument("--epochs", type=int, default=core.EPOCHS)
    ap.add_argument("--kinds", nargs="+", default=["qa_only", "rank_small", "rank_full", "specialist"])
    ap.add_argument("--model", default=core.MODEL_ID)
    ap.add_argument("--seed", type=int, default=core.SEED)
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.n_test, args.epochs = 32, 8, 1

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("CUDA がないため CPU で動かします。試運転でも時間がかかります。")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    train_p, test_p = DATA_DIR / "train.json", DATA_DIR / "test.json"
    if not (train_p.exists() and test_p.exists()):
        print("データ作成中...")
        seed_data = core.load_seed(HERE / "data_seed.json")
        core.save_rows(core.build_rows("train", args.n_train, seed_data, seed=args.seed), train_p)
        core.save_rows(core.build_rows("validation", args.n_test, seed_data, seed=args.seed), test_p)
    train_qa, test = core.load_rows(train_p), core.load_rows(test_p)
    rank_pool = core.build_rank_rows(train_qa, 10 * args.n_train, seed=args.seed)
    print(f"train_qa {len(train_qa)} / rank_pool {len(rank_pool)} / test {len(test)}")

    for kind in args.kinds:
        res = core.train_and_eval(kind, train_qa, rank_pool, test, model_id=args.model,
                                  epochs=args.epochs, seed=args.seed, device=device)
        for tag, r in res.items():
            p = RESULTS_DIR / f"rankrag_{tag}_local.json"
            p.write_text(json.dumps(r, indent=1, ensure_ascii=False))
            print(f"保存: {p}")


if __name__ == "__main__":
    main()
