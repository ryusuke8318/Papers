"""ローカル試運転: core.py と同じコードで小規模に学習・評価する。

  pip install -r vat_mini/requirements-local.txt
  python vat_mini/run_local.py --smoke        # 試運転 (train 64 / test 10 / 1 epoch)
  python vat_mini/run_local.py                # 本番規模 (train 1200 / test 150 / 2 epoch)

結果は vat_mini/results/vat_*_local.json に保存される。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import core  # noqa: E402

RESULTS_DIR = HERE / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="試運転: train 64 / test 10 / 1 epoch")
    ap.add_argument("--n-train", type=int, default=core.N_TRAIN)
    ap.add_argument("--n-test", type=int, default=core.N_TEST)
    ap.add_argument("--epochs", type=int, default=core.EPOCHS)
    ap.add_argument("--kinds", nargs="+", default=list(core.KINDS))
    ap.add_argument("--model", default=core.MODEL_ID)
    ap.add_argument("--seed", type=int, default=core.SEED)
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.n_test, args.epochs = 64, 10, 1

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("CUDA がないため CPU で動かします。試運転でも時間がかかります。")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    seed_data = core.load_seed(HERE / "data_seed.json")
    rng = random.Random(args.seed)
    train_pool = seed_data["train_pool"][:]
    rng.shuffle(train_pool)
    train = [r for r in train_pool if r["type"] != "chain"][:args.n_train]
    test = seed_data["test_pool"][:args.n_test]
    rules = seed_data["rules"]
    print(f"train {len(train)} / test {len(test)} / rules {len(rules)}")

    for kind in args.kinds:
        res = core.train_and_eval(kind, train, test, rules, model_id=args.model,
                                  epochs=args.epochs, seed=args.seed, device=device)
        for tag, r in res.items():
            p = RESULTS_DIR / f"vat_{tag}_local.json"
            p.write_text(json.dumps(r, indent=1, ensure_ascii=False))
            print(f"保存: {p} acc={r['acc']:.3f}")


if __name__ == "__main__":
    main()
