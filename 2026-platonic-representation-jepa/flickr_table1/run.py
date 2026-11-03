"""text JEPA / Next Token Prediction を学習し、画像側(凍結した DINOv2)との整合度を評価する。

  # Flickr30k (Table 1 の縮小版): 5 variant を順に実行(1 variant あたり 200 epoch ≒ 580 万サンプル)
  python flickr_table1/run.py --dataset flickr --variants random good bad ntp ntp_pairs --epochs 200

  # COCO (縮小スケーリング実験): サンプル数を指定して実行
  python flickr_table1/run.py --dataset coco --variants good ntp --samples 50000 100000 300000 --seeds 0 1 2

variant: good | bad | ntp | ntp_pairs | random(未学習)
結果は flickr_table1/results/<dataset>_<variant>_seed<seed>_s<samples>.json に保存される。
"""
from __future__ import annotations

import argparse

from common import FLICKR_TRAIN, RESULTS, load_features, save_json


def train_one(dataset: str, variant: str, seed: int, samples: int, n_evals: int = 10, log=print,
              batch: int = 256, grad_checkpoint: bool = False):
    """特徴量のキャッシュを読み込み、1 つの run を実行して結果(dict)を返す。Modal からも呼ばれる。"""
    import jepa_text  # torch を読み込むので、必要になるまで import しない

    img, captions, splits = load_features(dataset)
    keep = splits != "skip"  # 取得に失敗した画像を除く
    result = jepa_text.run(variant, seed, samples, {"img": img[keep]}, [c for c, k in zip(captions, keep) if k],
                           splits[keep], n_evals=n_evals, log=log, batch=batch, grad_checkpoint=grad_checkpoint)
    result["dataset"] = dataset
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="flickr", help="flickr | coco(試運転なら coco_limit512 など、特徴量のディレクトリ名)")
    ap.add_argument("--variants", nargs="+", default=["good"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--samples", nargs="+", type=int, help="学習で見るサンプル数(複数指定可)")
    g.add_argument("--epochs", type=int, help="Flickr30k の epoch 数(1 epoch = 29,000 サンプル)")
    ap.add_argument("--n-evals", type=int, default=10, help="学習中に検証 mutual 10-NN を測る回数")
    ap.add_argument("--batch", type=int, default=256, help="1 バッチの画像数(論文の設定は 256)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="勾配チェックポイント。VRAM が 24GB 未満(12GB のローカル GPU など)のときに付ける")
    ap.add_argument("--skip-existing", action="store_true", help="結果の JSON がある組は飛ばす")
    a = ap.parse_args()
    if a.samples is None and a.epochs is None:
        ap.error("--samples か --epochs のどちらかを指定してください")
    sample_list = a.samples or [a.epochs * FLICKR_TRAIN]

    for seed in a.seeds:
        for samples in sample_list:
            for variant in a.variants:
                name = f"{a.dataset}_{variant}_seed{seed}_s{samples}.json"
                if a.skip_existing and (RESULTS / name).exists():
                    print("飛ばす(結果あり):", name)
                    continue
                save_json(name, train_one(a.dataset, variant, seed, samples, a.n_evals, batch=a.batch,
                                          grad_checkpoint=a.grad_checkpoint))


if __name__ == "__main__":
    main()
