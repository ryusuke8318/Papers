"""パスとデータ読み込みの共通部分。

特徴量のキャッシュは `flickr_table1/data/<dataset>/` に置く(.gitignore 済み)。
環境変数 PLATONIC_DATA で場所を変えられる。HuggingFace のキャッシュも同じ場所の下に置く。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA_ROOT = Path(os.environ.get("PLATONIC_DATA", HERE / "data"))
RESULTS = HERE / "results"
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FLICKR_TRAIN = 29000  # Flickr30k の 1 epoch = 29,000 画像


def data_dir(dataset: str) -> Path:
    return DATA_ROOT / dataset


def load_features(dataset: str):
    """(画像特徴量, キャプション(画像ごとに 5 つ), split の配列) を返す。"""
    d = data_dir(dataset)
    img = np.load(d / "img_dinov2_base_cls.npy")
    meta = json.loads((d / "meta.json").read_text())
    return img, meta["captions"], np.array(meta["splits"])


def save_json(name: str, obj) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    print(f"保存: {path}")
    return path
