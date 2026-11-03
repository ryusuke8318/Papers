"""画像を凍結した DINOv2 ViT-B/14 に通し、CLS 特徴量をキャッシュに保存する(1 回だけ実行する)。

  python flickr_table1/extract_features.py --dataset flickr          # Flickr30k (約 3 万枚、HF から画像ごと取得)
  python flickr_table1/extract_features.py --dataset coco            # COCO (約 12 万枚、URL から取得。画像は保存しない)
  python flickr_table1/extract_features.py --dataset coco --limit 512   # 試運転(coco_limit512/ に書く。全体から等間隔に 512 枚)

COCO はチャンクごとに保存するので、中断しても途中から再開できる。
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch
from PIL import Image

from common import data_dir


def load_dinov2():
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval().half()
    return proc, model


def extract_flickr(limit: int = 0, batch: int = 64, workers: int = 4, on_done=None):
    from datasets import load_dataset

    dsd = load_dataset("nlphuji/flickr30k")  # 全画像が 1 つの HF split に入り、Karpathy 分割は "split" 列にある
    ds = dsd[list(dsd.keys())[0]]
    if limit:
        ds = ds.select(range(limit))

    def caps_of(r):
        c = r["caption"]
        return json.loads(c) if isinstance(c, str) else list(c)

    meta = ds.remove_columns("image")
    captions = [caps_of(r) for r in meta]
    splits = [r["split"] for r in meta]
    print("分割:", {s: splits.count(s) for s in set(splits)})

    proc, model = load_dinov2()

    def collate(b):
        return proc(images=[x["image"].convert("RGB") for x in b], return_tensors="pt")["pixel_values"]

    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=collate)
    feats = []
    with torch.no_grad():
        for i, px in enumerate(loader):
            feats.append(model(pixel_values=px.cuda().half()).last_hidden_state[:, 0].float().cpu())  # CLS
            if i % 20 == 0:
                print(f"  {i * batch}/{len(ds)}")
    out = data_dir("flickr" if not limit else f"flickr_limit{limit}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "img_dinov2_base_cls.npy", torch.cat(feats).numpy())
    (out / "meta.json").write_text(json.dumps({"captions": captions, "splits": splits}))
    print("保存:", out)
    if on_done:
        on_done()
    return {"n": len(splits), "splits": {s: splits.count(s) for s in set(splits)}}


def extract_coco(limit: int = 0, batch: int = 64, workers: int = 4, chunk_batches: int = 100, on_done=None):
    """Karpathy の train + restval を学習用 (113,287)、validation を val (5,000)、test を test (5,000) とする。

    on_done: チャンクを 1 つ保存するたびに呼ぶ関数(Modal の Volume への commit に使う)。
    """
    from datasets import load_dataset

    dsd = load_dataset("yerevann/coco-karpathy")
    rename = {"train": "train", "restval": "train", "validation": "val", "test": "test"}
    urls, captions, splits = [], [], []
    for hf_split in ("train", "restval", "validation", "test"):
        for r in dsd[hf_split]:
            s = r["sentences"]
            s = ast.literal_eval(s) if isinstance(s, str) else list(s)
            if len(s) < 5:
                continue
            urls.append(r["url"])
            captions.append([x.strip() for x in s[:5]])
            splits.append(rename[hf_split])
    if limit:  # 試運転: train / val / test がすべて含まれるよう、全体から等間隔に取る
        pick = np.linspace(0, len(urls) - 1, limit).astype(int)
        urls, captions, splits = [urls[i] for i in pick], [captions[i] for i in pick], [splits[i] for i in pick]
    print("分割:", {s: splits.count(s) for s in set(splits)})

    proc, model = load_dinov2()
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    def fetch(url):
        for attempt in range(5):
            try:
                r = session.get(url, timeout=30)
                r.raise_for_status()
                return Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception:
                time.sleep(1 + 2 * attempt)
        return None

    class Batches(torch.utils.data.Dataset):
        def __init__(self, lo, hi):
            self.lo, self.hi = lo, hi

        def __len__(self):
            return (self.hi - self.lo + batch - 1) // batch

        def __getitem__(self, b):
            u = urls[self.lo + b * batch : min(self.hi, self.lo + (b + 1) * batch)]
            with ThreadPoolExecutor(12) as ex:
                imgs = list(ex.map(fetch, u))
            failed = torch.tensor([im is None for im in imgs])
            imgs = [im if im is not None else Image.new("RGB", (224, 224)) for im in imgs]
            return proc(images=imgs, return_tensors="pt")["pixel_values"], failed

    base = data_dir("coco" if not limit else f"coco_limit{limit}")  # 試運転は別の場所に書く
    chunk_dir = base / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk = batch * chunk_batches
    for k, lo in enumerate(range(0, len(urls), chunk)):
        f = chunk_dir / f"chunk_{k:04d}.npz"
        if f.exists():
            continue
        hi = min(len(urls), lo + chunk)
        loader = torch.utils.data.DataLoader(Batches(lo, hi), batch_size=None, num_workers=workers, prefetch_factor=2)
        feats, failed = [], []
        t0 = time.time()
        with torch.no_grad():
            for px, fl in loader:
                feats.append(model(pixel_values=px.cuda().half()).last_hidden_state[:, 0].float().cpu())
                failed.append(fl)
        feats, failed = torch.cat(feats).numpy(), torch.cat(failed).numpy()
        np.savez(f, feats=feats, failed=failed)
        if on_done:
            on_done()
        print(f"  chunk {k}: {hi}/{len(urls)} ({time.time() - t0:.0f}s, 取得失敗 {int(failed.sum())})")

    parts = [np.load(chunk_dir / f"chunk_{k:04d}.npz") for k in range((len(urls) + chunk - 1) // chunk)]
    feats = np.concatenate([p["feats"] for p in parts])
    failed = np.concatenate([p["failed"] for p in parts])
    assert len(feats) == len(urls)
    # 取得に失敗した画像は黒画像の特徴量になっている。学習・評価から外すため split を "skip" にする。
    splits = ["skip" if f else s for s, f in zip(splits, failed)]
    np.save(base / "img_dinov2_base_cls.npy", feats)
    (base / "meta.json").write_text(json.dumps({"captions": captions, "splits": splits}))
    summary = {"n": len(feats), "dim": feats.shape[1], "failed": int(failed.sum()),
               "splits": {s: splits.count(s) for s in set(splits)}}
    print(summary)
    if on_done:
        on_done()
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["flickr", "coco"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="試運転用: 先頭の N 枚だけ")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4, help="DataLoader のプロセス数(CPU コア数に合わせる)")
    a = ap.parse_args()
    (extract_flickr if a.dataset == "flickr" else extract_coco)(a.limit, a.batch, a.workers)
