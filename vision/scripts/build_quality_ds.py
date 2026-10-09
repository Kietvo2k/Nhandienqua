"""Dựng dataset phân loại chất lượng (ImageFolder cho YOLO-cls) từ kho ảnh cắt.

data/quality/{train,val,test}/<fruit>__<good|defect>/*.jpg

Mỗi ảnh cắt được dán lên nền băng chuyền (giống ảnh crop từ bbox detector khi chạy thật) — ảnh tốt
và ảnh hư có CÙNG kiểu nền, tránh model học nền thay vì vết bệnh. Lớp hiếm được render nhiều lần hơn ở train.

    python scripts/build_quality_ds.py
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from nhandienqua import VISION_ROOT
from nhandienqua.data.backgrounds import load_backgrounds
from nhandienqua.data.compose import render_quality_sample

CUT = VISION_ROOT / "data" / "cutouts"
OUT = VISION_ROOT / "data" / "quality"
SIZE = 224
TARGET_TRAIN = 1500  # số ảnh mục tiêu mỗi lớp ở train (lớp hiếm được render lặp tới tối đa 8 lần)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rng = np.random.default_rng(0)
    rows = [json.loads(line) for line in open(CUT / "manifest.jsonl", encoding="utf-8")]
    by = defaultdict(list)
    for r in rows:
        by[(r["split"], f"{r['fruit']}__{r['label']}")].append(r)
    bgs = load_backgrounds(rng)
    if OUT.exists():
        shutil.rmtree(OUT)
    stats = Counter()
    for (split, cls), items in tqdm(sorted(by.items()), desc="quality"):
        reps = min(8, max(1, round(TARGET_TRAIN / len(items)))) if split == "train" else 1
        d = OUT / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for r in items:
            rgba = np.asarray(Image.open(CUT / r["path"]).convert("RGBA"))
            stem = r["path"].rsplit("/", 1)[-1][:-4]
            for k in range(reps):
                img = render_quality_sample(rgba, bgs, rng, out_size=SIZE)
                cv2.imwrite(str(d / f"{stem}_{k}.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                stats[(split, cls)] += 1
    print("\nSố ảnh theo tập / lớp:")
    for split in ("train", "val", "test"):
        for (s, cls), n in sorted(stats.items()):
            if s == split:
                print(f"  {split:<6} {cls:<24} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
