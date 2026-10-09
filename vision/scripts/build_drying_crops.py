"""Chuẩn bị dữ liệu cho model MỨC SẤY (chưa train được vì chưa có nhãn).

Không có dataset public nào gán nhãn mức sấy (chưa đạt / đạt / quá khô) cho xoài/chuối sấy. Script này:
  1. Cắt từng miếng "xoai-say" từ dataset Roboflow `kg_mango_dried` → data/drying_unlabeled/
     (bỏ bản augment trùng của Roboflow — chỉ giữ 1 bản mỗi ảnh gốc).
  2. Tạo sẵn khung thư mục data/drying/{train,val,test}/{under,ok,over}/.

Người gán nhãn (nên đo độ ẩm thật bằng cân sấy ẩm) kéo ảnh vào đúng thư mục, rồi báo để chạy:
    python scripts/train.py drying

    python scripts/build_drying_crops.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2

from nhandienqua import VISION_ROOT
from nhandienqua.data.boxes import from_yolo

SRC = VISION_ROOT / "data" / "raw" / "kg_mango_dried"
OUT = VISION_ROOT / "data" / "drying_unlabeled"
DRY = VISION_ROOT / "data" / "drying"
RF = re.compile(r"_(jpe?g|png)\.rf\.[0-9a-f]+$", re.I)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        for lab in ("under", "ok", "over"):
            (DRY / sp / lab).mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n_img = n_crop = 0
    for img_path in sorted(SRC.rglob("images/*.jpg")):
        base = RF.sub("", img_path.stem)
        if base in seen:  # Roboflow sinh 3 bản augment / ảnh gốc
            continue
        seen.add(base)
        lab = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None or not lab.exists():
            continue
        h, w = img.shape[:2]
        n_img += 1
        for k, line in enumerate(lab.read_text(encoding="utf-8").split("\n")):
            parts = line.split()
            if len(parts) != 5:
                continue
            x1, y1, x2, y2 = from_yolo(tuple(map(float, parts[1:])), w, h)
            px, py = (x2 - x1) * 0.1, (y2 - y1) * 0.1
            x1, y1, x2, y2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py))
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            cv2.imwrite(str(OUT / f"{base}_{k}.jpg"), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
            n_crop += 1
    print(f"{n_img} ảnh gốc (đã bỏ bản augment) → {n_crop} miếng xoài sấy trong {OUT}")
    print(f"Gán nhãn: kéo ảnh vào {DRY}/<train|val|test>/<under|ok|over>/ rồi chạy: python scripts/train.py drying")
    return 0


if __name__ == "__main__":
    sys.exit(main())
