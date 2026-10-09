"""Huấn luyện model (Ultralytics YOLO26) và chép trọng số tốt nhất vào vision/weights/.

    python scripts/train.py quality              # YOLO26n-cls tốt/hư → weights/fruit_quality.pt
    python scripts/train.py detect               # YOLO26n phát hiện 5 loại quả → weights/fruit_det.pt
    python scripts/train.py drying               # YOLO26n-cls mức sấy → weights/drying_level.pt (nếu có data/drying)
    python scripts/train.py detect --epochs 5    # chạy thử nhanh
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from nhandienqua import VISION_ROOT, WEIGHTS_DIR
from nhandienqua.yolo import YOLO

TASKS = {
    "detect": dict(model="yolo26n.pt", data=VISION_ROOT / "data/detect/data.yaml", imgsz=640, epochs=60, batch=32,
                   patience=15, out="fruit_det.pt", extra=dict(fliplr=0.5, flipud=0.5, degrees=10, mosaic=1.0, close_mosaic=10)),
    "quality": dict(model="yolo26n-cls.pt", data=VISION_ROOT / "data/quality", imgsz=224, epochs=30, batch=128,
                    patience=8, out="fruit_quality.pt", extra=dict(fliplr=0.5, flipud=0.5)),
    "drying": dict(model="yolo26n-cls.pt", data=VISION_ROOT / "data/drying", imgsz=224, epochs=30, batch=64,
                   patience=8, out="drying_level.pt", extra=dict(fliplr=0.5, flipud=0.5)),
}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--model", help="trọng số khởi tạo (mặc định YOLO26n)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", default="train")
    args = ap.parse_args(argv)
    cfg = TASKS[args.task]
    data = Path(cfg["data"])
    if not data.exists():
        print(f"Chưa có dữ liệu: {data} — chạy script build_* trước.", file=sys.stderr)
        return 2
    if data.is_dir():  # phân loại: mỗi lớp ở train/ phải có ảnh
        empty = [d.name for d in (data / "train").iterdir() if d.is_dir() and not any(d.iterdir())]
        if empty or not any((data / "train").iterdir()):
            print(f"Thư mục lớp chưa có ảnh: {empty or 'train/'} — gán nhãn xong rồi chạy lại.", file=sys.stderr)
            return 2

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WEIGHTS_DIR)  # mọi file Ultralytics tự tải (trọng số gốc, kiểm tra AMP) nằm trong vision/weights
    model = YOLO(str(WEIGHTS_DIR / (args.model or cfg["model"])))
    res = model.train(
        data=str(cfg["data"]),
        imgsz=cfg["imgsz"],
        epochs=args.epochs or cfg["epochs"],
        batch=args.batch or cfg["batch"],
        patience=cfg["patience"],
        workers=args.workers,
        project=str(VISION_ROOT / "runs" / args.task),
        name=args.name,
        exist_ok=True,
        seed=0,
        plots=True,
        **cfg["extra"],
    )
    best = Path(res.save_dir) / "weights" / "best.pt"
    dst = WEIGHTS_DIR / cfg["out"]
    shutil.copy2(best, dst)
    print(f"Đã chép {best} → {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
