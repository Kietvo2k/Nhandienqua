"""Chạy TRÊN máy Colab (do scripts/colab_job.py khởi động nền). Tham số giống hệt vision/scripts/train.py.

    python /content/train_remote.py quality detect
Kết quả: /content/weights/fruit_quality.pt, /content/weights/fruit_det.pt; log tiến độ ở /content/train.log;
xong hết ghi /content/DONE (hoặc /content/FAILED kèm lỗi).
"""

import shutil
import sys
import traceback
from pathlib import Path

from ultralytics import YOLO

ROOT = Path("/content")
RUNS = ROOT / "runs"
OUT = ROOT / "weights"
TASKS = {
    "quality": dict(init="yolo26n-cls.pt", out="fruit_quality.pt",
                    kw=dict(data=str(ROOT / "data/quality"), imgsz=224, epochs=30, batch=128, patience=8,
                            fliplr=0.5, flipud=0.5)),
    "detect": dict(init="yolo26n.pt", out="fruit_det.pt",
                   kw=dict(data=str(ROOT / "data/detect/data.yaml"), imgsz=640, epochs=60, batch=32, patience=15,
                           fliplr=0.5, flipud=0.5, degrees=10, mosaic=1.0, close_mosaic=10)),
}


def train(task: str) -> None:
    cfg = TASKS[task]
    run = RUNS / task / "train"
    last = run / "weights" / "last.pt"
    if (OUT / cfg["out"]).exists():
        print(f"[ndq] {task}: đã có {cfg['out']}, bỏ qua", flush=True)
        return
    if last.exists():  # máy bị khởi động lại lệnh → train tiếp
        print(f"[ndq] {task}: tiếp tục từ {last}", flush=True)
        YOLO(str(last)).train(resume=True)
    else:
        print(f"[ndq] {task}: bắt đầu", flush=True)
        YOLO(cfg["init"]).train(project=str(RUNS / task), name="train", exist_ok=True, seed=0, plots=True,
                               workers=8, cache="ram", **cfg["kw"])  # VM Colab ~57 GB RAM → nạp ảnh sẵn, nhanh hơn
    OUT.mkdir(exist_ok=True)
    shutil.copy(run / "weights" / "best.pt", OUT / cfg["out"])
    print(f"[ndq] {task}: xong → {OUT / cfg['out']}", flush=True)


def main() -> int:
    try:
        for t in sys.argv[1:] or ["quality", "detect"]:
            train(t)
        (ROOT / "DONE").write_text("ok")
        return 0
    except Exception:
        (ROOT / "FAILED").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    sys.exit(main())
