"""Xuất ONNX (chạy trên PC/ONNX Runtime) + kiểm tra ONNX khớp PyTorch; in lệnh build TensorRT cho Jetson.

    python scripts/export.py
"""

from __future__ import annotations

import os
import sys

from nhandienqua import VISION_ROOT, WEIGHTS_DIR
from nhandienqua.yolo import YOLO

DET = WEIGHTS_DIR / "fruit_det.pt"
QUAL = WEIGHTS_DIR / "fruit_quality.pt"
REAL = VISION_ROOT / "data/detect/data_real_test.yaml"
QDATA = VISION_ROOT / "data/quality"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.chdir(WEIGHTS_DIR)
    rows = []
    if DET.exists():
        onnx = YOLO(str(DET)).export(format="onnx", imgsz=640, simplify=True, dynamic=False)
        # rect=False: ONNX tĩnh 640×640 luôn pad vuông → so cùng điều kiện (rect=True của PT cho điểm cao hơn ~7 điểm trên ảnh thật)
        pt_m = YOLO(str(DET)).val(data=str(REAL), split="val", imgsz=640, batch=1, rect=False, verbose=False, plots=False, project=str(VISION_ROOT / "runs/export"), name="det_pt", exist_ok=True)
        ox_m = YOLO(onnx, task="detect").val(data=str(REAL), split="val", imgsz=640, batch=1, verbose=False, plots=False, project=str(VISION_ROOT / "runs/export"), name="det_onnx", exist_ok=True)
        rows.append(("Phát hiện (mAP50-95, ảnh thật)", pt_m.box.map, ox_m.box.map, pt_m.speed["inference"], ox_m.speed["inference"]))
    if QUAL.exists():
        onnx = YOLO(str(QUAL)).export(format="onnx", imgsz=224, simplify=True, dynamic=False)
        pt_m = YOLO(str(QUAL)).val(data=str(QDATA), split="test", imgsz=224, batch=1, verbose=False, plots=False, project=str(VISION_ROOT / "runs/export"), name="cls_pt", exist_ok=True)
        ox_m = YOLO(onnx, task="classify").val(data=str(QDATA), split="test", imgsz=224, batch=1, verbose=False, plots=False, project=str(VISION_ROOT / "runs/export"), name="cls_onnx", exist_ok=True)
        rows.append(("Chất lượng (top-1, tập test)", pt_m.top1, ox_m.top1, pt_m.speed["inference"], ox_m.speed["inference"]))

    print("\nSo khớp PyTorch ↔ ONNX:")
    print(f"  {'Model':<34} {'PT':>8} {'ONNX':>8} {'Lệch':>8} {'PT ms':>7} {'ONNX ms':>8}")
    ok = True
    for name, a, b, sa, sb in rows:
        rel = abs(a - b) / max(a, 1e-9)
        ok &= rel < 0.01
        print(f"  {name:<34} {a:8.4f} {b:8.4f} {rel:8.2%} {sa:7.2f} {sb:8.2f}")
    print("  → KHỚP (< 1%)" if ok else "  → LỆCH ≥ 1%, kiểm tra lại export")
    print("""
Trên Jetson Orin Nano (JetPack 6) — build TensorRT FP16 NGAY TRÊN Jetson (engine không mang được giữa máy):
  yolo export model=fruit_det.pt format=engine imgsz=384,640 half=True device=0   # camera 16:9 (giữ tỉ lệ, đỡ pad)
  yolo export model=fruit_quality.pt format=engine imgsz=224 half=True device=0
Xem docs/huong-dan-jetson.md""")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
