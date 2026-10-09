"""Đưa model ONNX sang web để thử trong trình duyệt (trang #/lab của web/).

    python scripts/export_web.py
→ web/public/models/{fruit_det,fruit_quality[,drying_level]}.onnx + models.json
→ web/public/lab-samples/ (ảnh tập TEST + video băng chuyền thử dạng WebM)
Thiếu .onnx nhưng có .pt thì export trước. Model mức sấy chỉ có khi đã train (weights/drying_level.pt).
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import cv2
import onnxruntime as ort

from nhandienqua import PROJECT_ROOT, VISION_ROOT, WEIGHTS_DIR
from nhandienqua.yolo import YOLO

WEB = PROJECT_ROOT / "web" / "public"
MODELS = {  # khoá trong models.json → (tên file, task, imgsz)
    "detect": ("fruit_det", "detect", 640),
    "quality": ("fruit_quality", "classify", 224),
    "drying": ("drying_level", "classify", 224),
}


def onnx_info(path) -> dict:
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = s.get_modelmeta().custom_metadata_map
    names = eval(meta["names"], {"__builtins__": {}})  # dict {id: tên} do Ultralytics ghi
    shape = s.get_inputs()[0].shape
    return {"names": [names[i] for i in sorted(names)], "imgsz": [int(shape[2]), int(shape[3])],
            "output": [d if isinstance(d, int) else -1 for d in s.get_outputs()[0].shape]}


def export_cam(out) -> dict:
    """Model tốt/hư có thêm đầu ra CAM (bản đồ nhiệt hư hỏng). Kiểm: xác suất phải trùng bản ONNX thường."""
    import numpy as np
    import torch

    from nhandienqua.explain import CamClassifier

    model = YOLO(str(WEIGHTS_DIR / "fruit_quality.pt")).model.eval().float()
    cam = CamClassifier(model).eval()
    dst = WEIGHTS_DIR / "fruit_quality_cam.onnx"
    torch.onnx.export(cam, torch.zeros(1, 3, 224, 224), str(dst), input_names=["images"], output_names=["probs", "cams"],
                      opset_version=17, dynamo=False)
    x = np.random.default_rng(0).random((1, 3, 224, 224), dtype=np.float32)
    ref = ort.InferenceSession(str(WEIGHTS_DIR / "fruit_quality.onnx"), providers=["CPUExecutionProvider"])
    got = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    p_ref = ref.run(None, {"images": x})[0]
    p_got, cams = got.run(None, {"images": x})
    diff = float(np.abs(p_ref - p_got).max())
    if diff > 1e-4:
        raise SystemExit(f"CAM ONNX lệch xác suất {diff:.2e} so với bản thường — dừng")
    shutil.copy2(dst, out / dst.name)
    print(f"quality+CAM: {dst.name} · lệch xác suất so với bản thường {diff:.1e} · CAM {cams.shape}")
    return {"file": f"models/{dst.name}", "bytes": dst.stat().st_size, "cam": list(cams.shape[2:])}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.chdir(WEIGHTS_DIR)
    out = WEB / "models"
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    for key, (stem, task, imgsz) in MODELS.items():
        pt, onnx = WEIGHTS_DIR / f"{stem}.pt", WEIGHTS_DIR / f"{stem}.onnx"
        if not onnx.exists() and pt.exists():
            YOLO(str(pt)).export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False)
        if not onnx.exists():
            manifest[key] = None
            print(f"{key}: chưa có model ({stem}.pt) — bỏ qua")
            continue
        shutil.copy2(onnx, out / onnx.name)
        manifest[key] = {"file": f"models/{onnx.name}", "task": task, "bytes": onnx.stat().st_size, **onnx_info(onnx)}
        print(f"{key}: {onnx.name} {onnx.stat().st_size / 1e6:.1f} MB · {len(manifest[key]['names'])} lớp")
    if manifest.get("quality"):
        manifest["quality"].update(export_cam(out))
    (out / "models.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    # ảnh/video mẫu (tập TEST — model chưa thấy khi train)
    smp = WEB / "lab-samples"
    smp.mkdir(parents=True, exist_ok=True)
    picks = []
    real = sorted((VISION_ROOT / "data/detect/images/test_real").glob("*.jpg"))
    synth = sorted((VISION_ROOT / "data/detect/images/test_synth").glob("*.jpg"))
    for i, p in enumerate([*real[::max(1, len(real) // 8)][:8], *synth[:6]]):
        img = cv2.imread(str(p))
        s = 1280 / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        name = f"{'that' if 'test_real' in str(p) else 'tonghop'}_{i:02d}.jpg"
        cv2.imwrite(str(smp / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        picks.append(name)
    vid = VISION_ROOT / "runs/demo/bang_chuyen_test.mp4"
    if vid.exists():
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter(str(smp / "bang_chuyen_test.webm"), cv2.VideoWriter_fourcc(*"VP80"), fps, (w, h))
        n = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            vw.write(f)
            n += 1
        vw.release()
        print(f"video: bang_chuyen_test.webm ({n} khung)")
    (smp / "samples.json").write_text(json.dumps({"images": picks, "video": "bang_chuyen_test.webm" if vid.exists() else None}), encoding="utf-8")
    print(f"{len(picks)} ảnh mẫu → {smp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
