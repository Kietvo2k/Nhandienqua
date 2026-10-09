"""CLI: nhandienqua predict <ảnh | thư mục | video | số camera | rtsp://…>

Ví dụ:
    nhandienqua predict samples/                         # ảnh → runs/predict/<thời gian>/ + decisions.jsonl
    nhandienqua predict bang_chuyen.mp4 --line 0.6       # video: theo dõi + bỏ phiếu, quyết định khi qua vạch 60% chiều ngang
    nhandienqua predict 0 --show                         # webcam / camera USB
    nhandienqua predict 0 --det weights/fruit_det.engine --cls weights/fruit_quality.engine   # Jetson TensorRT
    nhandienqua predict "gst:nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import VISION_ROOT
from .pipeline import Decision, FruitInspector

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# nhãn ASCII (cv2.putText không vẽ được dấu tiếng Việt)
SHORT = {"mango": "Xoai", "banana": "Chuoi", "dragon_fruit": "Thanh long", "jackfruit": "Mit", "sweet_potato": "Khoai lang"}


def draw(frame: np.ndarray, decisions: list[Decision], line_x: int | None = None, fps: float | None = None) -> np.ndarray:
    out = frame.copy()
    for d in decisions:
        x1, y1, x2, y2 = d.box
        # quả chạm mép khung (đang vào/ra): xám, chưa kết luận
        color = (150, 150, 150) if d.partial else (60, 60, 255) if d.action == "reject" else (90, 210, 90)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        verdict = "..." if d.partial else ("HU" if d.action == "reject" else "TOT")
        tag = f"{SHORT.get(d.fruit, d.fruit)} {verdict} {d.p_defect:.2f}"
        if d.track_id is not None:
            tag = f"#{d.track_id} {tag}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)
    if line_x is not None:
        cv2.line(out, (line_x, 0), (line_x, out.shape[0]), (0, 200, 255), 2)
    if fps is not None:
        cv2.putText(out, f"{fps:.1f} FPS", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def _build(args, track: bool) -> FruitInspector:
    from .models import UltralyticsClassifier, UltralyticsDetector

    det = UltralyticsDetector(args.det, conf=min(args.conf, 0.25), imgsz=args.imgsz, device=args.device, track=track)
    cls = UltralyticsClassifier(args.cls, imgsz=args.cls_imgsz, device=args.device)
    return FruitInspector(det, cls, det.names, cls.names, det_conf=args.conf, defect_threshold=args.defect_th)


def predict(args) -> int:
    src = args.source
    out_dir = Path(args.save) if args.save else VISION_ROOT / "runs" / "predict" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "decisions.jsonl", "w", encoding="utf-8")
    p = Path(src)
    is_cam = src.isdigit() or src.startswith(("rtsp://", "http://", "https://", "gst:"))

    if not is_cam and (p.is_dir() or p.suffix.lower() in IMG_EXT):
        insp = _build(args, track=False)
        files = sorted(x for x in (p.rglob("*") if p.is_dir() else [p]) if x.suffix.lower() in IMG_EXT)
        t0 = time.perf_counter()
        for f in files:
            frame = cv2.imread(str(f))
            if frame is None:
                continue
            ds = insp(frame)
            cv2.imwrite(str(out_dir / f.name), draw(frame, ds))
            log.write(json.dumps({"image": str(f), "decisions": [d.to_dict() for d in ds]}, ensure_ascii=False) + "\n")
        dt = time.perf_counter() - t0
        print(f"{len(files)} ảnh · {dt / max(1, len(files)) * 1000:.1f} ms/ảnh (gồm đọc/ghi file) → {out_dir}")
        return 0

    # video / camera: theo dõi + bỏ phiếu theo track
    from .voting import EmaSmoother, TrackVoter

    if src.startswith("gst:"):  # camera CSI trên Jetson: gst:nvarguscamerasrc ! ... ! appsink
        cap = cv2.VideoCapture(src[4:], cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        print(f"Không mở được nguồn video: {src}", file=sys.stderr)
        return 2
    insp = _build(args, track=True)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_x = int(w * args.line)
    voter = TrackVoter(line_x=line_x, threshold=args.defect_th)
    smoother = EmaSmoother(threshold=args.defect_th)  # chỉ cho nhãn hiển thị, không nhảy giữa các khung
    writer = cv2.VideoWriter(str(out_dir / "output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 30, (w, h))
    n, t0, fps = 0, time.perf_counter(), 0.0
    counts = {"accept": 0, "reject": 0}
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        t = time.perf_counter()
        ds = insp(frame)
        for ev in voter.update(ds):
            counts[ev["action"]] += 1
            ev["frame"] = n
            print(json.dumps(ev, ensure_ascii=False), flush=True)  # nơi nối tới PLC / van khí khi có phần cứng
            log.write(json.dumps(ev, ensure_ascii=False) + "\n")
        fps = 0.9 * fps + 0.1 / max(1e-6, time.perf_counter() - t) if n else 1 / max(1e-6, time.perf_counter() - t)
        vis = draw(frame, smoother.update(ds), line_x, fps)
        writer.write(vis)
        if args.show:
            cv2.imshow("nhandienqua", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        n += 1
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    dt = time.perf_counter() - t0
    print(f"{n} khung · {n / max(dt, 1e-6):.1f} FPS trung bình · nhận {counts['accept']} · loại {counts['reject']} → {out_dir}")
    return 0


def heatmap(args) -> int:
    """Ảnh → [bản đồ nhiệt phát hiện + khung | CAM "hư hỏng" từng quả]. Lưu *_heatmap.jpg."""
    import onnxruntime as ort
    import torch

    from .explain import CamClassifier, defect_map, detection_heatmap, letterbox, overlay
    from .models import UltralyticsDetector, _names
    from .pipeline import crop_with_pad, decide_quality
    from .yolo import YOLO

    det = UltralyticsDetector(args.det, conf=min(args.conf, 0.25), imgsz=args.imgsz, device=args.device)
    det_onnx = Path(args.det).with_suffix(".onnx")
    if not det_onnx.exists():  # cần đầu ra thô (trước NMS) để vẽ bản đồ phát hiện
        YOLO(args.det).export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False)
    sess = ort.InferenceSession(str(det_onnx), providers=["CPUExecutionProvider"])
    qyolo = YOLO(args.cls)
    qnames = _names(qyolo)
    cam_model = CamClassifier(qyolo.model.eval().float()).eval()

    out_dir = Path(args.save) if args.save else VISION_ROOT / "runs" / "heatmap" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = Path(args.source)
    files = sorted(x for x in (p.rglob("*") if p.is_dir() else [p]) if x.suffix.lower() in IMG_EXT)
    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        H, W = frame.shape[:2]
        # --- bản đồ phát hiện (lưới stride 8, bỏ phần pad của letterbox)
        lb, s, (px, py) = letterbox(frame, args.imgsz)
        x = np.ascontiguousarray(lb[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        heat = detection_heatmap(sess.run(None, {sess.get_inputs()[0].name: x})[0], imgsz=args.imgsz)
        st = 8
        hh = heat[py // st : (py + round(H * s)) // st, px // st : (px + round(W * s)) // st]
        left = overlay(frame, hh, lambda v: np.minimum(0.7, v * 0.9))  # chỉ tô nơi model thấy quả
        # --- CAM "hư hỏng" từng quả
        tiles = []
        for box, cls, conf, _ in det(frame):
            if conf < args.conf:
                continue
            fruit = det.names[int(cls)]
            crop, _ = crop_with_pad(frame, tuple(int(v) for v in box), 0.1)
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            k = 224 / min(ch, cw)  # Ultralytics classify: resize cạnh ngắn → cắt giữa 224
            r = cv2.resize(crop, (max(224, round(cw * k)), max(224, round(ch * k))), interpolation=cv2.INTER_LINEAR)
            oy, ox = (r.shape[0] - 224) // 2, (r.shape[1] - 224) // 2
            r = r[oy : oy + 224, ox : ox + 224]
            t = torch.from_numpy(np.ascontiguousarray(r[..., ::-1].transpose(2, 0, 1))).float()[None] / 255
            with torch.no_grad():
                probs, cams = cam_model(t)
            label, pd = decide_quality(probs[0].numpy(), qnames, fruit, args.defect_th)
            dm = defect_map(cams[0].numpy(), qnames, fruit)
            tile = r.copy()
            if dm is not None:
                v = np.sqrt(np.clip(dm, 0, None) / max(float(dm.max()), 1e-6))
                k = min(1.0, 0.1 + pd)  # quả chắc là tốt → tô rất nhạt (giống trang thử trên web)
                tile = overlay(r, v, lambda x: (0.25 + 0.4 * x) * k)
            color = (60, 60, 255) if label == "defect" else (90, 210, 90)
            cv2.rectangle(tile, (0, 0), (223, 223), color, 3)
            cv2.putText(tile, f"{SHORT.get(fruit, fruit)} {'HU' if label == 'defect' else 'TOT'} {pd:.2f}", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            tiles.append(tile)
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(left, (x1, y1), (x2, y2), color, 2)
        sc = 720 / H
        left = cv2.resize(left, (round(W * sc), 720))
        col = np.full((720, 224 * max(1, (len(tiles) + 2) // 3), 3), 30, np.uint8)
        for i, t in enumerate(tiles[: 3 * (col.shape[1] // 224)]):
            col[(i % 3) * 240 : (i % 3) * 240 + 224, (i // 3) * 224 : (i // 3 + 1) * 224] = t
        cv2.imwrite(str(out_dir / f"{f.stem}_heatmap.jpg"), np.hstack([left, col]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(files)} ảnh → {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from .models import DEFAULT_DET, DEFAULT_QUALITY

    ap = argparse.ArgumentParser(prog="nhandienqua", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("predict", help="chạy phát hiện + phân loại tốt/hư")
    pp.add_argument("source")
    pp.add_argument("--det", default=str(DEFAULT_DET), help="trọng số phát hiện (.pt/.onnx/.engine)")
    pp.add_argument("--cls", default=str(DEFAULT_QUALITY), help="trọng số chất lượng (.pt/.onnx/.engine)")
    pp.add_argument("--conf", type=float, default=0.35, help="ngưỡng tin cậy phát hiện")
    pp.add_argument("--defect-th", type=float, default=0.5, help="ngưỡng xác suất hư để loại")
    pp.add_argument("--imgsz", type=int, default=640)
    pp.add_argument("--cls-imgsz", type=int, default=224)
    pp.add_argument("--line", type=float, default=0.6, help="vạch quyết định (tỉ lệ chiều ngang khung hình)")
    pp.add_argument("--device", default=None, help="vd. 0 (GPU), cpu")
    pp.add_argument("--save", default=None, help="thư mục lưu kết quả")
    pp.add_argument("--show", action="store_true")
    pp.add_argument("--max-frames", type=int, default=0)
    hp = sub.add_parser("heatmap", help="bản đồ nhiệt: phát hiện + CAM hư hỏng từng quả (ảnh / thư mục)")
    hp.add_argument("source")
    hp.add_argument("--det", default=str(DEFAULT_DET), help="trọng số phát hiện .pt (cạnh đó cần/ sẽ xuất .onnx)")
    hp.add_argument("--cls", default=str(DEFAULT_QUALITY), help="trọng số chất lượng .pt")
    hp.add_argument("--conf", type=float, default=0.35)
    hp.add_argument("--defect-th", type=float, default=0.5)
    hp.add_argument("--imgsz", type=int, default=640)
    hp.add_argument("--device", default=None)
    hp.add_argument("--save", default=None)
    args = ap.parse_args(argv)
    return heatmap(args) if args.cmd == "heatmap" else predict(args)


if __name__ == "__main__":
    sys.exit(main())
