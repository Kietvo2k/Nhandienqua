"""Đánh giá model trên tập test và ghi docs/ket-qua-model.md (số liệu lấy từ lần chạy này).

    python scripts/evaluate.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from nhandienqua import PROJECT_ROOT, VISION_ROOT, WEIGHTS_DIR
from nhandienqua.pipeline import decide_quality
from nhandienqua.yolo import YOLO

DOC = PROJECT_ROOT / "docs" / "ket-qua-model.md"
DET = WEIGHTS_DIR / "fruit_det.pt"
QUAL = WEIGHTS_DIR / "fruit_quality.pt"
DRY = WEIGHTS_DIR / "drying_level.pt"
DDIR = VISION_ROOT / "data" / "detect"
QDIR = VISION_ROOT / "data" / "quality"
VI = yaml.safe_load((VISION_ROOT / "configs/classes.yaml").read_text(encoding="utf-8"))["vi"]


def count_labels(split: str, names: list[str]) -> tuple[int, dict[str, int]]:
    files = list((DDIR / "labels" / split).glob("*.txt"))
    c: dict[str, int] = defaultdict(int)
    for f in files:
        for line in f.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                c[names[int(line.split()[0])]] += 1
    return len(files), c


def eval_detect(lines: list[str]) -> None:
    m = YOLO(str(DET))
    names = [m.names[i] for i in sorted(m.names)]
    lines += ["## 1. Phát hiện quả (YOLO26n, 640 px)", ""]
    for title, yml, split in (("Ảnh THẬT (test_real)", "data_real_test.yaml", "test_real"), ("Ảnh tổng hợp băng chuyền (test_synth)", "data_synth_test.yaml", "test_synth")):
        r = m.val(data=str(DDIR / yml), split="val", imgsz=640, batch=16, workers=2, plots=True, verbose=False,
                  project=str(VISION_ROOT / "runs/eval"), name=f"det_{split}", exist_ok=True)
        n_img, cnt = count_labels(split, names)
        lines += [f"### {title} — {n_img} ảnh", "",
                  f"mAP50 = **{r.box.map50:.3f}**, mAP50-95 = **{r.box.map:.3f}**, precision = {r.box.mp:.3f}, recall = {r.box.mr:.3f}",
                  f"Tốc độ (RTX 3060, PyTorch): {r.speed['inference']:.1f} ms/ảnh suy luận", "",
                  "| Lớp | Số bbox | mAP50-95 |", "|---|---:|---:|"]
        for i, n in enumerate(names):
            k = cnt.get(n, 0)
            lines.append(f"| {VI.get(n, n)} | {k} | {r.box.maps[i]:.3f} |" if k else f"| {VI.get(n, n)} | 0 | — (không có mẫu) |")
        lines += [f"", f"Biểu đồ: `vision/runs/eval/det_{split}/`", ""]


def eval_quality(lines: list[str]) -> None:
    m = YOLO(str(QUAL))
    qnames = [m.names[i] for i in sorted(m.names)]
    files = sorted((QDIR / "test").rglob("*.jpg"))
    truth = [f.parent.name for f in files]
    probs = []
    t0 = time.perf_counter()
    for i in range(0, len(files), 64):
        rs = m.predict([str(f) for f in files[i : i + 64]], imgsz=224, verbose=False)
        probs += [r.probs.data.cpu().numpy() for r in rs]
    dt = (time.perf_counter() - t0) / max(1, len(files)) * 1000
    probs = np.stack(probs)
    top1 = float(np.mean([qnames[int(p.argmax())] == t for p, t in zip(probs, truth)]))
    lines += ["## 2. Phân loại tốt / hư (YOLO26n-cls, 224 px)", "",
              f"Tập test: {len(files)} ảnh (ảnh cắt dán lên nền băng chuyền, nhóm ảnh gốc KHÔNG trùng train).",
              f"Top-1 trên {len(qnames)} lớp = **{top1:.3f}** · {dt:.1f} ms/ảnh (gồm đọc file, RTX 3060).", "",
              "Theo cách pipeline thật ra quyết định (chỉ xét lớp của loại quả mà detector báo, ngưỡng 0.5):", "",
              "| Quả | Ảnh test (tốt/hư) | Độ chính xác | Precision (hư) | Recall (hư) | Loại nhầm quả tốt |",
              "|---|---|---:|---:|---:|---:|"]
    by = defaultdict(list)
    for p, t in zip(probs, truth):
        fruit, lab = t.split("__")
        pred, _ = decide_quality(p, qnames, fruit, 0.5)
        by[fruit].append((lab, pred))
    for fruit, xs in sorted(by.items()):
        tp = sum(1 for a, b in xs if a == "defect" and b == "defect")
        fp = sum(1 for a, b in xs if a == "good" and b == "defect")
        fn = sum(1 for a, b in xs if a == "defect" and b == "good")
        ng = sum(1 for a, _ in xs if a == "good")
        nd = len(xs) - ng
        acc = sum(1 for a, b in xs if a == b) / len(xs)
        prec = f"{tp / (tp + fp):.3f}" if tp + fp else "—"
        rec = f"{tp / nd:.3f}" if nd else "— (không có ảnh hư)"
        fpr = f"{fp / ng:.3f}" if ng else "— (không có ảnh tốt)"
        lines.append(f"| {VI.get(fruit, fruit)} | {ng}/{nd} | {acc:.3f} | {prec} | {rec} | {fpr} |")
    lines += ["",
              "**Đọc số liệu cho đúng:**", "",
              "- **Khoai lang:** ảnh *hư* chỉ từ một bộ chụp phòng lab (GitHub aipal-nchu), ảnh *tốt* từ Fruits-360 + ảnh web. "
              "Hai lớp khác NGUỒN → model có thể học khác biệt nguồn (ánh sáng, độ nét) thay vì vết hư. Số liệu cao ở đây "
              "**không đáng tin**; cần ảnh khoai tốt và hư chụp cùng điều kiện.",
              "- **Mít:** *hư* = quả méo/dị dạng (không phải bệnh), ít ảnh → yếu nhất.",
              "- **Xoài, chuối, thanh long:** tốt và hư cùng nguồn trong mỗi bộ dữ liệu → đáng tin hơn, nhưng vẫn là ảnh public, không phải camera băng chuyền.",
              ""]


def video_section(lines: list[str]) -> None:
    """Đối chiếu kết quả chế độ video (theo dõi + bỏ phiếu) với đáp án của video thử, nếu đã chạy."""
    import json

    truth_p = VISION_ROOT / "runs/demo/truth.json"
    dec_p = VISION_ROOT / "runs/predict/demo_video/decisions.jsonl"
    if not (truth_p.exists() and dec_p.exists()):
        return
    truth = json.loads(truth_p.read_text(encoding="utf-8"))
    dec = [json.loads(x) for x in dec_p.read_text(encoding="utf-8").splitlines() if x.strip()]
    ok_f = sum(t["fruit"] == d["fruit"] for t, d in zip(truth, dec))
    ok_q = sum(("reject" if t["label"] == "defect" else "accept") == d["action"] for t, d in zip(truth, dec))
    lines += ["## 3. Chế độ video (theo dõi ByteTrack + bỏ phiếu khi qua vạch)", "",
              "Video băng chuyền mô phỏng từ ảnh cắt tập TEST (`scripts/make_demo_video.py`), chạy "
              "`nhandienqua predict runs/demo/bang_chuyen_test.mp4 --line 0.6`.", "",
              f"- Số quả: {len(truth)} · số quyết định: {len(dec)} (mỗi quả đúng 1 lần gạt/nhận)",
              f"- Đúng loại quả: {ok_f}/{len(truth)} · đúng tốt/hư: {ok_q}/{len(truth)}", ""]
    cons = VISION_ROOT / "runs/eval/consistency.json"
    if cons.exists():
        c = json.loads(cons.read_text(encoding="utf-8"))
        lines += ["**Độ nhất quán giữa các khung** (`scripts/consistency.py`; cùng một quả qua nhiều khung):", "",
                  "| Chế độ | Đổi loại quả | Đổi tốt/hư giữa 2 khung liền | Khung lệch đa số | Track có lúc đổi |",
                  "|---|---:|---:|---:|---:|"]
        for mode, r in c.items():
            if isinstance(r, dict):
                lines.append(f"| {mode} | {r['fruit_flip']:.2%} | {r['action_flip']:.2%} | {r['off_majority']:.2%} | "
                             f"{r['unstable_tracks']}/{r['tracks']} |")
        lines += ["", "Quyết định gạt dùng trung bình cả track (không phụ thuộc làm mượt hiển thị).", ""]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.chdir(WEIGHTS_DIR)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    lines = ["# Kết quả model", "",
             f"Sinh tự động bởi `vision/scripts/evaluate.py` lúc {datetime.now():%Y-%m-%d %H:%M} trên {gpu}. "
             "Không sửa tay số liệu — chạy lại script.", "",
             "> **Cảnh báo domain gap:** dữ liệu là ảnh public (điện thoại, vườn, phòng lab) và ảnh tổng hợp. "
             "Trên băng chuyền thật (ánh sáng, camera, giống quả Việt Nam khác) độ chính xác sẽ THẤP HƠN số dưới đây. "
             "Muốn dùng thật cần chụp vài trăm ảnh/loại trên máy và fine-tune.", ""]
    if DET.exists():
        eval_detect(lines)
    else:
        lines += ["## 1. Phát hiện quả", "", "Chưa có `weights/fruit_det.pt`.", ""]
    if QUAL.exists():
        eval_quality(lines)
    else:
        lines += ["## 2. Phân loại tốt / hư", "", "Chưa có `weights/fruit_quality.pt`.", ""]
    video_section(lines)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n→ {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
