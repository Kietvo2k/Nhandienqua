"""Bbox dạng pixel (x1, y1, x2, y2) — x2/y2 loại trừ — và dạng YOLO chuẩn hoá (cx, cy, w, h)."""

from __future__ import annotations

import numpy as np

Box = tuple[int, int, int, int]


def mask_to_bbox(mask: np.ndarray) -> Box | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def to_yolo(box: Box, w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def from_yolo(y: tuple[float, float, float, float], w: int, h: int) -> tuple[float, float, float, float]:
    cx, cy, bw, bh = y
    return ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)


def yolo_line(cls: int, box: Box, w: int, h: int) -> str:
    return f"{cls} " + " ".join(f"{v:.6f}" for v in to_yolo(box, w, h))
