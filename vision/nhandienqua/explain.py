"""Bản đồ nhiệt giải thích model.

- CAM cho model tốt/hư (YOLO-cls): đầu Classify = conv → trung bình toàn cục → Linear, nên áp Linear lên TỪNG ô
  feature map cho bản đồ điểm lớp CHÍNH XÁC (trung bình các ô = logit của model) — không cần gradient, xuất ONNX được.
- Bản đồ phát hiện: điểm lớp cao nhất của từng anchor trên các lưới stride 8/16/32 của detector.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CamClassifier(nn.Module):
    """Bọc model phân loại Ultralytics: trả (probs (B,nc), cams (B,nc,h,w))."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.layers = model.model

    def _backbone(self, x: torch.Tensor):
        y = []
        for m in self.layers[:-1]:
            f = getattr(m, "f", -1)
            if f != -1:  # lớp nhận đầu vào từ lớp khác (như _predict_once của Ultralytics)
                x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]
            x = m(x)
            y.append(x)
        return x

    def forward(self, x: torch.Tensor):
        head = self.layers[-1]
        feat = self._backbone(x)
        if isinstance(feat, list):
            feat = torch.cat(feat, 1)
        feat = head.conv(feat)
        w, b = head.linear.weight, head.linear.bias
        cams = torch.einsum("kc,bchw->bkhw", w, feat) + b[None, :, None, None]
        return cams.mean((2, 3)).softmax(1), cams


def defect_map(cams: np.ndarray, names: list[str], fruit: str) -> np.ndarray | None:
    """Bằng chứng "hư" theo vị trí cho đúng loại quả: TB các lớp hư − lớp tốt (dương = nghiêng về hư)."""
    bad = [i for i, n in enumerate(names) if n.startswith(f"{fruit}__") and not n.endswith("__good")]
    good = [i for i, n in enumerate(names) if n == f"{fruit}__good"]
    if not bad or not good:
        return None
    return cams[bad].mean(0) - cams[good[0]]


def detection_heatmap(out: np.ndarray, grids: list[tuple[int, int]] | None = None, imgsz: int = 640) -> np.ndarray:
    """Đầu ra detector (1, 4+nc, N) → bản đồ điểm lớp cao nhất trên lưới mịn nhất (stride 8)."""
    import cv2

    grids = grids or [(imgsz // s, imgsz // s) for s in (8, 16, 32)]
    score = out[0, 4:].max(0)
    gh0, gw0 = grids[0]
    heat = np.zeros((gh0, gw0), np.float32)
    k = 0
    for gh, gw in grids:
        g = score[k : k + gh * gw].reshape(gh, gw).astype(np.float32)
        k += gh * gw
        heat = np.maximum(heat, cv2.resize(g, (gw0, gh0), interpolation=cv2.INTER_LINEAR) if (gh, gw) != (gh0, gw0) else g)
    return heat


def letterbox(img: np.ndarray, size: int = 640):
    """Như Ultralytics: giữ tỉ lệ, pad màu 114 hai bên. Trả (ảnh size×size, scale, (padX, padY))."""
    import cv2

    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = round(w * s), round(h * s)
    px, py = round((size - nw) / 2 - 0.1), round((size - nh) / 2 - 0.1)
    out = np.full((size, size, 3), 114, np.uint8)
    out[py : py + nh, px : px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, s, (max(0, px), max(0, py))


def overlay(img: np.ndarray, heat01: np.ndarray, alpha) -> np.ndarray:
    """Phủ bản đồ (0..1) lên ảnh với độ trong suốt THEO TỪNG ĐIỂM (`alpha(v)`) — vùng 0 không bị tô."""
    import cv2

    h, w = img.shape[:2]
    v = cv2.resize(heat01.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    col = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    a = np.clip(alpha(v), 0, 1)[..., None]
    return (img.astype(np.float32) * (1 - a) + col * a).astype(np.uint8)


def colorize(heat: np.ndarray, size: tuple[int, int], vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Chuẩn hoá về 0..1 rồi tô màu JET (BGR) ở kích thước (w, h)."""
    import cv2

    lo = heat.min() if vmin is None else vmin
    hi = heat.max() if vmax is None else vmax
    n = np.clip((heat - lo) / max(hi - lo, 1e-6), 0, 1)
    n = cv2.resize(n.astype(np.float32), size, interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_JET)
