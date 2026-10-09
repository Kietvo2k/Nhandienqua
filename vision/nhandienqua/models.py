"""Adapter Ultralytics cho pipeline. Nạp được .pt (PyTorch), .onnx (ONNX Runtime) và .engine (TensorRT trên Jetson)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import WEIGHTS_DIR
from .yolo import YOLO

DEFAULT_DET = WEIGHTS_DIR / "fruit_det.pt"
DEFAULT_QUALITY = WEIGHTS_DIR / "fruit_quality.pt"


def _names(model) -> list[str]:
    n = model.names
    return [n[i] for i in sorted(n)] if isinstance(n, dict) else list(n)


class UltralyticsDetector:
    def __init__(self, weights: str | Path = DEFAULT_DET, conf: float = 0.25, imgsz: int = 640, device=None,
                 track: bool = False, tracker: str = "bytetrack.yaml"):
        self.model = YOLO(str(weights), task="detect")
        self.conf, self.imgsz, self.device, self.track, self.tracker = conf, imgsz, device, track, tracker
        self.names = _names(self.model)

    def __call__(self, frame: np.ndarray):
        kw = dict(conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)
        r = (self.model.track(frame, persist=True, tracker=self.tracker, **kw) if self.track else self.model.predict(frame, **kw))[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy = b.xyxy.cpu().numpy().astype(int)
        cls = b.cls.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()
        ids = b.id.cpu().numpy().astype(int) if b.id is not None else [None] * len(xyxy)
        return [(tuple(x), int(c), float(p), None if i is None else int(i)) for x, c, p, i in zip(xyxy, cls, conf, ids)]


def predict_in_batches(predict, crops: list[np.ndarray], batch: int | None) -> np.ndarray:
    """Model xuất tĩnh (.onnx/.engine, batch=1) không nhận nhiều ảnh một lần → chia lô đúng cỡ batch."""
    step = batch or len(crops)
    out = []
    for i in range(0, len(crops), step):
        out.extend(predict(crops[i : i + step]))
    return np.stack(out)


def tta_views(crops: list[np.ndarray]) -> list[np.ndarray]:
    """Mỗi ảnh cắt → [gốc, lật ngang, lật dọc] (quả nằm trên băng không có chiều trên/dưới cố định)."""
    return [v for c in crops for v in (c, np.ascontiguousarray(c[:, ::-1]), np.ascontiguousarray(c[::-1]))]


def tta_merge(probs: np.ndarray, n: int) -> np.ndarray:
    return probs.reshape(n, -1, probs.shape[-1]).mean(1)


class UltralyticsClassifier:
    def __init__(self, weights: str | Path = DEFAULT_QUALITY, imgsz: int = 224, device=None, tta: bool = False):
        self.model = YOLO(str(weights), task="classify")
        self.imgsz, self.device, self.tta = imgsz, device, tta
        self.names = _names(self.model)
        # .pt nhận batch tuỳ ý; ONNX/TensorRT xuất với batch cố định = 1 (scripts/export.py, jetson/run.sh)
        self.batch = None if Path(weights).suffix == ".pt" else 1

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        def predict(batch):
            rs = self.model.predict(batch, imgsz=self.imgsz, device=self.device, verbose=False)
            return [r.probs.data.cpu().numpy() for r in rs]

        if not self.tta:
            return predict_in_batches(predict, crops, self.batch)
        return tta_merge(predict_in_batches(predict, tta_views(crops), self.batch), len(crops))
