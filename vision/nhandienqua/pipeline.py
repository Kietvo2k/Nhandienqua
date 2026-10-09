"""Pipeline chạy thật: YOLO phát hiện quả → cắt ảnh → model chất lượng (chỉ xét lớp của loại quả đó) → quyết định.

Detector/classifier là hàm gọi được (adapter Ultralytics ở `nhandienqua.models`) → test được bằng model giả.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np

Box = tuple[int, int, int, int]
# detector(frame) -> [(box, class_index, conf, track_id|None)]
Detector = Callable[[np.ndarray], Sequence[tuple[Box, int, float, int | None]]]
# classifier([crop, ...]) -> mảng xác suất (N, số lớp chất lượng)
Classifier = Callable[[list[np.ndarray]], np.ndarray]


@dataclass
class Decision:
    fruit: str
    box: Box
    det_conf: float
    quality: str  # good | defect
    p_defect: float
    action: str  # accept | reject
    track_id: int | None = None
    partial: bool = False  # quả chạm mép khung (đang vào/ra) → ảnh cắt thiếu, kết quả kém tin cậy

    def to_dict(self) -> dict:
        d = asdict(self)
        d["box"] = list(self.box)
        d["det_conf"] = round(self.det_conf, 4)
        d["p_defect"] = round(self.p_defect, 4)
        return d


def decide_quality(probs: np.ndarray, quality_names: Sequence[str], fruit: str, threshold: float) -> tuple[str, float]:
    """Chỉ xét các lớp `<fruit>__*`, chuẩn hoá lại; p_defect = tổng xác suất các lớp không phải `__good`.

    Loại quả chưa có lớp chất lượng → trả 'defect' (an toàn: loại ra cho người kiểm tra).
    """
    idx = [i for i, n in enumerate(quality_names) if n.startswith(f"{fruit}__")]
    total = float(np.sum(probs[idx])) if idx else 0.0
    if total <= 0:
        return "defect", 1.0
    p_defect = float(sum(probs[i] for i in idx if not quality_names[i].endswith("__good"))) / total
    return ("defect" if p_defect >= threshold else "good"), p_defect


def crop_with_pad(img: np.ndarray, box: Box, pad: float = 0.1) -> tuple[np.ndarray, Box]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    b = (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))
    return img[b[1] : b[3], b[0] : b[2]], b


class FruitInspector:
    def __init__(
        self,
        detector: Detector,
        classifier: Classifier,
        fruit_names: Sequence[str],
        quality_names: Sequence[str],
        det_conf: float = 0.35,
        defect_threshold: float = 0.5,
        pad: float = 0.1,
    ):
        self.detector = detector
        self.classifier = classifier
        self.fruit_names = list(fruit_names)
        self.quality_names = list(quality_names)
        self.det_conf = det_conf
        self.defect_threshold = defect_threshold
        self.pad = pad

    def __call__(self, frame: np.ndarray) -> list[Decision]:
        dets = [d for d in self.detector(frame) if d[2] >= self.det_conf]
        crops = [crop_with_pad(frame, tuple(int(v) for v in d[0]), self.pad)[0] for d in dets]
        crops_ok = [c for c in crops if c.size]
        probs = self.classifier(crops_ok) if crops_ok else np.zeros((0, len(self.quality_names)))
        out: list[Decision] = []
        H, W = frame.shape[:2]
        k = 0
        for (box, cls, conf, tid), crop in zip(dets, crops):
            fruit = self.fruit_names[int(cls)]
            if crop.size:
                quality, p = decide_quality(probs[k], self.quality_names, fruit, self.defect_threshold)
                k += 1
            else:
                quality, p = "defect", 1.0
            b = tuple(int(v) for v in box)
            partial = b[0] <= 1 or b[1] <= 1 or b[2] >= W - 1 or b[3] >= H - 1
            out.append(Decision(fruit, b, float(conf), quality, p, "reject" if quality == "defect" else "accept", tid, partial))
        return out
