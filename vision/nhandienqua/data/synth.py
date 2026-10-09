"""Ghép ảnh cắt (RGBA) lên nền băng chuyền, sinh bbox theo phần NHÌN THẤY của từng vật."""

from __future__ import annotations

import numpy as np

from .boxes import Box, mask_to_bbox

PasteSpec = tuple[np.ndarray, int, tuple[int, int]]  # (cutout RGBA, lớp, (x, y) góc trên-trái)


def paste_objects(bg: np.ndarray, objs: list[PasteSpec], min_visible: float = 0.4) -> tuple[np.ndarray, list[tuple[int, Box]]]:
    """Dán lần lượt; vật dán sau che vật dán trước.

    Bbox tính từ pixel còn nhìn thấy. Bỏ nhãn vật bị che quá (tỉ lệ nhìn thấy < `min_visible`,
    so với phần nằm trong ảnh — vật chỉ bị cắt mép ảnh vẫn giữ).
    """
    img = bg.astype(np.float32).copy()
    H, W = bg.shape[:2]
    owner = np.full((H, W), -1, np.int32)
    in_image_area = []
    for idx, (rgba, _cls, (x, y)) in enumerate(objs):
        h, w = rgba.shape[:2]
        x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 >= x2 or y1 >= y2:
            in_image_area.append(0)
            continue
        crop = rgba[y1 - y : y2 - y, x1 - x : x2 - x]
        a = crop[..., 3:4].astype(np.float32) / 255.0
        img[y1:y2, x1:x2] = img[y1:y2, x1:x2] * (1 - a) + crop[..., :3].astype(np.float32) * a
        solid = crop[..., 3] > 127
        owner[y1:y2, x1:x2][solid] = idx
        in_image_area.append(int(solid.sum()))

    boxes: list[tuple[int, Box]] = []
    for idx, (_rgba, cls, _pos) in enumerate(objs):
        if in_image_area[idx] == 0:
            continue
        vis = owner == idx
        if vis.sum() / in_image_area[idx] < min_visible:
            continue
        box = mask_to_bbox(vis)
        if box is not None:
            boxes.append((cls, box))
    return np.clip(img, 0, 255).astype(np.uint8), boxes
