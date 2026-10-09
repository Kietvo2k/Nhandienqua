"""Dựng ảnh huấn luyện từ kho ảnh cắt: nền băng chuyền, xoay, đổ bóng, nhiễu ánh sáng, mờ chuyển động."""

from __future__ import annotations

import cv2
import numpy as np

from .synth import paste_objects

Rng = np.random.Generator

_BELTS = {
    "rubber": ((38, 38, 40), 10),   # cao su đen/xám
    "green": ((44, 86, 58), 10),    # PVC xanh
    "white": ((206, 208, 204), 8),  # PU trắng (thực phẩm)
    "blue": ((58, 104, 150), 10),   # PU xanh dương
}


def procedural_belt(h: int, w: int, rng: Rng, kind: str | None = None) -> np.ndarray:
    kind = kind or rng.choice(list(_BELTS), p=[0.45, 0.15, 0.25, 0.15])
    base, noise = _BELTS[kind]
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    img *= rng.uniform(0.85, 1.15)
    img += rng.normal(0, noise, (h, w, 1))
    # vân ngang (mắt xích / mối nối băng) theo chiều chạy
    step = int(rng.integers(18, 60))
    for x in range(int(rng.integers(0, step)), w, step):
        img[:, x : x + 2] *= rng.uniform(0.8, 0.95)
    # ánh sáng không đều (đèn dải phía trên)
    yy = np.linspace(-1, 1, h)[:, None, None]
    xx = np.linspace(-1, 1, w)[None, :, None]
    img *= 1 - rng.uniform(0.0, 0.25) * (xx**2 * rng.uniform(0, 1) + yy**2)
    return np.clip(img, 0, 255).astype(np.uint8)


def rotate_rgba(rgba: np.ndarray, angle: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos + 0.5), int(h * cos + w * sin + 0.5)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(rgba, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _scale_to(rgba: np.ndarray, long_side: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    s = long_side / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(rgba, (max(2, round(w * s)), max(2, round(h * s))), interpolation=interp)


def _augment_obj(rgba: np.ndarray, rng: Rng, long_side: float) -> np.ndarray:
    o = rotate_rgba(rgba, float(rng.uniform(0, 360)))
    if rng.random() < 0.5:
        o = o[:, ::-1]
    o = _scale_to(o, long_side)
    rgb = o[..., :3].astype(np.float32) * rng.uniform(0.85, 1.15)
    o = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), o[..., 3]])
    return o


def _crop_bg(bgs: list[np.ndarray], rng: Rng, h: int, w: int) -> np.ndarray:
    bg = bgs[int(rng.integers(len(bgs)))]
    bh, bw = bg.shape[:2]
    s = max(h / bh, w / bw) * rng.uniform(1.0, 1.4)
    if s != 1:
        bg = cv2.resize(bg, (int(bw * s) + 1, int(bh * s) + 1), interpolation=cv2.INTER_LINEAR)
    bh, bw = bg.shape[:2]
    y, x = int(rng.integers(0, bh - h + 1)), int(rng.integers(0, bw - w + 1))
    out = bg[y : y + h, x : x + w]
    if rng.random() < 0.5:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def _shadow(img: np.ndarray, rgba: np.ndarray, x: int, y: int, rng: Rng) -> None:
    """Đổ bóng mềm lệch xuống dưới-phải (đèn phía trên)."""
    a = rgba[..., 3].astype(np.float32) / 255
    k = max(3, int(max(a.shape) * 0.12) | 1)
    a = cv2.GaussianBlur(a, (k, k), 0) * rng.uniform(0.25, 0.5)
    dx, dy = int(a.shape[1] * 0.06), int(a.shape[0] * 0.08)
    H, W = img.shape[:2]
    x1, y1 = max(0, x + dx), max(0, y + dy)
    x2, y2 = min(W, x + dx + a.shape[1]), min(H, y + dy + a.shape[0])
    if x1 >= x2 or y1 >= y2:
        return
    sub = a[y1 - y - dy : y2 - y - dy, x1 - x - dx : x2 - x - dx][..., None]
    img[y1:y2, x1:x2] = (img[y1:y2, x1:x2].astype(np.float32) * (1 - sub)).astype(np.uint8)


def _photometric(img: np.ndarray, rng: Rng, motion_p: float) -> np.ndarray:
    f = img.astype(np.float32)
    f = (f - 128) * rng.uniform(0.85, 1.15) + 128 + rng.uniform(-18, 18)
    f *= rng.uniform(0.94, 1.06, size=3)  # lệch cân bằng trắng nhẹ
    f += rng.normal(0, rng.uniform(0, 5), f.shape)
    out = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < motion_p:  # mờ chuyển động theo chiều băng chạy
        k = int(rng.integers(3, 10))
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1 / k
        out = cv2.filter2D(out, -1, kern)
    return out


def render_quality_sample(rgba: np.ndarray, bgs: list[np.ndarray], rng: Rng, out_size: int = 224) -> np.ndarray:
    """Một quả nằm giữa ảnh trên nền băng chuyền — mô phỏng ảnh cắt từ bbox của detector."""
    obj = _augment_obj(rgba, rng, out_size * rng.uniform(0.62, 0.85))
    img = _crop_bg(bgs, rng, out_size, out_size).copy()
    h, w = obj.shape[:2]
    j = int(out_size * 0.06)
    x = (out_size - w) // 2 + int(rng.integers(-j, j + 1))
    y = (out_size - h) // 2 + int(rng.integers(-j, j + 1))
    _shadow(img, obj, x, y, rng)
    img, _ = paste_objects(img, [(obj, 0, (x, y))], min_visible=0.0)
    return _photometric(img, rng, motion_p=0.15)


def compose_detect_image(
    pool: dict[int, list[np.ndarray]],
    bgs: list[np.ndarray],
    rng: Rng,
    size: int = 640,
    n_objects: tuple[int, int] = (2, 9),
    scale_px: dict[int, tuple[int, int]] | None = None,
    min_box: int = 10,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int, int, int]]]]:
    """Ảnh băng chuyền nhìn từ trên xuống với nhiều quả; nhãn theo phần nhìn thấy."""
    img = _crop_bg(bgs, rng, size, size).copy()
    classes = sorted(pool)
    specs = []
    for _ in range(int(rng.integers(n_objects[0], n_objects[1] + 1))):
        cls = classes[int(rng.integers(len(classes)))]
        lo, hi = (scale_px or {}).get(cls, (70, 160))
        obj = _augment_obj(pool[cls][int(rng.integers(len(pool[cls])))], rng, rng.uniform(lo, hi))
        h, w = obj.shape[:2]
        x = int(rng.integers(-w // 5, size - w + w // 5 + 1))
        y = int(rng.integers(-h // 5, size - h + h // 5 + 1))
        _shadow(img, obj, x, y, rng)
        specs.append((obj, cls, (x, y)))
    img, boxes = paste_objects(img, specs, min_visible=0.4)
    boxes = [(c, b) for c, b in boxes if b[2] - b[0] >= min_box and b[3] - b[1] >= min_box]
    return _photometric(img, rng, motion_p=0.25), boxes
