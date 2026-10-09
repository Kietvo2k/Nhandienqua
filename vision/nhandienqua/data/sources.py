"""Bộ chuyển đổi cho từng dataset nguồn: ánh xạ nhãn, liệt kê bản ghi, tách mask đơn giản."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .. import VISION_ROOT

RAW = VISION_ROOT / "data" / "raw"
IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".bmp", ".webp"}

# ------------------------------------------------------------------ ánh xạ nhãn chất lượng
_GOOD = "good"
_BAD = "defect"

_QUALITY_MAP: dict[str, Callable[[object], str]] = {
    "kg_mango_disease": lambda r: _GOOD if str(r).lower() == "healthy" else _BAD,
    "hf_banana_guava_quality": lambda r: _GOOD if int(r) in (0, 1) else _BAD,  # Class_A, Class_B | Defect
    "hf_dragonfruit_quality": lambda r: _GOOD if int(r) == 1 else _BAD,  # 1 Fresh | 0 Defect
    "kg_dragonfruit_vn": lambda r: _GOOD if str(r).lower() == "healthy" else _BAD,
    "kg_jackfruit_shapes": lambda r: _GOOD if str(r).lower() == "perfect" else _BAD,
    "gh_sweet_potato_defect": lambda r: _BAD,  # chỉ có ảnh lỗi
    "hf_fruits360": lambda r: _GOOD,
    "hf_fruitseg30": lambda r: _GOOD,
    "kg_kritik_sweetpotato": lambda r: _GOOD,
}


def quality_label(source: str, raw: object) -> str:
    return _QUALITY_MAP[source](raw)


# ------------------------------------------------------------------ fruits10 (nhãn lớp nằm trong tên file)
_FRUITS10 = {"mango": "mango", "banana": "banana", "jackfruit": "jackfruit"}


def fruits10_class_from_name(name: str) -> str | None:
    word = re.sub(r"^\d+_\s*", "", Path(name).stem).strip().lower()
    return _FRUITS10.get(word)


# ------------------------------------------------------------------ tách mask theo ngưỡng
def threshold_mask(img: np.ndarray, background: str, erode: int = 0) -> np.ndarray:
    """Mask vật thể trên nền trắng (Fruits-360) hoặc nền đen (bảng LED khoai lang)."""
    import cv2

    g = img.astype(np.int16)
    if background == "white":
        m = (g.min(axis=2) < 235) | (g.max(axis=2) - g.min(axis=2) > 25)
    elif background == "black":
        m = g.max(axis=2) > 45
    else:
        raise ValueError(background)
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(m.shape, bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = lab == biggest
    # lấp lỗ bên trong vật
    filled = keep.astype(np.uint8)
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    if erode:  # bỏ viền sáng/tối còn sót quanh mép quả
        filled = cv2.erode(filled, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    return filled.astype(bool)


_ROT = re.compile(r"_rot(\d+)$", re.I)


def is_original_copy(name: str) -> bool:
    """Bỏ các bản xoay sẵn (`_rot90`…) để không nhân bản ảnh gốc."""
    m = _ROT.search(Path(name).stem)
    return m is None or m.group(1) == "0"


# ------------------------------------------------------------------ bản ghi chất lượng
@dataclass
class Record:
    source: str
    fruit: str
    label: str
    key: str  # định danh duy nhất trong nguồn
    group: str  # khoá nhóm để chia tập
    load: Callable[[], np.ndarray]  # trả ảnh RGB uint8
    mask: Callable[[np.ndarray], np.ndarray | None] | None = None  # mask có sẵn (nếu nguồn cung cấp)


def _open_rgb(path_or_bytes) -> np.ndarray:
    from PIL import Image, ImageOps

    if isinstance(path_or_bytes, (bytes, bytearray)):
        im = Image.open(io.BytesIO(path_or_bytes))
    else:
        if str(path_or_bytes).lower().endswith(".heic"):
            import pillow_heif

            pillow_heif.register_heif_opener()
        im = Image.open(path_or_bytes)
    im = ImageOps.exif_transpose(im)
    return np.asarray(im.convert("RGB"))


def _files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def _parquet_rows(path: Path, columns: list[str]):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64, columns=columns):
        yield from batch.to_pylist()


def iter_quality_records() -> Iterator[Record]:
    from .split import group_key_from_name

    # Xoài bệnh (MangoFruitDDS dẫn xuất) — nhóm theo ảnh gốc trong tên file
    src = "kg_mango_disease"
    for p in _files(RAW / src):
        raw = p.parent.name
        yield Record(src, "mango", quality_label(src, raw), p.name, f"{raw}/{group_key_from_name(p.name)}", lambda p=p: _open_rgb(p))

    # Chuối A/B/Defect
    src = "hf_banana_guava_quality"
    for i, r in enumerate(_parquet_rows(RAW / src / "data/train-00000-of-00001.parquet", ["image", "label", "crop_type"])):
        if r["crop_type"] != "banana":
            continue
        b = r["image"]["bytes"]
        yield Record(src, "banana", quality_label(src, r["label"]), str(i), f"{src}/{i}", lambda b=b: _open_rgb(b))

    # Thanh long tươi / lỗi (Mendeley 2jpzbx8tm6)
    src = "hf_dragonfruit_quality"
    for i, r in enumerate(_parquet_rows(RAW / src / "raw/train-00000-of-00001.parquet", ["image", "label"])):
        b = r["image"]["bytes"]
        yield Record(src, "dragon_fruit", quality_label(src, r["label"]), str(i), f"{src}/{i}", lambda b=b: _open_rgb(b))

    # Thanh long VN — chỉ phần quả, bỏ bản xoay sẵn
    src = "kg_dragonfruit_vn"
    for p in _files(RAW / src / "oversample" / "Fruit"):
        if not is_original_copy(p.name):
            continue
        raw = p.parent.name
        yield Record(src, "dragon_fruit", quality_label(src, raw), f"{raw}/{p.name}", f"{src}/{raw}/{group_key_from_name(p.name)}", lambda p=p: _open_rgb(p))

    # Mít perfect / deformed / malformed
    src = "kg_jackfruit_shapes"
    for p in _files(RAW / src / "jackfruit_dataset"):
        raw = p.parent.name
        yield Record(src, "jackfruit", quality_label(src, raw), f"{raw}/{p.name}", f"{src}/{raw}/{p.stem.lower()}", lambda p=p: _open_rgb(p))

    # Khoai lang lỗi — mask từ ảnh nhãn (vùng khác xanh dương), không có nhãn thì tách nền đen
    src = "gh_sweet_potato_defect"
    base = RAW / src / "sweet_potato_defect_dataset-main"
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name.endswith("_label"):
            continue
        label_dir = base / f"{d.name[0]}_label"
        for p in _files(d):
            lp = label_dir / p.name

            def mask(img, lp=lp):
                if lp.exists():
                    lb = _open_rgb(lp)
                    return ~((lb[..., 2] > 200) & (lb[..., 0] < 50) & (lb[..., 1] < 50))
                return threshold_mask(img, "black")

            yield Record(src, "sweet_potato", quality_label(src, d.name), p.name, f"{src}/{p.stem.lower()}", lambda p=p: _open_rgb(p), mask)

    # Fruits-360 (nền trắng): khoai lang, xoài, chuối, thanh long — mỗi lớp là vài quả chụp xoay vòng
    src = "hf_fruits360"
    want = {94: "sweet_potato", 53: "mango", 11: "banana", 88: "dragon_fruit"}
    for part in sorted((RAW / src / "data").glob("*.parquet")):
        counters: dict[int, int] = {}
        for r in _parquet_rows(part, ["image", "label"]):
            lab = r["label"]
            if lab not in want:
                continue
            j = counters[lab] = counters.get(lab, -1) + 1
            if j % 6:  # khung hình liền nhau gần như trùng → lấy 1/6
                continue
            b = r["image"]["bytes"]
            # 60 khung liên tiếp ≈ một vòng xoay của cùng một quả → cùng nhóm
            yield Record(src, want[lab], "good", f"{part.stem}/{lab}/{j}", f"{src}/{part.stem}/{lab}/{j // 60}",
                         lambda b=b: _open_rgb(b), lambda img: threshold_mask(img, "white", erode=2))

    # Khoai lang lành từ ảnh web (kritikseth, CC0) — tăng đa dạng cho lớp sweet_potato__good
    src = "kg_kritik_sweetpotato"
    for p in _files(RAW / src):
        yield Record(src, "sweet_potato", "good", p.name, f"{src}/{p.stem.lower()}", lambda p=p: _open_rgb(p))

    # FruitSeg30: có mask sẵn
    src = "hf_fruitseg30"
    fmap = {"banana": "banana", "dragon": "dragon_fruit"}
    for i, r in enumerate(_parquet_rows(RAW / src / "data/train-00000-of-00001.parquet", ["image", "mask", "crop_type"])):
        ct = r["crop_type"].lower()
        fruit = "mango" if ct.startswith("mango") else fmap.get(ct)
        if fruit is None:
            continue
        b, mb = r["image"]["bytes"], r["mask"]["bytes"]
        yield Record(src, fruit, "good", str(i), f"{src}/{i}", lambda b=b: _open_rgb(b),
                     lambda img, mb=mb: _open_rgb(mb).max(axis=2) > 127)
