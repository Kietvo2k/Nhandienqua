"""Dựng dataset phát hiện quả (YOLO) = ảnh tổng hợp trên băng chuyền + ảnh thật có bbox.

data/detect/images/{train,val,test_real,test_synth}/  +  labels/…  +  data.yaml, data_synth_test.yaml

- Tổng hợp: dán ảnh cắt (cả quả tốt lẫn hư) lên nền băng chuyền; bbox theo phần nhìn thấy.
- Thật: fruits10 (xoài/chuối/mít), lakshay (chuối), LVIS (chuối, khoai lang). Ảnh không chứa quả mục tiêu
  được giữ một phần làm mẫu âm (dạy model không báo nhầm quả khác).
- test_real CHỈ gồm ảnh thật → số liệu trung thực; test_synth đo trên ảnh tổng hợp.

    python scripts/build_detect_ds.py [--n-train 6000]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from nhandienqua import VISION_ROOT
from nhandienqua.data.backgrounds import load_backgrounds
from nhandienqua.data.boxes import yolo_line
from nhandienqua.data.compose import compose_detect_image
from nhandienqua.data.sources import RAW, fruits10_class_from_name
from nhandienqua.data.split import group_split

OUT = VISION_ROOT / "data" / "detect"
CUT = VISION_ROOT / "data" / "cutouts"
NAMES: list[str] = yaml.safe_load((VISION_ROOT / "configs" / "classes.yaml").read_text(encoding="utf-8"))["detect"]
CID = {n: i for i, n in enumerate(NAMES)}
# Cỡ quả (cạnh dài, px) trong khung 640 khi camera nhìn băng rộng ~1 m
SCALE_PX = {CID["mango"]: (70, 140), CID["banana"]: (90, 190), CID["dragon_fruit"]: (75, 145),
            CID["jackfruit"]: (170, 320), CID["sweet_potato"]: (80, 175)}


def write_sample(split: str, name: str, img_bgr_or_path, lines: list[str]) -> None:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    if isinstance(img_bgr_or_path, Path):
        im = Image.open(img_bgr_or_path).convert("RGB")
        im.save(OUT / "images" / split / f"{name}.jpg", quality=92)
    else:
        cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), img_bgr_or_path, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_yolo(path: Path) -> list[tuple[int, list[str]]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        parts = line.split()
        if len(parts) == 5:
            out.append((int(float(parts[0])), parts[1:]))
    return out


# ------------------------------------------------------------------ ảnh thật
def real_samples(rng: np.random.Generator):
    """Sinh (khoá, ảnh, [dòng nhãn YOLO]) với lớp đã ánh xạ sang lớp dự án."""
    # fruits10: lớp nằm trong tên file, id trong file luôn là 0
    neg = 0
    for img in sorted((RAW / "kg_fruits10_yolo").rglob("*.jpg")):
        fruit = fruits10_class_from_name(img.name)
        rows = read_yolo(img.with_suffix(".txt"))
        if fruit is None:
            if neg >= 150:
                continue
            neg += 1
            yield f"f10_{img.stem}", img, []
        elif rows:
            yield f"f10_{img.stem}", img, [f"{CID[fruit]} {' '.join(r)}" for _, r in rows]

    # lakshay: [Apple, Banana, Grape, Orange, Pineapple, Watermelon] → chỉ Banana
    base = RAW / "kg_lakshay_fruits" / "Fruits-detection"
    neg = 0
    imgs = sorted(p for p in base.rglob("images/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng.shuffle(imgs)
    for img in imgs:
        rows = read_yolo(img.parent.parent / "labels" / f"{img.stem}.txt")
        keep = [f"{CID['banana']} {' '.join(r)}" for c, r in rows if c == 1]
        if keep:
            yield f"lk_{img.stem}", img, keep
        elif neg < 600:
            neg += 1
            yield f"lk_{img.stem}", img, []

    # LVIS: 6 banana, 58 sweet potato — chỉ giữ ảnh có quả mục tiêu (nhãn LVIS không đầy đủ → không dùng làm mẫu âm)
    base = RAW / "kg_lvis_fruits_veg" / "LVIS_Fruits_And_Vegetables"
    for img in sorted((base / "images").rglob("*.jpg")):
        rows = read_yolo(base / "labels" / img.parent.name / f"{img.stem}.txt")
        keep = [f"{CID['banana'] if c == 6 else CID['sweet_potato']} {' '.join(r)}" for c, r in rows if c in (6, 58)]
        if keep:
            yield f"lv_{img.stem}", img, keep


MASKED_SOURCES = ("hf_fruitseg30", "gh_sweet_potato_defect")


def append_masked_real(stats: Counter) -> None:
    """Ảnh chụp THẬT có mask do người gán (fruitseg30, khoai lang GitHub) → bbox; theo đúng tập đã chia ở kho ảnh cắt.

    Bổ sung ảnh thật cho thanh long / khoai lang / xoài (các nguồn bbox public gần như không có).
    """
    from nhandienqua.data.boxes import mask_to_bbox
    from nhandienqua.data.sources import iter_quality_records

    split_of = {}
    for line in open(CUT / "manifest.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["source"] in MASKED_SOURCES:
            split_of[(r["source"], r["group"])] = r["split"]
    recs = [r for r in iter_quality_records() if r.source in MASKED_SOURCES and (r.source, r.group) in split_of]
    for rec in tqdm(recs, desc="ảnh thật có mask"):
        sp = split_of[(rec.source, rec.group)]
        sp = "test_real" if sp == "test" else sp
        img = rec.load()
        box = mask_to_bbox(rec.mask(img))
        if box is None:
            continue
        h, w = img.shape[:2]
        s = min(1.0, 1280 / max(h, w))  # ảnh khoai lang 6000×4000 → thu nhỏ
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            box = tuple(int(round(v * s)) for v in box)
            h, w = img.shape[:2]
        key = f"mask_{rec.source}_{rec.key}".replace("/", "_")
        write_sample(sp, key, img[..., ::-1].copy(), [yolo_line(CID[rec.fruit], box, w, h)])
        stats[(sp, "ảnh thật có mask")] += 1
        stats[(sp, rec.fruit)] += 1


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--append-masked", action="store_true", help="chỉ thêm ảnh thật có mask vào dataset đã dựng")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    stats: Counter = Counter()
    if args.append_masked:
        append_masked_real(stats)
        print(", ".join(f"{s}:{k}={v}" for (s, k), v in sorted(stats.items())))
        return 0
    if OUT.exists():
        shutil.rmtree(OUT)

    # ---- ảnh thật
    real = list(real_samples(rng))
    splits = group_split([k for k, _, _ in real], val=0.15, test=0.15, seed=1)
    for (key, img, lines), sp in tqdm(list(zip(real, splits)), desc="ảnh thật"):
        sp = "test_real" if sp == "test" else sp
        write_sample(sp, key, img, lines)
        stats[(sp, "ảnh thật")] += 1
        for line in lines:
            stats[(sp, NAMES[int(line.split()[0])])] += 1

    append_masked_real(stats)

    # ---- ảnh tổng hợp
    rows = [json.loads(line) for line in open(CUT / "manifest.jsonl", encoding="utf-8")]
    pools: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["touches_border"]:
            continue  # quả bị mép ảnh cắt → ghép trông giả
        pools[r["split"]][CID[r["fruit"]]].append(np.asarray(Image.open(CUT / r["path"]).convert("RGBA")))
    for sp in pools:
        print(f"Kho ảnh cắt {sp}: " + ", ".join(f"{NAMES[c]}={len(v)}" for c, v in sorted(pools[sp].items())))
    bgs = load_backgrounds(rng)
    for sp, n in (("train", args.n_train), ("val", args.n_eval), ("test", args.n_eval)):
        out_sp = "test_synth" if sp == "test" else sp
        for i in tqdm(range(n), desc=f"tổng hợp {out_sp}"):
            img, boxes = compose_detect_image(pools[sp], bgs, rng, size=640, n_objects=(2, 9), scale_px=SCALE_PX)
            write_sample(out_sp, f"syn_{sp}_{i:05d}", img[..., ::-1], [yolo_line(c, b, 640, 640) for c, b in boxes])
            stats[(out_sp, "ảnh tổng hợp")] += 1
            for c, _ in boxes:
                stats[(out_sp, NAMES[c])] += 1

    common = {"path": str(OUT).replace("\\", "/"), "names": dict(enumerate(NAMES))}
    (OUT / "data.yaml").write_text(yaml.safe_dump({**common, "train": "images/train", "val": "images/val", "test": "images/test_real"}, allow_unicode=True), encoding="utf-8")
    (OUT / "data_synth_test.yaml").write_text(yaml.safe_dump({**common, "train": "images/train", "val": "images/test_synth"}, allow_unicode=True), encoding="utf-8")
    (OUT / "data_real_test.yaml").write_text(yaml.safe_dump({**common, "train": "images/train", "val": "images/test_real"}, allow_unicode=True), encoding="utf-8")

    print("\nThống kê (ảnh / số bbox theo lớp):")
    for sp in ("train", "val", "test_real", "test_synth"):
        print(f"  {sp}: " + ", ".join(f"{k}={v}" for (s, k), v in sorted(stats.items()) if s == sp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
