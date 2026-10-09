"""Tách quả khỏi nền ở mọi nguồn chất lượng → kho ảnh cắt RGBA dùng chung cho cả 2 model.

data/cutouts/<split>/<fruit>__<label>/<source>__<id>.png  +  data/cutouts/manifest.jsonl

Mask lấy theo thứ tự: mask có sẵn của nguồn → YOLOE (câu lệnh văn bản) → bỏ ảnh.
Chia train/val/test theo NHÓM ảnh gốc (xem nhandienqua.data.split).

    python scripts/build_cutouts.py                 # toàn bộ
    python scripts/build_cutouts.py --limit 40      # thử nhanh 40 ảnh/nguồn
    python scripts/build_cutouts.py --only kg_kritik_sweetpotato   # thêm/làm lại một nguồn, giữ phần còn lại
    python scripts/build_cutouts.py --resplit       # chỉ chia lại train/val/test (phân tầng theo lớp)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from nhandienqua import VISION_ROOT, WEIGHTS_DIR
from nhandienqua.data.boxes import mask_to_bbox
from nhandienqua.data.sources import iter_quality_records
from nhandienqua.data.split import stratified_group_split

OUT = VISION_ROOT / "data" / "cutouts"
MAX_SIDE = 384
PROMPTS = {
    "mango": ["mango", "fruit"],
    "banana": ["banana", "fruit"],
    "dragon_fruit": ["dragon fruit", "pitaya", "fruit"],
    "jackfruit": ["jackfruit", "durian", "fruit"],
    "sweet_potato": ["sweet potato", "potato", "fruit"],
}


class YoloeSegmenter:
    def __init__(self, conf: float = 0.25):
        from nhandienqua.yolo import YOLO  # noqa: F401  (cấu hình thư mục trước)
        from ultralytics import YOLOE

        self.model = YOLOE(str(WEIGHTS_DIR / "yoloe-26s-seg.pt"))
        self.conf = conf
        self.cur: tuple[str, ...] | None = None

    def __call__(self, img: np.ndarray, fruit: str) -> np.ndarray | None:
        prompts = PROMPTS[fruit]
        if self.cur != tuple(prompts):
            self.model.set_classes(prompts, self.model.get_text_pe(prompts))
            self.cur = tuple(prompts)
        r = self.model.predict(img[..., ::-1], conf=self.conf, verbose=False)[0]  # ultralytics nhận BGR
        if r.masks is None or len(r.boxes) == 0:
            return None
        conf = r.boxes.conf.cpu().numpy()
        best = float(conf.max())
        # Chuối thường theo nải → gộp các quả tự tin; quả khác lấy quả tự tin nhất
        idx = np.nonzero(conf >= max(0.2, 0.5 * best))[0] if fruit == "banana" else [int(conf.argmax())]
        mask = np.zeros(img.shape[:2], np.uint8)
        for i in idx:
            poly = r.masks.xy[i]
            if len(poly) >= 3:
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        return mask.astype(bool) if mask.any() else None


def make_cutout(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, bool] | None:
    h, w = mask.shape
    if mask.sum() < 0.01 * h * w:
        return None
    x1, y1, x2, y2 = mask_to_bbox(mask)
    if mask.sum() / ((x2 - x1) * (y2 - y1)) > 0.95:  # mask gần như chữ nhật → tách hỏng, ăn cả nền
        return None
    m = int(0.03 * max(x2 - x1, y2 - y1))
    x1, y1, x2, y2 = max(0, x1 - m), max(0, y1 - m), min(w, x2 + m), min(h, y2 + m)
    # quả bị mép ảnh cắt > 3% chu vi → không dùng cho ảnh ghép detection
    edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    touches = edge.mean() > 0.03
    alpha = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (0, 0), 1.0)
    rgba = np.dstack([img, alpha])[y1:y2, x1:x2]
    s = MAX_SIDE / max(rgba.shape[:2])
    if s < 1:
        rgba = cv2.resize(rgba, (round(rgba.shape[1] * s), round(rgba.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return rgba, bool(touches)


MANIFEST = OUT / "manifest.jsonl"


def resplit(rows: list[dict]) -> list[dict]:
    """Chia train/val/test theo nhóm, PHÂN TẦNG theo lớp; di chuyển file sang thư mục tập mới."""
    # bỏ dòng trùng đường dẫn; tìm lại file nếu lần chạy trước dừng giữa chừng khi đang di chuyển
    uniq: dict[str, dict] = {}
    for r in rows:
        tail = r["path"][r["path"].index("/"):]
        if tail in uniq:
            continue
        if not (OUT / r["path"]).exists():
            found = next((f"{sp}{tail}" for sp in ("train", "val", "test", "new") if (OUT / f"{sp}{tail}").exists()), None)
            if found is None:
                continue
            r["path"] = found
        uniq[tail] = r
    rows = list(uniq.values())
    classes = [f"{r['fruit']}__{r['label']}" for r in rows]
    splits = stratified_group_split([r["group"] for r in rows], classes, val=0.15, test=0.15, seed=0)
    for r, sp in zip(rows, splits):
        new = sp + r["path"][r["path"].index("/"):]
        if new != r["path"]:
            (OUT / new).parent.mkdir(parents=True, exist_ok=True)
            (OUT / r["path"]).replace(OUT / new)
        r["path"], r["split"] = new, sp
    with open(MANIFEST, "w", encoding="utf-8") as mf:
        for r in rows:
            mf.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def report(rows: list[dict]) -> None:
    c = Counter((r["split"], f"{r['fruit']}__{r['label']}") for r in rows)
    print("Theo tập / lớp:")
    for (a, b), n in sorted(c.items()):
        print(f"  {a:<6} {b:<24} {n}")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="số ảnh tối đa mỗi nguồn (thử nhanh)")
    ap.add_argument("--only", nargs="*", help="chỉ xử lý các nguồn này, giữ kết quả các nguồn khác")
    ap.add_argument("--resplit", action="store_true", help="chỉ chia lại tập theo manifest hiện có")
    args = ap.parse_args(argv)

    if args.resplit:
        report(resplit([json.loads(x) for x in open(MANIFEST, encoding="utf-8")]))
        return 0

    recs = list(iter_quality_records())
    keep: list[dict] = []
    if args.only:
        recs = [r for r in recs if r.source in args.only]
        for r in (json.loads(x) for x in open(MANIFEST, encoding="utf-8")):
            if r["source"] in args.only:
                (OUT / r["path"]).unlink(missing_ok=True)
            else:
                keep.append(r)
    elif OUT.exists():
        shutil.rmtree(OUT)
    if args.limit:
        per = defaultdict(list)
        for r in recs:
            per[r.source].append(r)
        recs = [r for v in per.values() for r in v[:: max(1, len(v) // args.limit)][: args.limit]]

    seg = YoloeSegmenter()
    stats: Counter = Counter()
    new_rows: list[dict] = []
    for rec in tqdm(recs, desc="cutouts"):
        try:
            img = rec.load()
        except Exception as e:  # ảnh hỏng
            stats[(rec.source, "lỗi đọc")] += 1
            print(f"[bỏ] {rec.source}/{rec.key}: {e}", file=sys.stderr)
            continue
        mask, method = (rec.mask(img), "source") if rec.mask else (None, "")
        if mask is None:
            mask, method = seg(img, rec.fruit), "yoloe"
        res = make_cutout(img, mask) if mask is not None else None
        if res is None:
            stats[(rec.source, "không tách được")] += 1
            continue
        rgba, touches = res
        cls = f"{rec.fruit}__{rec.label}"
        safe = re.sub(r"[^\w.-]+", "_", rec.key)[:80]
        path = OUT / "new" / cls / f"{rec.source}__{safe}.png"  # resplit() chuyển vào train/val/test
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba).save(path)
        new_rows.append({
            "path": str(path.relative_to(OUT)).replace("\\", "/"), "split": "new", "fruit": rec.fruit,
            "label": rec.label, "source": rec.source, "group": rec.group, "method": method, "touches_border": touches,
        })
        stats[(rec.source, "ok")] += 1

    print("\nTheo nguồn:")
    for (a, b), n in sorted(stats.items()):
        print(f"  {a:<26} {b:<18} {n}")
    report(resplit(keep + new_rows))
    shutil.rmtree(OUT / "new", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
