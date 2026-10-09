"""Tạo video băng chuyền mô phỏng (quả trôi trái → phải) từ ảnh cắt tập TEST để thử chế độ video + bỏ phiếu theo track.

    python scripts/make_demo_video.py                 # → runs/demo/bang_chuyen_test.mp4 + truth.json
    nhandienqua predict runs/demo/bang_chuyen_test.mp4 --line 0.6

truth.json ghi loại quả + tốt/hư thật của từng quả theo thứ tự đi qua vạch → đối chiếu với decisions.jsonl.
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np
from PIL import Image

from nhandienqua import VISION_ROOT
from nhandienqua.data.compose import _augment_obj, procedural_belt
from nhandienqua.data.synth import paste_objects

CUT = VISION_ROOT / "data" / "cutouts"
OUT = VISION_ROOT / "runs" / "demo"
W, H, FPS, SPEED = 960, 540, 30, 9  # khung 16:9, băng chạy 9 px/khung
SIZE = {"mango": 120, "banana": 160, "dragon_fruit": 125, "jackfruit": 230, "sweet_potato": 150}


def main(n_fruits: int = 24) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rng = np.random.default_rng(7)
    rows = [json.loads(x) for x in open(CUT / "manifest.jsonl", encoding="utf-8")]
    pool = [r for r in rows if r["split"] == "test" and not r["touches_border"]]
    picks = [pool[i] for i in rng.choice(len(pool), n_fruits, replace=False)]
    gap = 300
    strip_w = W + gap * n_fruits + W
    strip = procedural_belt(H, strip_w, rng, kind="rubber")
    specs, truth = [], []
    # quả đầu tiên nằm ở mép PHẢI dải (đi qua vạch trước) — dải trượt sang phải theo thời gian
    for i, r in enumerate(picks):
        obj = _augment_obj(np.asarray(Image.open(CUT / r["path"]).convert("RGBA")), rng, SIZE[r["fruit"]])
        h, w = obj.shape[:2]
        x = strip_w - W - i * gap - w // 2
        y = int(rng.integers(40, H - h - 40)) if H - h - 80 > 0 else 0
        specs.append((obj, 0, (x, y)))
        truth.append({"order": i, "fruit": r["fruit"], "label": r["label"], "source": r["source"]})
    strip, _ = paste_objects(strip, specs, min_visible=0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bang_chuyen_test.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    # cửa sổ nhìn bắt đầu ở mép phải dải rồi lùi dần sang trái → nội dung trôi trái → phải
    x0 = strip_w - W
    n = 0
    while x0 >= 0:
        vw.write(strip[:, x0 : x0 + W, ::-1])
        x0 -= SPEED
        n += 1
    vw.release()
    (OUT / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{path} · {n} khung · {n_fruits} quả (tập test) · truth.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
