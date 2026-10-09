import numpy as np
import pytest

from nhandienqua.data.boxes import mask_to_bbox, to_yolo, from_yolo
from nhandienqua.data.split import group_split, group_key_from_name
from nhandienqua.data.synth import paste_objects


# ------------------------------------------------------------------ boxes
def test_mask_to_bbox():
    m = np.zeros((10, 20), bool)
    m[2:5, 3:9] = True
    assert mask_to_bbox(m) == (3, 2, 9, 5)  # x1, y1, x2, y2 (x2/y2 loại trừ)


def test_mask_to_bbox_empty_is_none():
    assert mask_to_bbox(np.zeros((4, 4), bool)) is None


def test_yolo_roundtrip():
    box = (10, 20, 50, 100)
    y = to_yolo(box, w=200, h=400)
    assert y == pytest.approx((0.15, 0.15, 0.2, 0.2))
    assert from_yolo(y, w=200, h=400) == pytest.approx(box)


# ------------------------------------------------------------------ split
def test_group_key_strips_augmentation_markers():
    assert group_key_from_name("aug_143_anthracnose_012.jpg") == "anthracnose_012"
    assert group_key_from_name("Brown Spot_002_rot180.jpeg") == "brown spot_002"
    assert group_key_from_name("IMG_1924.HEIC") == "img_1924"


def test_group_split_deterministic_and_no_leak():
    keys = [f"g{i // 4}" for i in range(400)]  # 100 nhóm × 4 ảnh
    a = group_split(keys, val=0.15, test=0.15, seed=0)
    b = group_split(keys, val=0.15, test=0.15, seed=0)
    assert a == b
    by_group = {}
    for k, s in zip(keys, a):
        by_group.setdefault(k, set()).add(s)
    assert all(len(v) == 1 for v in by_group.values())  # một nhóm chỉ nằm trong một tập
    frac = {s: a.count(s) / len(a) for s in ("train", "val", "test")}
    assert 0.6 < frac["train"] < 0.8
    assert 0.07 < frac["val"] < 0.25
    assert 0.07 < frac["test"] < 0.25


def test_stratified_group_split_every_class_gets_val_and_test():
    from nhandienqua.data.split import stratified_group_split

    # lớp lớn 300 nhóm + lớp nhỏ chỉ 10 nhóm (×10 ảnh) — chia toàn cục dễ làm lớp nhỏ thiếu test
    keys = [f"big{i}" for i in range(300)] + [f"small{i // 10}" for i in range(100)]
    strata = ["big"] * 300 + ["small"] * 100
    s = stratified_group_split(keys, strata, val=0.15, test=0.15, seed=0)
    small = [x for x, st in zip(s, strata) if st == "small"]
    assert {"train", "val", "test"} <= set(small)
    by_group = {}
    for k, x in zip(keys, s):
        by_group.setdefault(k, set()).add(x)
    assert all(len(v) == 1 for v in by_group.values())


def test_group_split_small_dataset_still_has_val():
    keys = [f"g{i}" for i in range(6)]
    s = group_split(keys, val=0.15, test=0.15, seed=0)
    assert "val" in s and "train" in s


# ------------------------------------------------------------------ synth
def _cutout(h, w, color):
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = 255
    return rgba


def test_paste_bbox_matches_position():
    bg = np.zeros((100, 100, 3), np.uint8)
    img, boxes = paste_objects(bg, [(_cutout(10, 20, 200), 3, (5, 30))])  # (cutout, class, (x, y))
    assert boxes == [(3, (5, 30, 25, 40))]
    assert img[35, 10].tolist() == [200, 200, 200]
    assert img[0, 0].tolist() == [0, 0, 0]


def test_paste_clips_at_border():
    bg = np.zeros((50, 50, 3), np.uint8)
    _, boxes = paste_objects(bg, [(_cutout(20, 20, 99), 0, (40, 40))])
    assert boxes == [(0, (40, 40, 50, 50))]


def test_paste_occluded_object_box_shrinks_or_drops():
    bg = np.zeros((100, 100, 3), np.uint8)
    objs = [
        (_cutout(20, 40, 50), 0, (10, 10)),   # bị che phần bên phải
        (_cutout(40, 30, 150), 1, (30, 0)),   # đè lên x 30..60
        (_cutout(10, 10, 80), 2, (70, 70)),   # bị che hoàn toàn bởi vật sau
        (_cutout(30, 30, 90), 3, (65, 65)),
    ]
    _, boxes = paste_objects(bg, objs, min_visible=0.4)
    d = dict(boxes)
    assert d[0] == (10, 10, 30, 30)  # phần nhìn thấy còn x 10..30
    assert d[1] == (30, 0, 60, 40)
    assert 2 not in d                 # bị che > 60% → bỏ nhãn
    assert d[3] == (65, 65, 95, 95)
